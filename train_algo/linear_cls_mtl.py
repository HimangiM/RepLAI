#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import builtins
import math
import os
import sys
import time
import logging

import torch
from torch import nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist

import models
import optimizers
import data
from utils import checkpointing, meters_lincls, training_ops, distributed_ops
from torch.utils.tensorboard import SummaryWriter

class ClassifierMTL(nn.Module):
    def __init__(self, backbone, dim: int, num_classes: dict, normalize=True):
        super().__init__()
        self.backbone = backbone
        self.fc = nn.ModuleDict({
            k: nn.Linear(dim, num_classes[k])
            for k in num_classes})
        self.cls_vars = list(num_classes.keys())
        self.normalize = normalize

        for k in self.cls_vars:
            self.fc[k].weight.data.normal_(mean=0.0, std=0.01)
            self.fc[k].bias.data.zero_()

    def forward(self, x):
        with torch.no_grad():
            x = self.backbone(x)
        if self.normalize:
            x = F.normalize(x, p=2, dim=-1)
        return {k: self.fc[k](x.flatten(1, -1)) for k in self.cls_vars}


def main_worker(gpu, ngpus_per_node, args):
    # args.environment.gpu = gpu if ngpus_per_node > 0 else None
    args.environment.gpu = gpu if ngpus_per_node > 0 and args.environment.distributed else None

    # suppress printing if not master
    if args.environment.multiprocessing_distributed and args.environment.gpu != 0:
        def print_pass(*args):
            pass
        builtins.print = print_pass

    # setup distributed environment
    if args.environment.distributed:
        args = distributed_ops.init_distributed_environment(
            gpu, ngpus_per_node, args)

    # Setup logger
    eval_name = args.logging.name + f'_LinCls_{args.optim.method}{args.logging.suffix}_Ep{args.train_algo.resume_epoch}'
    print_path = os.path.join(args.logging.submitit_dir, eval_name, f'rank{args.environment.rank:02d}.out')
    if not args.environment.multiprocessing_distributed or args.environment.gpu == 0:
        os.makedirs(os.path.dirname(print_path), exist_ok=True)

    orig_print = builtins.print
    def new_print(*out, **kwargs):
        if not (args.environment.multiprocessing_distributed
                and args.environment.gpu != 0):
            orig_print(*out, **kwargs)
        open(print_path, 'a').write(' '.join([str(o) for o in out]) + '\n')
    builtins.print = new_print

    # Log config
    def pretty_print(cfg, t=0):
        for k in cfg:
            if isinstance(cfg[k], type(cfg)):
                print(' '*t + k)
                pretty_print(cfg[k], t+2)
            else:
                print(' '*t + k, str(cfg[k]))
    pretty_print(args)

    # Create model, criterion and optimizer
    print("=> creating model '{}'".format(args.backbone.video.arch))
    backbone = models.build_video_model(args.backbone.video, remove_head=True)
    num_classes = {k: args.train_algo.prediction_variables[k].num_classes
                   for k in args.train_algo.prediction_variables}
    model = ClassifierMTL(backbone,
                          dim=args.backbone.video.outp_dim,
                          num_classes=num_classes,
                          normalize=args.train_algo.normalize_feats)

    # freeze all layers but the last fc
    for name, param in model.backbone.named_parameters():
        param.requires_grad = False
    print(model)

    # Load pretrained model
    pretrain_dir = os.path.join(args.logging.ckpt_dir, args.logging.name)
    model_name, last_epoch = '', -1
    if args.train_algo.resume_epoch < 0:
        ckpt_fname = os.path.join(pretrain_dir, 'checkpoint_{:04d}.pth')
        for i in range(500 - 1, -1, -1):
            if os.path.exists(ckpt_fname.format(i)):
                model_name = ckpt_fname.format(i)
                last_epoch = i
                break
    elif args.train_algo.resume_epoch % 1 == 0:
        ckpt_fname = os.path.join(pretrain_dir, 'checkpoint_{:04d}.pth')
        last_epoch = args.train_algo.resume_epoch
        model_name = ckpt_fname.format(last_epoch)
    else:
        ckpt_fname = os.path.join(pretrain_dir, 'checkpoint_{:010.4f}.pth')
        last_epoch = args.train_algo.resume_epoch
        model_name = ckpt_fname.format(last_epoch)

    if model_name != '':
        if not os.path.isfile(model_name):
            print(f'{model_name} does not exist')
            sys.exit(0)
        print("=> loading checkpoint '{}'".format(model_name))
        checkpoint = torch.load(model_name, map_location="cpu")

        # find state
        state_dict = checkpoint['model']
        prefixes = ['module.backbone.video.', 'module.backbone.', 'backbone.video.', 'backbone.', ]
        for k in list(state_dict.keys()):
            # retain only encoder_q up to before the embedding layer
            for pref in prefixes:
                if k.startswith(pref):
                    state_dict[k[len(pref):]] = state_dict[k]
                    break
            # delete renamed or unused k
            del state_dict[k]

        msg = model.backbone.load_state_dict(state_dict, strict=False)
        assert len(msg.missing_keys) == 0
        print("=> loaded pre-trained model '{}'".format(model_name))
    else:
        print("=> no checkpoint found")
    model = distributed_ops.send_to_device(model, args.environment.distributed, args.environment.gpu)

    print("=> creating optimizer '{}'".format(args.optim.method))
    parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
    optimizer = optimizers.build_optimizer(parameters, args.optim)
    print(str(optimizer)+'\n')

    # Create data loaders
    print("=> creating dataloaders '{}'".format(args.data.name))
    args.data.args.num_clips = 1
    loaders = data.build_video_data_loaders(
        cfg=args.data,
        augm_cfg=args.data_augm,
        batch_size=args.optim.batch_size,
        workers=args.environment.workers,
        distributed=args.environment.distributed,
    )
    for k in loaders:
        print(k, len(loaders[k]), loaders[k].dataset.num_videos, loaders[k].dataset)

    # Resume from a checkpoint
    log_string = args.logging.name + f'_LinCls_{args.optim.method}{args.logging.suffix}_Ep{last_epoch}'
    modules = {'model': model, 'optimizer': optimizer, 'sampler': loaders['train'].batch_sampler}
    ckp_manager = checkpointing.CheckpointManager(
        modules=modules,
        ckpt_dir=os.path.join(args.logging.result_dir, log_string),
        epoch_size=len(loaders['train']),
        save_freq_mints=args.logging.save_freq_mints,
    )
    if args.environment.resume:
        args.optim.start_epoch = ckp_manager.resume()
    # import ipdb; ipdb.set_trace()
    # Create TB loggers
    writer = None
    if args.logging.log_tb and (args.environment.gpu is None or args.environment.gpu == 0):
        print('Setting up tb logging')
        logdir = os.path.join(args.logging.tb_dir, args.logging.name + args.logging.suffix + '_LinCLS_2')
        os.makedirs(logdir, exist_ok=True)
        writer = SummaryWriter(logdir)

    cudnn.benchmark = True
    for epoch in range(args.optim.start_epoch, args.optim.epochs):
        print('Epoch {}'.format(epoch))
        run_phase(loaders['train'],
                  model,
                  optimizer,
                  ckp_manager,
                  epoch,
                  args,
                  train_mode=True,
                  writer=writer)

        if epoch % args.logging.eval_freq == 0 or epoch == args.optim.epochs - 1:
            with torch.no_grad():
                stats = run_phase(
                    loaders['eval'],
                    model,
                    optimizer,
                    ckp_manager,
                    epoch,
                    args,
                    train_mode=False,
                    writer=None)

            if writer is not None:
                for k, v in stats.items():
                    writer.add_scalar(f'Eval/{k}', v, (epoch+1)*len(loaders['train']))
                sys.stdout.flush()

            stats['epoch'] = epoch+1
            stats['batch_i'] = 0
            ckp_manager.checkpoint(
                epoch=epoch+1,
                save_dict=stats)

    stats['epoch'] = args.optim.epochs
    stats['batch_i'] = 0
    ckp_manager.force_checkpoint(
        ckpt_fname='checkpoint.pth',
        save_dict=stats)


def run_phase(loader,
              model,
              optimizer,
              ckp_manager,
              epoch,
              args,
              train_mode=True,
              writer=None):
    cls_vars = list(dict(args.train_algo.prediction_variables).keys())

    # Setup meters
    phase = 'Train' if train_mode else 'Eval'
    batch_time = meters_lincls.AverageMeter(f'Time', ':6.3f', tbname=f'{phase}/Time')
    data_time = meters_lincls.AverageMeter(f'Data', ':6.3f', tbname=f'{phase}/Data')
    acc1_mtrs = {k: meters_lincls.AverageMeter(f'{k} Acc1', ':3.2f', tbname=f'{phase}/{k} Acc1')
                 for k in cls_vars}
    acc5_mtrs = {k: meters_lincls.AverageMeter(f'{k} Acc5', ':3.2f', tbname=f'{phase}/{k} Acc5')
                 for k in cls_vars}
    loss_mtrs = {k: meters_lincls.AverageMeter(f'{k} Loss', ':.4e', tbname=f'{phase}/{k} Loss')
                 for k in cls_vars}
    lr = meters_lincls.AverageMeter(f'LR', ':1.5f', tbname=f'{phase}/LR')
    progress = meters_lincls.ProgressMeter(
        len(loader), [batch_time, data_time, lr] + list(acc1_mtrs.values()) + list(acc5_mtrs.values()) + list(loss_mtrs.values()),
        prefix=f"[{phase}][{epoch}]", tbwriter=writer)

    # Make sure distributed sampler uses different samples in each process.
    loader.batch_sampler.set_epoch(epoch=epoch)
    training_ops.adjust_learning_rate(optimizer, epoch, args.optim)
    world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1

    # switch to train/eval mode
    model.train(mode=train_mode)
    for p in model.parameters():
        p.requires_grad = train_mode

    end = time.time()
    criterion = nn.CrossEntropyLoss()
    for data in loader:
        batch_i = loader.batch_sampler.advance_batches_seen() - 1

        # measure data loading time
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
        data_time.update(time.time() - end)
        lr.update(optimizer.param_groups[0]["lr"])
        # end = time.time()
        # progress.display(batch_i)
        # continue

        # prepare data
        video = data['video']
        labels = {k: data[k] for k in cls_vars}
        if args.environment.gpu is not None:
            video = video.cuda(args.environment.gpu, non_blocking=True)
            labels = {k: labels[k].cuda(args.environment.gpu, non_blocking=True) for k in labels}
        else:
            video = video.cuda(non_blocking=True)
            labels = {k: labels[k].cuda(non_blocking=True) for k in labels}

        # compute output and loss
        logits = model(video)
        total_loss = 0.
        for k in cls_vars:
            # print (torch.max(logits[k], dim=1)[1], labels[k])
            loss = criterion(logits[k], labels[k])
            loss_mtrs[k].update(loss.item(), video.size(0))
            acc1, acc5 = meters_lincls.accuracy(logits[k], labels[k], topk=(1,5))
            acc1_mtrs[k].update(acc1.item(), video.size(0))
            acc5_mtrs[k].update(acc5.item(), video.size(0))
            total_loss += loss
        total_loss /= float(len(cls_vars))

        # for k in cls_vars:
        #     _, res = logits[k].topk(1, 1, True, True)
        #     lab = torch.unsqueeze(labels[k], dim=1)
        #     final_res = torch.cat((res, lab), dim=1)

        # compute gradient and do SGD step
        if train_mode:
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if batch_i % args.logging.print_freq == 0:
            # import ipdb; ipdb.set_trace()
            tb_step = ((epoch * len(loader.dataset) // args.optim.batch_size) + batch_i * world_size)
            progress.display(batch_i)
            progress.tbwrite(tb_step)
            # print ('Tb step: {}, batch_i: {}, epoch: {}, len of loader.dataset: {}, '
            #        'args.optim.batch:{}, world_size: {}'.format(tb_step, batch_i, epoch, len(loader.dataset),
            #                                                     args.optim.batch_size, world_size))
            sys.stdout.flush()
            sys.stdout.flush()

        stats = {}
        for k in cls_vars:
            stats[f"{k}_acc1"] = acc1_mtrs[k].avg
            stats[f"{k}_acc5"] = acc5_mtrs[k].avg
            stats[f"{k}_loss"] = loss_mtrs[k].avg
        stats['epoch'] = epoch
        stats['batch_i'] = batch_i
        if train_mode:
            ckp_manager.checkpoint(
                epoch=epoch,
                batch_i=batch_i,
                save_dict=stats
            )
        # break

    progress.display(len(loader))
    sys.stdout.flush()
    return stats
