#!/usr/bin/env python
import builtins
import math
import os
import sys
import time

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torchvision.utils
from PIL import Image

import models
import criterions
import optimizers
import data
from utils import checkpointing, meters, training_ops, distributed_ops
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def main_worker(gpu, ngpus_per_node, args):
    args.environment.gpu = gpu if ngpus_per_node > 0 and args.environment.distributed else None

    # suppress printing if not master
    if args.environment.multiprocessing_distributed and args.environment.gpu != 0:
        def print_pass(*args):
            pass
        builtins.print = print_pass

    # setup distributed environment
    if args.environment.distributed:
        args = distributed_ops.init_distributed_environment(
            gpu, ngpus_per_node, args
        )
    def pretty_print(cfg, t=0):
        for k in cfg:
            if isinstance(cfg[k], type(cfg)):
                print(' '*t + k)
                pretty_print(cfg[k], t+2)
            else:
                print(' '*t + k, str(cfg[k]))
    pretty_print(args)

    # Create model, criterion and optimizer
    backbone = models.build_audio_video_model(args.backbone, remove_head=True)
    model_args = dict(args.criterion.args)
    model_args['backbone'] = backbone
    model_args['in_dim'] = args.backbone.video.outp_dim
    model_args['num_clips'] = args.train_algo.num_clips
    model = criterions.build_criterion(
        name=args.criterion.name,
        args=model_args)
    model = distributed_ops.send_to_device(
        model,
        distributed=args.environment.distributed,
        device=args.environment.gpu,
    )
    print(f"=> model: video '{args.backbone.video.arch}'")
    print(model)

    # params = [{'params': model.get_backbone_params(), 'lr': args.optim.args.lr/10},
    #               {'params': model.get_head_params()}]

    # params = list(model.parameters())
    model_without_ddp = model if not args.environment.distributed else model.module
    params = [{'params': model_without_ddp.mlp.parameters()},
              {'params': model_without_ddp.backbone.parameters(), 'lr': args.optim.args.lr * args.optim.backbone_lr_mult}]
    optimizer = optimizers.build_optimizer(params, args.optim)
    print(f"=> optimizer '{args.optim.method}'\n" + str(optimizer)+'\n')
    if args.optim.use_lr_scheduler:
        lr_scheduler = torch.optim.lr_scheduler.__dict__[args.optim.lr_scheduler](
            optimizer, **args.optim.lr_scheduler_args
        )
        print(f"=> using lr scheduler '{args.optim.lr_scheduler}'\n" + str(lr_scheduler)+'\n')
    else:
        lr_scheduler = None
    # Create data loaders
    print(f"=> creating dataloaders '{args.data.name}'")
    augm_args = {
        # 'num_clips': args.train_algo.num_clips,
        # 'clip_duration': args.data.args.video_duration,
        # 'transform': args.data_augm.video.name,
        # 'transform_args': args.data_augm.video.args,
    }
    args.data.args.num_clips = 1
    args.data.args.decode_audio = True
    # args.data.args.video_duration = args.train_algo.segment_duration
    # args.data.args.audio_duration = args.train_algo.segment_duration
    # args.data_augm.video.name = 'UniformClips'
    # args.data_augm.video.args = augm_args
    augm_args_audio = args.data_augm.audio.args
    augm_args_audio.num_temporal_frames = int(1000 * args.data.args.audio_duration / augm_args_audio.mel_step_size)
    loaders = data.build_video_data_loaders(
        args.data,
        augm_cfg=args.data_augm,
        batch_size=args.optim.batch_size,
        workers=args.environment.workers,
        distributed=args.environment.distributed,
    )
    for k in loaders:
        assert len(loaders[k]) > 0
        print(k, len(loaders[k]), loaders[k].dataset.num_videos, loaders[k].dataset)

    # Optionally resume from a checkpoint
    modules = {'model': model, 'optimizer': optimizer, 'sampler': loaders['train'].batch_sampler}
    ckp_manager = checkpointing.CheckpointManager(
        modules=modules,
        ckpt_dir=os.path.join(args.logging.ckpt_dir, args.logging.name),
        epoch_size=len(loaders['train']),
        save_freq_mints=args.logging.save_freq_mints,
        save_freq_epoch=args.logging.save_freq,
    )
    if args.environment.resume:
        args.optim.start_epoch = ckp_manager.resume()

    # Create TB loggers
    writer = None
    if args.logging.log_tb and (args.environment.gpu == 0 or args.environment.gpu is None):
        logdir = os.path.join(args.logging.tb_dir, args.logging.name + args.logging.suffix)
        os.makedirs(logdir, exist_ok=True)
        writer = SummaryWriter(logdir)

    cudnn.benchmark = True

    for epoch in range(args.optim.start_epoch, args.optim.epochs):
        print('Epoch {}'.format(epoch))
        train(loaders['train'],
              model,
              optimizer,
              ckp_manager,
              epoch,
              args,
              writer=writer,
              lr_scheduler=lr_scheduler)

        # Checkpoint
        ckp_manager.checkpoint(
            epoch=epoch+1,
            save_dict={'epoch': epoch+1})


def train(loader, model, optimizer,
          ckp_manager, epoch, args, writer=None, lr_scheduler=None):

    # Setup meters
    phase = 'Train'
    batch_time = meters.AverageMeter(f'Time', ':6.3f')
    data_time = meters.AverageMeter(f'Data', ':6.3f')
    lr_time = meters.AverageMeter(f'Learning Rate', ':6.3f')
    grad_time = meters.AverageMeter(f'Gradient', ':6.3f')
    clr_loss_mtr = meters.AverageMeter(f'CLR Loss', ':.4e')
    aot_loss_mtr = meters.AverageMeter(f'AoT Loss', ':.4e')
    loss_mtr = meters.AverageMeter(f'{phase} Loss', ':.4e')
    progress = meters.ProgressMeter(
        len(loader), [batch_time, data_time, lr_time, grad_time, clr_loss_mtr, aot_loss_mtr, loss_mtr],
        prefix=f"[{phase}][{epoch}]", tbwriter=writer)

    # Make sure distributed sampler uses different samples in each process.
    loader.batch_sampler.set_epoch(epoch=epoch)
    if lr_scheduler:
        lr_scheduler.step()
    else:
        training_ops.adjust_learning_rate(optimizer, epoch, args.optim)
    world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1

    # switch to train mode
    model.train()

    end = time.time()
    for data in loader:
        batch_i = loader.batch_sampler.advance_batches_seen() - 1
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
        data_time.update(time.time() - end)

        # prepare data
        # clips_l = 1 x B x 3 x 8 x 112 x 112, clips_r = 1 x B x 3 x 8 x 112 x 112
        # audio = 1 x 16 x 1 x 562 x 80
        clips_l, clips_r, audio = data['video_l'], data['video_r'], data['audio']
        if args.environment.gpu is not None:
            clips_l = [x.cuda(args.environment.gpu, non_blocking=True) for x in clips_l]
            clips_r = [x.cuda(args.environment.gpu, non_blocking=True) for x in clips_r]
            audio = [x.cuda(args.environment.gpu, non_blocking=True) for x in audio]
        # else:
        #     clips_l = [x.cuda(non_blocking=True) for x in clips_l]
        #     clips_r = [x.cuda(non_blocking=True) for x in clips_r]
        #     audio = audio[0].cuda(non_blocking=True)

        # compute output and loss
        loss, stats = model(clips_l, clips_r, audio)
        loss_mtr.update(loss.item(), clips_l[0].shape[0])
        clr_loss_mtr.update(stats['clr_loss'].item(), clips_l[0].shape[0])
        aot_loss_mtr.update(stats['aot_loss'].item(), clips_l[0].shape[0])

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()

        if args.optim.grad_clipping and args.optim.grad_norm:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.optim.grad_clip_val)
        if args.optim.grad_clipping and args.optim.grad_val:
            torch.nn.utils.clip_grad_value_(model.parameters(), args.optim.grad_clip_val)

        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        lr = optimizer.param_groups[0]["lr"]
        lr_time.update(lr)

        with torch.no_grad():
            norm_type = 2
            total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type) for p in model.parameters()]), norm_type)
            grad_time.update(total_norm)

        if batch_i % args.logging.print_freq == 0:
            tb_step = (
                    epoch * len(loader.dataset) // args.optim.batch_size +
                    batch_i * world_size)
            progress.display(batch_i)
            progress.tbwrite(tb_step)

        # Checkpoint
        ckp_manager.checkpoint(
            epoch=epoch,
            batch_i=batch_i,
            save_dict={'epoch': epoch}
        )
