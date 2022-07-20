import glob
import os
import torch
import re
import numpy as np
import time
import torch.distributed as dist


def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)


def resume_from_checkpoint(ckpt_fname, modules):
    print(">>>> loading checkpoint '{}'".format(ckpt_fname))
    checkpoint = torch.load(ckpt_fname, map_location='cpu')

    # Load state dict
    for k in modules:
        modules[k].load_state_dict(checkpoint[k])
        del checkpoint[k]
    print(">>>> loaded checkpoint '{}' (epoch {})".format(
        ckpt_fname, checkpoint['epoch']))
    return checkpoint


def resume(modules, args):
    all_ckpt_fnames = glob.glob(os.path.join(args.logging.ckpt_dir, args.logging.name, 'checkpoint_*.pth'))
    if not all_ckpt_fnames:
        return

    # Find last checkpoint
    epochs = [float(re.match('checkpoint_(\d+\.*\d*).pth', fn.split('/')[-1]).group(1)) for fn in all_ckpt_fnames]
    ckpt_fname = all_ckpt_fnames[np.argsort(-np.array(epochs))[-1]]

    # Load checkpoint
    resume_from_checkpoint(ckpt_fname, modules)


class CheckpointManager:
    def __init__(self, modules, ckpt_dir, epoch_size, save_freq_epoch=None, save_freq_mints=None):
        self.modules = modules
        self.ckpt_dir = ckpt_dir
        self.epoch_size = epoch_size
        self.save_freq = save_freq_epoch
        self.save_freq_mints = save_freq_mints
        self.retain_num_ckpt = 0

        self.time = time.time()
        self.distributed = dist.is_available() and dist.is_initialized()
        self.world_size = dist.get_world_size() if self.distributed else 1
        self.rank = dist.get_rank() if self.distributed else 0

        os.makedirs(self.ckpt_dir, exist_ok=True)

    def resume(self):
        ckpt_fname = os.path.join(self.ckpt_dir, 'checkpoint_latest.pth')
        start_epoch = 0
        if os.path.isfile(ckpt_fname):
            checkpoint = torch.load(ckpt_fname, map_location='cpu')

            # Load state dict
            for k in self.modules:
                self.modules[k].load_state_dict(checkpoint[k])
            start_epoch = checkpoint['epoch']
            print("=> loaded checkpoint '{}' (epoch {})".format(
                    ckpt_fname, checkpoint['epoch']))
        return start_epoch

    def timed_checkpoint(self, save_dict=None):
        if self.save_freq_mints is None or self.save_freq_mints <= 0:
            return
        t = time.time() - self.time
        t_all = [t for _ in range(self.world_size)]
        if self.world_size > 1:
            dist.all_gather_object(t_all, t)
        if min(t_all) > self.save_freq_mints * 60:
            self.time = time.time()
            ckpt_fname = os.path.join(self.ckpt_dir, 'checkpoint_latest.pth')

            state = self.create_state_dict(save_dict)
            if self.rank == 0:
                save_checkpoint(state, filename=ckpt_fname)
                print(f"Saved checkpoint '{ckpt_fname}")

    def midway_epoch_checkpoint(self, epoch, batch_i, save_dict=None):
        if self.save_freq is None:
            return
        if ((batch_i + 1) / float(self.epoch_size) %
                self.save_freq) < (
                    batch_i / float(self.epoch_size) %
                    self.save_freq):
            ckpt_fname = os.path.join(self.ckpt_dir, 'checkpoint_{:010.4f}.pth')
            ckpt_fname = ckpt_fname.format(
                epoch + batch_i / float(self.epoch_size))

            state = self.create_state_dict(save_dict)
            if self.rank == 0:
                save_checkpoint(state, filename=ckpt_fname)
                print(f"Saved checkpoint '{ckpt_fname}' (epoch {epoch}, batch_i {batch_i})")

    def end_epoch_checkpoint(self, epoch, save_dict=None):
        if self.save_freq is None:
            return
        if (epoch % self.save_freq == 0) or self.save_freq < 1:
            ckpt_fname = os.path.join(self.ckpt_dir, 'checkpoint_{:04d}.pth')
            ckpt_fname = ckpt_fname.format(epoch + 1)

            state = self.create_state_dict(save_dict)
            if self.rank == 0:
                save_checkpoint(state, filename=ckpt_fname)
                print(f"Saved checkpoint '{ckpt_fname}'  (epoch {epoch})")

            if self.retain_num_ckpt > 0:
                ckpt_fname = os.path.join(self.ckpt_dir, 'checkpoint_{:04d}.pth')
                ckpt_fname = ckpt_fname.format(epoch - self.save_freq * (self.retain_num_ckpt + 1))
                if os.path.exists(ckpt_fname):
                    os.remove(ckpt_fname.format(ckpt_fname))

    def create_state_dict(self, save_dict):
        state = {k: self.modules[k].state_dict()
                 for k in self.modules}
        if save_dict is not None:
            state.update(save_dict)
        return state

    def checkpoint(self, epoch, batch_i=None, save_dict=None):
        if batch_i is None:
            self.end_epoch_checkpoint(epoch, save_dict)
        else:
            self.timed_checkpoint(save_dict)
            self.midway_epoch_checkpoint(epoch, batch_i, save_dict=save_dict)

    def force_checkpoint(self, ckpt_fname, save_dict=None):
        ckpt_fname = os.path.join(self.ckpt_dir, ckpt_fname)
        state = self.create_state_dict(save_dict)
        if self.rank == 0:
            save_checkpoint(state, filename=ckpt_fname)
            print(f"Saved checkpoint '{ckpt_fname}'")
