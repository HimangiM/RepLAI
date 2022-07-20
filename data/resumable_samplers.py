from typing import Optional, Sized
import math
import torch
from torch import Generator
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.dataset import Dataset
from torch.utils.data import Sampler, RandomSampler


def default_sampler(db, distributed=False):
    if distributed:
        return DistributedSampler(
            dataset=db,
            shuffle=True,
            drop_last=True,
        )
    else:
        return RandomSampler(db)


class ResumableBatchSampler(Sampler):
    def __init__(self,
                 batch_size: int,
                 db: Dataset = None,
                 sampler: Sampler = None,
                 distributed: bool = False,
                 drop_last: bool = False) -> None:
        self.seed_base = 93823982

        self.batch_size = batch_size
        self.drop_last = drop_last
        self.epoch = 0

        assert db is None or sampler is None
        if sampler is None:
            assert db is not None
            sampler = default_sampler(db, distributed=distributed)
        else:
            assert sampler is not None
        self.sampler = sampler

        self.db_head = 0
        self.num_batches_seen = 0
        self.init_from_ckpt = False

    def state_dict(self):
        return {'num_batches_seen': self.num_batches_seen,
                'epoch': self.epoch}

    def load_state_dict(self, state):
        self.db_head = state['num_batches_seen'] * self.batch_size
        self.num_batches_seen = state['num_batches_seen']
        self.set_epoch(state['epoch'])
        self.init_from_ckpt = True

    def advance_batches_seen(self):
        self.num_batches_seen += 1
        return self.num_batches_seen

    def __iter__(self):
        self._set_seeds()
        indices = list(self.sampler)

        if not self.init_from_ckpt:
            self.db_head = 0
            self.num_batches_seen = 0

        while self.db_head < len(indices):
            batch_idx = indices[self.db_head:self.db_head+self.batch_size]
            self.db_head += len(batch_idx)
            if self.drop_last and len(batch_idx) < self.batch_size:
                break
            yield batch_idx

        self.init_from_ckpt = False

    def __len__(self) -> int:
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        else:
            return math.ceil(len(self.sampler) / self.batch_size)

    def _set_seeds(self):
        if isinstance(self.sampler, RandomSampler):
            self.sampler.generator = Generator()
            self.sampler.generator.manual_seed(self.seed_base + self.epoch)
        elif isinstance(self.sampler, DistributedSampler):
            self.sampler.set_epoch(self.epoch)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch


if __name__ == '__main__':
    from data import builder
    from omegaconf import OmegaConf

    cfg = OmegaConf.load('configs/action-recg/data/hmdb51.yaml')
    cfg.args.base_path = '/Users/morgado/datasets/hmdb51'
    cfg.update(augmentation=OmegaConf.load('configs/action-recg/data/augmentation/resize_crop_flip.yaml'))
    loader = builder.build_video_data_loaders(cfg, 32, 0, False)['train']

    sampler = loader.dataset.video_sampler
    sampler.set_epoch(0)

    dt = []
    for it, batch in enumerate(loader):
        dt += [batch]
        if it == 2:
            state_dict = sampler.state_dict()
        if it == 4:
            break

    sampler.load_state_dict(state_dict)
    sampler.set_epoch(0)

    dt_2 = []
    for it, batch in enumerate(loader):
        dt += [batch]
        if it == 2:
            break

