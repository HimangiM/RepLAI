import torch
from data import video_datasets
from data import video_transforms
from data import audio_transforms
from data.resumable_samplers import ResumableBatchSampler
from torch.utils.data import DataLoader


class Transform:
    def __init__(self, transforms, data_shapes):
        self.transforms = transforms
        self.data_shapes = data_shapes

    def __call__(self, x):
        for k in self.transforms:
            if k not in x or x[k] is None:
                x[k] = torch.zeros(tuple(self.data_shapes[k])).float()
            else:
                x[k] = self.transforms[k](x[k][0], x[k][1])
        return x


def build_transforms(cfg, augment):
    transforms, data_shapes = {}, {}
    for k in cfg:
        if cfg[k].name in vars(video_transforms):
            transforms[k] = video_transforms.__dict__[cfg[k].name](
                **cfg[k].args, augment=augment)
        elif cfg[k].name in vars(audio_transforms):
            transforms[k] = audio_transforms.__dict__[cfg[k].name](
                **cfg[k].args, augment=augment)
        else:
            raise NotImplementedError(f"Transform {cfg.name} not found.")
        data_shapes[k] = cfg[k].data_shape
    return Transform(transforms, data_shapes)


def build_video_dataset(cfg, subset, transform=None):
    if cfg.name in vars(video_datasets):
        dataset = video_datasets.__dict__[cfg.name](
            transform=transform,
            subset=subset,
            **cfg.args
        )
    else:
        raise NotImplementedError(f"Dataset {cfg.dataset.name} not found.")

    return dataset


def build_video_data_loaders(cfg, augm_cfg, batch_size, workers=0, distributed=True):
    loaders = {}
    for mode in cfg.splits:
        augment = mode == 'train'

        transform = build_transforms(augm_cfg, augment=augment)
        db = build_video_dataset(cfg, subset=cfg.splits[mode], transform=transform)
        batch_sampler = ResumableBatchSampler(
            batch_size, db=db,
            distributed=distributed,
            drop_last=True
        )
        loaders[mode] = DataLoader(
            db,
            batch_sampler=batch_sampler,
            num_workers=workers,
            pin_memory=True,
            persistent_workers=False
        )

    return loaders
