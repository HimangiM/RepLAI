import os
import re
from typing import Any, Callable, Optional
from data.video_paths import VideoPaths
from data.video_dataset import VideoDataset

YODA_PATH = '/glusterfs/pmorgado/datasets/charades'
YODA_LOCAL_PATH = '/nvme-scratch/pmorgado/datasets/charades'


def parse_label_list(fn):
    labels = {}
    for l in open(fn):
        l = l.strip()
        code = l[:4]
        desc = l[4:]
        labels[code] = desc
    return labels


class Charades(VideoDataset):
    def __init__(
            self,
            subset: str = 'train',
            base_path: str = YODA_PATH,
            clip_sampling: str = 'random',
            video_duration: float = 0.5,
            audio_duration: float = 2.0,
            data_dir: str = 'data',
            transform: Optional[Callable[[dict], Any]] = None,
            decode_audio: bool = True
    ) -> None:
        anno_dir = f'{base_path}/annotations'
        classes_fn = f'{anno_dir}/Charades_v1_classes.txt'
        self.classes = parse_label_list(classes_fn)

        # Filter subset
        import csv
        subset_fn = f'{anno_dir}/Charades_v1_{subset}.csv'
        data = list(csv.DictReader(open(subset_fn)))

        sample_meta = []
        for m in data:
            if not m['actions']:
                continue
            actions = m['actions'].split(';')
            for a in actions:
                action_lbl, ts, tf = a.split()
                sample_meta.append(dict(filename=f"{m['id']}.mp4",
                                        person_id=f"{m['subject']}",
                                        timestamp=(float(ts), float(tf)),
                                        action_label=action_lbl))

        # Create dataset
        path_prefix = f'{base_path}/{data_dir}'
        video_paths = VideoPaths(sample_metadata=sample_meta, path_prefix=path_prefix)
        super().__init__(
            video_info=video_paths,
            clip_sampling=clip_sampling,
            video_duration=video_duration,
            audio_duration=audio_duration,
            transform=transform,
            decode_audio=decode_audio)

    def copy2local(self, local_path=YODA_LOCAL_PATH):
        # Only runs from a single GPU in distributed settings.
        assert local_path != self._video_info._path_prefix
        import torch.distributed as dist
        if dist.is_initialized() and dist.get_rank() != 0:
            return
        os.system(f"mkdir -p {local_path} && gsutil -m -q rsync -r {self._video_info._path_prefix} {local_path}")
        self._video_info._path_prefix = local_path

    def __len__(self):
        return self.num_videos


if __name__ == '__main__':
    import time
    import argparse
    from utils.meters import AverageMeter

    parser = argparse.ArgumentParser()
    parser.add_argument("--clip_duration", default=2, type=float)
    args = parser.parse_args()

    dataset = Charades(
        subset='train',
        clip_sampling='timestamps',
        decode_audio=True,
        base_path='/Users/morgado/Projects/datasets/charades'
    )

    timer = AverageMeter('time', fmt=':.2f')
    t = time.time()
    for it, sample in enumerate(dataset):
        timer.update(time.time() - t)
        t = time.time()
        if it % 20 == 0:
            print(timer)
