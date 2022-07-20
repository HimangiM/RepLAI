import os
import re
from typing import Any, Callable, Optional
from data.video_paths import VideoPaths
from data.video_dataset import VideoDataset

YODA_PATH = '/glusterfs/pmorgado/datasets/ava'
YODA_LOCAL_PATH = '/nvme-scratch/pmorgado/datasets/ava'


def parse_label_list(fn):
    content = " ".join([l.strip() for l in open(fn)])
    labels = {}
    while True:
        m = re.match('\s*item \{ name: "([\w/\(\)\.,\s]+)" id: (\d+) \}', content)
        if m is None:
            break
        labels[int(m.group(2)) - 1] = m.group(1)    # 0-index
        content = content[len(m.group(0)):]
    return labels


class AVA(VideoDataset):
    def __init__(
            self,
            subset: str = 'train',
            base_path: str = YODA_PATH,
            clip_sampling: str = 'random',
            video_duration: float = 0.5,
            audio_duration: float = 2.0,
            data_dir: str = '256p_16fps',
            transform: Optional[Callable[[dict], Any]] = None,
            decode_audio: bool = True
    ) -> None:
        anno_dir = f'{base_path}/annotations'
        classes_fn = f'{anno_dir}/ava_action_list_v2.1_for_activitynet_2018.pbtxt'
        self.classes = parse_label_list(classes_fn)

        # Filter subset
        sample_meta = []
        for ln in open(f'{anno_dir}/ava_{subset}_v2.1.csv'):
            row = ln.strip().split(',')
            yid = row[0]
            ts = int(row[1]) - 15*60     # assumes videos have been trimed to start at t=15min
            bbox = [float(c) for c in row[2:6]]
            action_label = int(row[6]) - 1
            person_id = int(row[7])
            sample_meta.append(dict(filename=f"{yid}.mp4", video_id=yid, timestamp=ts, bbox=bbox, action_label=action_label, person_id=person_id))

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

    dataset = AVA(
        subset='train',
        clip_sampling='timestamps',
        decode_audio=args.decode_audio,
        base_path='/Users/morgado/Projects/datasets/ava'
    )

    timer = AverageMeter('time', fmt=':.2f')
    t = time.time()
    for it, sample in enumerate(dataset):
        timer.update(time.time() - t)
        t = time.time()
        if it % 20 == 0:
            print(timer)
