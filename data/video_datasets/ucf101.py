import os
from typing import Any, Callable, Optional
from data.video_paths import VideoPaths
from data.video_dataset import VideoDataset

YODA_PATH = '/glusterfs/pmorgado/datasets/ucf101'
YODA_LOCAL_PATH = '/nvme-scratch/pmorgado/datasets/ucf101'


class UCF101(VideoDataset):
    def __init__(
            self,
            subset: str = 'train-1',
            base_path: str = YODA_PATH,
            clip_sampling: str = 'random',
            video_duration: float = 0.5,
            audio_duration: float = 2.,
            num_clips: int = 1,
            data_dir: str = '360p_16fps',
            transform: Optional[Callable[[dict], Any]] = None,
            decode_audio: bool = True
    ) -> None:
        split_dir = f'{base_path}/ucfTrainTestlist'
        classes_fn = f'{split_dir}/classInd.txt'
        self.classes = sorted([l.strip().split()[1] for l in open(classes_fn)])

        # Create dataset
        subset_fn = f'{split_dir}/{subset}.txt'
        filenames = [l.strip().split()[0] for l in open(subset_fn)]
        sample_metadata = [{'filename': fn, 'label': self.classes.index(fn.split('/')[0])}
                           for fn in filenames]
        path_prefix = f'{base_path}/{data_dir}'
        video_info = VideoPaths(sample_metadata=sample_metadata, path_prefix=path_prefix)
        super().__init__(
            video_info=video_info,
            clip_sampling=clip_sampling,
            video_duration=video_duration,
            audio_duration=audio_duration,
            num_clips=num_clips,
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
    dataset = UCF101(
        subset='trainlist01',
        clip_sampling='random',
        decode_audio=True,
    )

    import time
    from utils.meters import AverageMeter
    timer = AverageMeter('time', fmt=':.2f')
    t = time.time()
    for it, sample in enumerate(dataset):
        timer.update(time.time() - t)
        t = time.time()
        if it % 20 == 0:
            print(timer)
