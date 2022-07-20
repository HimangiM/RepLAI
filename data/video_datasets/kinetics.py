import os
from typing import Any, Callable, Optional
from data.video_paths import VideoPaths
from data.video_dataset import VideoDataset


YODA_PATH = '/nvme-scratch/spurushw/kinetics'
YODA_LOCAL_PATH = '/nvme-scratch/spurushw/kinetics'


class Kinetics(VideoDataset):
    def __init__(
            self,
            subset: str = 'train',
            base_path: str = YODA_PATH,
            clip_sampling: str = 'random',
            video_duration: float = 0.5,
            audio_duration: float = 2.,
            num_clips: int = 1,
            transform: Optional[Callable[[dict], Any]] = None,
            decode_audio: bool = True
    ) -> None:
        data_root = f'{base_path}/data/{subset}_256'
        video_paths = VideoPaths.from_directory(data_root)
        super().__init__(
            video_info=video_paths,
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


if __name__ == '__main__':
    dataset = Kinetics(
        subset='train',
        clip_sampling='random',
    )

    import time
    from utils.meters import AverageMeter
    timer = AverageMeter('time', fmt=':.2f')
    t = time.time()
    for it, sample in enumerate(dataset):
        timer.update(time.time() - t)
        t = time.time()
        if it % 1 == 0:
            print(timer)
