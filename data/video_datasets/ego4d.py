import os
import json
from datetime import datetime
from typing import Any, Callable, Optional

import torchvision.models

from data.video_paths import VideoPaths, VideoSegmentsPaths
from data.video_dataset import VideoDataset, FramesDataset
from data.video_audiopeak_dataset import VideoAudioPeakDataset
from collections import defaultdict
from data.video_moi_dataset import VideoAudioMoIDataset, FramesMoIDataset
from data.video_dense_data import VideoDenseDataset, FramesDenseDataset

YODA_PATH = '/glusterfs/hmittal/datasets'
YODA_LOCAL_PATH = '/nvme-scratch/pmorgado/datasets/epic-kitchens'

class Ego4D(VideoDataset):
    def __init__(
            self,
            subset: str = 'train',
            base_path: str = YODA_PATH,
            data_dir: str = '360p_16fps',
            clip_sampling: str = 'random',
            video_duration: float = 0.5,
            audio_duration: float = 2.,
            transform: Optional[Callable[[dict], Any]] = None,
            decode_audio: bool = True
    ) -> None:
        import csv
        if subset == 'train':
            subset_path = f"{base_path}/annotations/EPIC_100_train.csv"
        elif subset == 'eval':
            subset_path = f"{base_path}/annotations/EPIC_100_validation.csv"
        elif subset == 'test':
            subset_path = f"{base_path}/annotations/EPIC_100_test_timestamps.csv"
        else:
            raise ValueError(f'Subset {subset} does not exist. Choose train, eval, or test.')

        sample_meta = []

        def timestamp2seconds(timestamp):
            date_time = datetime.strptime(timestamp, "%H:%M:%S.%f")
            a_timedelta = date_time - datetime(1900, 1, 1)
            return a_timedelta.total_seconds()

        for m in csv.DictReader(open(subset_path)):
            sample_meta.append({
                'filename': f"{m['video_id']}.MP4",
                'timestamp': (timestamp2seconds(m['start_timestamp']),
                              timestamp2seconds(m['stop_timestamp'])),
                'narration': m['narration'],
                'verb_class': int(m['verb_class']),
                'noun_class': int(m['noun_class']),
            })

        path_prefix = f'{base_path}/{data_dir}'
        video_paths = VideoPaths(sample_meta, path_prefix=path_prefix)
        video_paths.filter_missing()
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


class Ego4DSegments(VideoDataset):
    def __init__(
            self,
            subset: str = 'train',
            base_path: str = YODA_PATH,
            data_dir: str = 'segments',
            clip_sampling: str = 'random',
            video_duration: float = 0.5,
            audio_duration: float = 2.,
            decode_audio: bool = True,
            num_clips: int = 1,
            transform: Optional[Callable[[dict], Any]] = None,
    ) -> None:
        import csv
        import glob
        import re
        if subset == 'train':
            subset_path = f"{base_path}/annotations/EPIC_100_train.csv"
        elif subset == 'eval':
            subset_path = f"{base_path}/annotations/EPIC_100_validation.csv"
        elif subset == 'test':
            subset_path = f"{base_path}/annotations/EPIC_100_test_timestamps.csv"
        else:
            raise ValueError(f'Subset {subset} does not exist. Choose train, eval, or test.')

        subset_video_ids = set([m['video_id'] for m in csv.DictReader(open(subset_path))])

        path_prefix = f'{base_path}/{data_dir}'
        all_files = glob.glob(f"{path_prefix}/*.MP4")

        sample_meta = []
        for fn in all_files:
            vid = re.match('(P\d+_\d+)_\d+_\d+\.MP4', fn.split('/')[-1]).group(1)
            if vid in subset_video_ids:
                sample_meta.append({
                    'filename': fn.split('/')[-1],
                })
        video_paths = VideoPaths(sample_meta, path_prefix=path_prefix)

        # video_paths.filter_missing()
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


class Ego4DV2(VideoDataset):
    def __init__(
            self,
            subset: str = 'train',
            base_path: str = YODA_PATH,
            data_dir: str = 'segments',
            clip_sampling: str = 'random',
            video_duration: float = 0.5,
            audio_duration: float = 2.,
            num_clips: int = 1,
            transform: Optional[Callable[[dict], Any]] = None,
            decode_audio: bool = True
    ) -> None:
        import csv
        import glob
        import re
        if subset == 'train':
            subset_path = f"{base_path}/annotations/EPIC_100_train.csv"
        elif subset == 'eval':
            subset_path = f"{base_path}/annotations/EPIC_100_validation.csv"
        elif subset == 'test':
            subset_path = f"{base_path}/annotations/EPIC_100_test_timestamps.csv"
        else:
            raise ValueError(f'Subset {subset} does not exist. Choose train, eval, or test.')

        subset_video_ids = set([m['video_id'] for m in csv.DictReader(open(subset_path))])

        path_prefix = f'{base_path}/{data_dir}'
        all_files = glob.glob(f"{path_prefix}/*.MP4")

        sample_meta = {vid: defaultdict(list) for vid in subset_video_ids}
        for fn in all_files:
            vid = re.match('(P\d+_\d+)_\d+_\d+\.MP4', fn.split('/')[-1]).group(1)
            if vid in subset_video_ids:
                sample_meta[vid]['filenames'] += [fn.split('/')[-1]]
        sample_meta = [sample_meta[vid] for vid in sample_meta if sample_meta[vid]['filenames']]
        video_paths = VideoSegmentsPaths(sample_meta, path_prefix=path_prefix)

        # video_paths.filter_missing()
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


class Ego4DActionSegments(VideoDataset):
    def __init__(
            self,
            subset: str = 'train',
            base_path: str = YODA_PATH,
            data_dir: str = 'action_segments',
            clip_sampling: str = 'random',
            video_duration: float = 0.5,
            audio_duration: float = 2.,
            num_clips: int = 1,
            transform: Optional[Callable[[dict], Any]] = None,
            decode_audio: bool = True
    ) -> None:
        import csv
        if subset == 'train':
            subset_path = f"{base_path}/annotations/EPIC_100_train.csv"
        elif subset == 'eval':
            subset_path = f"{base_path}/annotations/EPIC_100_validation.csv"
        elif subset == 'test':
            subset_path = f"{base_path}/annotations/EPIC_100_test_timestamps.csv"
        else:
            raise ValueError(f'Subset {subset} does not exist. Choose train, eval, or test.')

        sample_meta = []
        for m in csv.DictReader(open(subset_path)):
            ss = m['start_timestamp'].replace(':', '').replace('.', '')
            ff = m['stop_timestamp'].replace(':', '').replace('.', '')
            sample_meta.append({
                'filename': f"{m['video_id']}_{ss}_{ff}.MP4",
                'narration': m['narration'],
                'verb_class': int(m['verb_class']),
                'noun_class': int(m['noun_class']),
            })
        print('meta', len(sample_meta))

        path_prefix = f'{base_path}/{data_dir}'
        video_paths = VideoPaths(sample_meta, path_prefix=path_prefix)
        video_paths.filter_missing()
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


class Ego4DActionSegmentsFrames(FramesDataset):
    def __init__(
            self,
            subset: str = 'train',
            base_path: str = YODA_PATH,
            data_dir: str = 'action_segments',
            clip_sampling: str = 'random',
            audio_duration: float = 2.,
            transform: Optional[Callable[[dict], Any]] = None,
            decode_audio: bool = True
    ) -> None:
        import csv
        if subset == 'train':
            subset_path = f"{base_path}/annotations/EPIC_100_train.csv"
        elif subset == 'eval':
            subset_path = f"{base_path}/annotations/EPIC_100_validation.csv"
        elif subset == 'test':
            subset_path = f"{base_path}/annotations/EPIC_100_test_timestamps.csv"
        else:
            raise ValueError(f'Subset {subset} does not exist. Choose train, eval, or test.')

        sample_meta = []
        for m in csv.DictReader(open(subset_path)):
            ss = m['start_timestamp'].replace(':', '').replace('.', '')
            ff = m['stop_timestamp'].replace(':', '').replace('.', '')
            sample_meta.append({
                'filename': f"{m['video_id']}_{ss}_{ff}.MP4",
                'narration': m['narration'],
                'verb_class': int(m['verb_class']),
                'noun_class': int(m['noun_class']),
            })
        print('meta', len(sample_meta))

        path_prefix = f'{base_path}/{data_dir}'
        video_paths = VideoPaths(sample_meta, path_prefix=path_prefix)
        video_paths.filter_missing()
        super().__init__(
            video_info=video_paths,
            clip_sampling=clip_sampling,
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


class Ego4DAudioPeakSegments(VideoAudioPeakDataset):
    def __init__(
            self,
            subset: str = 'train',
            base_path: str = YODA_PATH,
            data_dir: str = 'segments',
            video_duration: float = 0.5,
            audio_duration: float = 2.,
            decode_audio: bool = True,
            num_clips: int = 1,
            transform: Optional[Callable[[dict], Any]] = None,
            delta_non_overlap: float = 0.,
            audio_event_prominence: float = 1.,
    ) -> None:
        import csv
        import glob
        import re

        all_files = glob.glob(f"{base_path}/*.mp4")

        sample_meta = []
        for fn in all_files:
            sample_meta.append({
                'filename': fn.split('/')[-1],
            })
        video_paths = VideoPaths(sample_meta, path_prefix=base_path)

        # video_paths.filter_missing()
        super().__init__(
            video_info=video_paths,
            video_duration=video_duration,
            audio_duration=audio_duration,
            num_clips=num_clips,
            transform=transform,
            decode_audio=decode_audio,
            delta_non_overlap=delta_non_overlap,
            audio_event_prominence=audio_event_prominence)

    def copy2local(self, local_path=YODA_LOCAL_PATH):
        # Only runs from a single GPU in distributed settings.
        assert local_path != self._video_info._path_prefix
        import torch.distributed as dist
        if dist.is_initialized() and dist.get_rank() != 0:
            return
        os.system(f"mkdir -p {local_path} && gsutil -m -q rsync -r {self._video_info._path_prefix} {local_path}")
        self._video_info._path_prefix = local_path

class Ego4DMoI(VideoAudioMoIDataset):
    def __init__(
            self,
            prob_moi,
            subset: str = 'train',
            base_path: str = YODA_PATH,
            data_dir: str = 'segments',
            video_duration: float = 0.5,
            audio_duration: float = 2.,
            decode_audio: bool = True,
            num_clips: int = 1,
            transform: Optional[Callable[[dict], Any]] = None,
            delta_non_overlap: float = 0.,
            dataset='ego4D'
    ) -> None:
        import csv
        import glob
        import re

        path_prefix = os.path.join(base_path, data_dir)
        all_files = glob.glob(f"{path_prefix}/*.mp4")

        sample_meta = []
        for fn in all_files:
            sample_meta.append({
                'filename': fn.split('/')[-1],
            })
        video_paths = VideoPaths(sample_meta, path_prefix=path_prefix)
        # video_paths.filter_missing()
        super().__init__(
            video_info=video_paths,
            prob_moi=prob_moi,
            video_duration=video_duration,
            audio_duration=audio_duration,
            num_clips=num_clips,
            transform=transform,
            decode_audio=decode_audio,
            delta_non_overlap=delta_non_overlap)

    def copy2local(self, local_path=YODA_LOCAL_PATH):
        # Only runs from a single GPU in distributed settings.
        assert local_path != self._video_info._path_prefix
        import torch.distributed as dist
        if dist.is_initialized() and dist.get_rank() != 0:
            return
        os.system(f"mkdir -p {local_path} && gsutil -m -q rsync -r {self._video_info._path_prefix} {local_path}")
        self._video_info._path_prefix = local_path


class Ego4DFramesMoI(FramesMoIDataset):
    def __init__(
            self,
            prob_moi = None,
            subset: str = 'train',
            base_path: str = YODA_PATH,
            data_dir: str = 'segments_frames',
            audio_duration: float = 2.,
            decode_audio: bool = True,
            transform: Optional[Callable[[dict], Any]] = None,
            delta_non_overlap: float = 0.
    ) -> None:
        import csv
        import glob
        import re
        if subset == 'train':
            subset_path = f"{base_path}/annotations/EPIC_100_train.csv"
        elif subset == 'eval':
            subset_path = f"{base_path}/annotations/EPIC_100_validation.csv"
        elif subset == 'test':
            subset_path = f"{base_path}/annotations/EPIC_100_test_timestamps.csv"
        else:
            raise ValueError(f'Subset {subset} does not exist. Choose train, eval, or test.')

        subset_video_ids = set([m['video_id'] for m in csv.DictReader(open(subset_path))])

        path_prefix = os.path.join(base_path, data_dir)
        all_files = glob.glob(f"{path_prefix}/*")

        sample_meta = []
        for fn in all_files:
            vid = re.match('(P\d+_\d+)_\d+_\d+', fn.split('/')[-1]).group(1)
            if vid in subset_video_ids:
                sample_meta.append({
                    'filename': fn.split('/')[-1],
                })
        video_paths = VideoPaths(sample_meta, path_prefix=path_prefix)
        # video_paths.filter_missing()
        super().__init__(
            video_info=video_paths,
            prob_moi=prob_moi,
            audio_duration=audio_duration,
            transform=transform,
            decode_audio=decode_audio,
            delta_non_overlap=delta_non_overlap)

    def copy2local(self, local_path=YODA_LOCAL_PATH):
        # Only runs from a single GPU in distributed settings.
        assert local_path != self._video_info._path_prefix
        import torch.distributed as dist
        if dist.is_initialized() and dist.get_rank() != 0:
            return
        os.system(f"mkdir -p {local_path} && gsutil -m -q rsync -r {self._video_info._path_prefix} {local_path}")
        self._video_info._path_prefix = local_path


class Ego4DDenseSegments(VideoDenseDataset):
    def __init__(
            self,
            subset: str = 'train',
            base_path: str = YODA_PATH,
            data_dir: str = 'segments',
            clip_sampling: str = 'random',
            video_duration: float = 0.5,
            audio_duration: float = 2.,
            decode_video: bool = True,
            decode_audio: bool = True,
            num_clips: int = 1,
            delta_non_overlap: float = 0.1,
            dense_steps: int = 8,
            transform: Optional[Callable[[dict], Any]] = None,
    ) -> None:
        import csv
        import glob
        import re

        path_prefix = f'{base_path}/{data_dir}'
        all_files = glob.glob(f"{path_prefix}/*.mp4")

        sample_meta = []
        for fn in all_files:
            sample_meta.append({
                'filename': fn.split('/')[-1],
            })
        video_paths = VideoPaths(sample_meta, path_prefix=path_prefix)
        # video_paths.filter_missing()
        super().__init__(
            video_info=video_paths,
            clip_sampling=clip_sampling,
            video_duration=video_duration,
            audio_duration=audio_duration,
            num_clips=num_clips,
            delta_non_overlap=delta_non_overlap,
            dense_steps=dense_steps,
            transform=transform,
            decode_audio=decode_audio,
            decode_video=decode_video)

    def copy2local(self, local_path=YODA_LOCAL_PATH):
        # Only runs from a single GPU in distributed settings.
        assert local_path != self._video_info._path_prefix
        import torch.distributed as dist
        if dist.is_initialized() and dist.get_rank() != 0:
            return
        os.system(f"mkdir -p {local_path} && gsutil -m -q rsync -r {self._video_info._path_prefix} {local_path}")
        self._video_info._path_prefix = local_path


class Ego4DDenseSegmentsFrames(FramesDenseDataset):
    def __init__(
            self,
            subset: str = 'train',
            base_path: str = YODA_PATH,
            data_dir: str = 'segments_frames',
            clip_sampling: str = 'random',
            audio_duration: float = 2.,
            decode_video: bool = True,
            decode_audio: bool = True,
            delta_non_overlap: float = 0.1,
            dense_steps: int = 8,
            transform: Optional[Callable[[dict], Any]] = None,
    ) -> None:
        import csv
        import glob
        import re
        if subset == 'train':
            subset_path = f"{base_path}/annotations/EPIC_100_train.csv"
        elif subset == 'eval':
            subset_path = f"{base_path}/annotations/EPIC_100_validation.csv"
        elif subset == 'test':
            subset_path = f"{base_path}/annotations/EPIC_100_test_timestamps.csv"
        else:
            raise ValueError(f'Subset {subset} does not exist. Choose train, eval, or test.')

        subset_video_ids = set([m['video_id'] for m in csv.DictReader(open(subset_path))])

        path_prefix = f'{base_path}/{data_dir}'
        all_files = glob.glob(f"{path_prefix}/*")

        sample_meta = []
        for fn in all_files:
            vid = re.match('(P\d+_\d+)_\d+_\d+', fn.split('/')[-1]).group(1)
            if vid in subset_video_ids:
                sample_meta.append({
                    'filename': fn.split('/')[-1],
                })
        video_paths = VideoPaths(sample_meta, path_prefix=path_prefix)
        # video_paths.filter_missing()
        super().__init__(
            video_info=video_paths,
            clip_sampling=clip_sampling,
            audio_duration=audio_duration,
            delta_non_overlap=delta_non_overlap,
            dense_steps=dense_steps,
            transform=transform,
            decode_audio=decode_audio,
            decode_video=decode_video)

    def copy2local(self, local_path=YODA_LOCAL_PATH):
        # Only runs from a single GPU in distributed settings.
        assert local_path != self._video_info._path_prefix
        import torch.distributed as dist
        if dist.is_initialized() and dist.get_rank() != 0:
            return
        os.system(f"mkdir -p {local_path} && gsutil -m -q rsync -r {self._video_info._path_prefix} {local_path}")
        self._video_info._path_prefix = local_path


if __name__ == '__main__':
    from data.builder import build_transforms
    from omegaconf import OmegaConf
    augm_cfg = OmegaConf.load('configs/data_augm/audio_frames_moi.yaml')
    transform = build_transforms(augm_cfg, augment=True)
    dataset = Ego4DFramesMoI(
        base_path='/Users/morgado/datasets/epic-kitchens',
        audio_duration=2,
        delta_non_overlap=1,
        transform=transform,
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

