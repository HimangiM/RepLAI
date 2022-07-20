from typing import Dict, Any
import random
from pytorchvideo.data.clip_sampling import ClipSampler, ClipInfo
from pytorchvideo.data.clip_sampling import UniformClipSampler, RandomClipSampler, ConstantClipsPerVideoSampler

__all__ = ['AnchoredRandomClipSampler', 'UniformClipSampler', 'RandomClipSampler', 'ConstantClipsPerVideoSampler']


class AnchoredRandomClipSampler(ClipSampler):
    """
    Randomly samples clip of size clip_duration from the videos.
    """

    def __call__(
        self, last_clip_time: float, video_duration: float, annotation: Dict[str, Any]
    ) -> ClipInfo:
        """
        Args:
            last_clip_time (float): Not used for RandomClipSampler.
            video_duration: (float): the duration (in seconds) for the video that's
                being sampled
            annotation (Dict): Not used by this sampler.
        Returns:
            clip_info (ClipInfo): includes the clip information of (clip_start_time,
            clip_end_time, clip_index, aug_index, is_last_clip). The times are in seconds.
            clip_index, aux_index and is_last_clip are always 0, 0 and True, respectively.

        """
        anchor_timestamp = annotation['timestamp']
        if isinstance(anchor_timestamp, (int, float)):
            anchor_timestamp = float(min(max(anchor_timestamp, 0.), video_duration))

            clip_start_window = (
                max(anchor_timestamp - self._clip_duration, 0),
                min(anchor_timestamp, max(video_duration - self._clip_duration, 0))
            )
        elif isinstance(anchor_timestamp, (list, tuple)) and len(anchor_timestamp) == 2:
            anchor_start_time = float(min(max(anchor_timestamp[0], 0.), video_duration))
            anchor_stop_time = float(min(max(anchor_timestamp[1], 0.), video_duration))

            clip_start_window = (
                max(anchor_start_time - self._clip_duration, 0),
                min(anchor_stop_time, max(video_duration - self._clip_duration, 0))
            )
        else:
            raise NotImplementedError

        clip_start_sec = random.uniform(clip_start_window[0], clip_start_window[1])
        return ClipInfo(
            clip_start_sec, clip_start_sec + self._clip_duration, 0, 0, True
        )
