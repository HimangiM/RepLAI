# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from __future__ import annotations

import logging
from typing import Any, Callable, Optional

import random
import numpy as np
import torch.utils.data
from data.clip_sampling import RandomClipSampler, AnchoredRandomClipSampler
from pytorchvideo.data.video import VideoPathHandler


logger = logging.getLogger(__name__)


def random_clip_sampler(segment_start_sec, segment_stop_sec, clip_duration):
    max_possible_clip_start = max(segment_stop_sec - clip_duration, 0)
    clip_start_sec = random.uniform(segment_start_sec, max_possible_clip_start)
    return clip_start_sec, clip_start_sec + clip_duration


class VideoDataset(torch.utils.data.Dataset):
    """
    LabeledVideoDataset handles the storage, loading, decoding and clip sampling for a
    video dataset. It assumes each video is stored as an encoded video
    (e.g. mp4, avi).
    """

    _MAX_CONSECUTIVE_FAILURES = 10
    _SKIP_ERRORS = True

    def __init__(
        self,
        video_info,
        clip_sampling: str = 'random',
        video_duration: float = 0.5,
        audio_duration: float = 2.,
        num_clips: int = 1,
        transform: Optional[Callable[[dict], Any]] = None,
        decode_audio: bool = True,
        decoder: str = "pyav",
    ) -> None:
        """
        Args:
            video_info: List containing
                    video file paths and associated labels. If video paths are a folder
                    it's interpreted as a frame video, otherwise it must be an encoded
                    video.

            clip_sampling (ClipSampler): Defines how clips should be sampled from each
                video. See the clip sampling documentation for more information.

            transform (Callable): This callable is evaluated on the clip output before
                the clip is returned. It can be used for user defined preprocessing and
                augmentations on the clips. The clip output format is described in __next__().

            decode_audio (bool): If True, also decode audio from video.

            decoder (str): Defines what type of decoder used to decode a video. Not used for
                frame videos.
        """
        self._decode_audio = decode_audio
        self._transform = transform
        self._video_info = video_info
        self._decoder = decoder
        self._num_clips = num_clips

        clip_duration = video_duration if not decode_audio else max(video_duration, audio_duration)
        if clip_sampling == 'random':
            clip_sampling = RandomClipSampler(clip_duration)
        elif clip_sampling == 'timestamps':
            clip_sampling = AnchoredRandomClipSampler(clip_duration)
        self._clip_sampling = clip_sampling
        self._video_clip_sampling = RandomClipSampler(video_duration)
        self._audio_clip_sampling = RandomClipSampler(audio_duration)

        self._video_duration = video_duration
        self._audio_duration = audio_duration
        self._clip_duration = clip_duration

        self.video_path_handler = VideoPathHandler()

    @property
    def num_videos(self):
        """
        Returns:
            Number of videos in dataset.
        """
        return len(self._video_info)

    @staticmethod
    def crop_or_pad_video(x, num_frames, rnd_crop=False):
        if len(x) == num_frames:
            return x
        if len(x) > num_frames:
            if rnd_crop:
                ss_pts = random.sample(range(len(x) - num_frames), k=1)[0]
            else:
                ss_pts = (len(x) - num_frames) // 2
            return x[ss_pts:ss_pts + num_frames]

        if len(x) < num_frames:
            while len(x) < num_frames:
                n_missing = num_frames - len(x)
                x += x[:n_missing]
            return x

    @staticmethod
    def crop_or_pad_audio(x, num_frames, rnd_crop=False):
        if x.shape[1] == num_frames:
            return x
        if x.shape[1] > num_frames:
            if rnd_crop:
                ss_pts = random.sample(range(x.shape[1] - num_frames), k=1)[0]
            else:
                ss_pts = (x.shape[1] - num_frames) // 2
            return x[:, ss_pts:ss_pts + num_frames]

        if x.shape[1] < num_frames:
            while x.shape[1] < num_frames:
                n_missing = num_frames - x.shape[1]
                x = np.concatenate((x, x[:, :n_missing]), 1)
            return x

    def load_video_clip(self, video, video_start, video_end, i_try):
        # Decode audio and video data
        video_clip = video.get_video_clip(video_start, video_end)
        if video_clip is None:
            logger.debug(
                "Failed to load clip {}; trial {}".format(video.name, i_try)
            )
            return None, None
        else:
            nt = int(self._video_duration * video._video_fps)
            video_clip = self.crop_or_pad_video(video_clip, nt, rnd_crop=True)
        return video_clip

    def load_audio_clip(self, video, audio_start, audio_end, i_try):
        audio_clip = None
        if self._decode_audio:
            audio_clip = video.get_audio_clip(audio_start, audio_end)
            if audio_clip is None:
                logger.debug(
                    "Failed to load clip {}; trial {}".format(video.name, i_try)
                )
                return None, None
            else:
                nt = int(self._audio_duration * video._audio_fps)
                audio_clip = self.crop_or_pad_audio(audio_clip, nt, rnd_crop=True)
        return audio_clip

    def load_clip(self, video, video_start, video_end, audio_start, audio_end, i_try):
        video_clip = self.load_video_clip(video, video_start, video_end, i_try)
        audio_clip = self.load_audio_clip(video, audio_start, audio_end, i_try)
        return video_clip, audio_clip

    def __getitem__(self, video_index) -> dict:
        """
        Retrieves the next clip based on the clip sampling strategy and video sampler.

        Returns:
            A dictionary with the following format.

            .. code-block:: text

                {
                    'video': <video_tensor>,
                    'label': <index_label>,
                    'video_label': <index_label>
                    'video_index': <video_index>,
                    'clip_index': <clip_index>,
                    'aug_index': <aug_index>,
                }
        """

        video_path, info_dict = self._video_info[video_index]
        for i_try in range(self._MAX_CONSECUTIVE_FAILURES):
            # Reuse previously stored video if there are still clips to be sampled from
            # the last loaded video.

            try:
                from .pyav_decoder import PyAVDecoder
                video = PyAVDecoder(video_path, decode_audio=self._decode_audio)
                # video = self.video_path_handler.video_from_path(
                #     video_path,
                #     decode_audio=self._decode_audio,
                #     decoder=self._decoder,
                # )
            except Exception as e:
                logger.debug(
                    "Failed to load video with error: {}; trial {}".format(
                        e,
                        i_try,
                    )
                )
                continue

            sample_dict = {
                "video_name": video.name,
                "video_index": video_index,
                **info_dict,
                "video": [],
                "time": []
            }
            if self._decode_audio:
                sample_dict['audio'] = []

            success = True
            for i_clip in range(self._num_clips): # num_clips = 1, video_duration = 10.008, clip_duration = 6.0
                # random clips - clip_start = t, clip_end = t + 6
                clip_start, clip_end = random_clip_sampler(0, video.duration, self._clip_duration)
                # video_start = clip_start, video_end = clip_end, maybe?
                video_start, video_end = random_clip_sampler(clip_start, clip_end, self._video_duration)
                audio_start, audio_end = 0, 0
                if self._decode_audio: # True
                    audio_start, audio_end = random_clip_sampler(clip_start, clip_end, self._audio_duration)
                # len(video_clip) = 96, video_fps = 16, 16*6 = 96, len(audio_clip) = 2, audio_fps = 44100
                video_clip, audio_clip = self.load_clip(video, video_start, video_end, audio_start, audio_end, i_try)
                if video_clip is None and audio_clip is None:
                    success = False
                    break

                # Apply transforms
                # video_clip = 96 images of 426 x 240
                clips = {'video': (video_clip, video._video_fps),
                         **({"audio": (audio_clip, video._audio_fps)} if self._decode_audio else {})}
                # import ipdb; ipdb.set_trace()
                if self._transform is not None:
                    clips = self._transform(clips)   # len(clips['video']) = 8
                for k in clips:
                    if isinstance(clips[k], torch.Tensor):
                        clips[k] = [clips[k]]
                sample_dict["video"] += clips['video']   # len(clips['video']) = 8, clips['video][0] = 3 x 8 x 112 x 112
                if self._decode_audio:
                    sample_dict["audio"] += clips['audio']

                sample_dict['time'].append(clip_start)
                sample_dict['time'].append(clip_end)

            video.close()
            if success:
                if len(sample_dict['video']) == 1:
                    sample_dict['video'] = sample_dict['video'][0]
                if 'audio' in sample_dict and len(sample_dict['audio']) == 1:
                    if self._decode_audio:
                        sample_dict['audio'] = sample_dict['audio'][0]
                return sample_dict

        else:
            if not self._SKIP_ERRORS:
                raise RuntimeError(
                    f"Failed to load video after {self._MAX_CONSECUTIVE_FAILURES} retries. {video_path}"
                )
            video_index = random.sample(range(len(self)), k=1)[0]
            return self[video_index]

    def __len__(self):
        return len(self._video_info)
