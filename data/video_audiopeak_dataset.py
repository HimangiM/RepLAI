# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from __future__ import annotations

import logging
import traceback
from typing import Any, Callable, Optional

import random
import numpy as np
import torch.utils.data
# from data.clip_sampling import RandomClipSampler, AnchoredRandomClipSampler
from pytorchvideo.data.video import VideoPathHandler
from scipy import signal
import copy

logger = logging.getLogger(__name__)


def random_clip_sampler(segment_start_sec, segment_stop_sec, clip_duration):
    max_possible_clip_start = max(segment_stop_sec - clip_duration, 0)
    clip_start_sec = random.uniform(segment_start_sec, max_possible_clip_start)
    return clip_start_sec, clip_start_sec + clip_duration


class VideoAudioPeakDataset(torch.utils.data.Dataset):
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
        video_duration: float = 10.0,
        audio_duration: float = 10.0,
        num_clips: int = 1,
        transform: Optional[Callable[[dict], Any]] = None,
        decode_audio: bool = True,
        decoder: str = "pyav",
        audio_event_prominence: float = 1.,
        delta_non_overlap: float = 0.
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
        self._audio_event_prominence = audio_event_prominence

        self._video_duration = video_duration
        self._audio_duration = audio_duration

        self.video_path_handler = VideoPathHandler()
        self.delta_non_overlap = delta_non_overlap

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
        nf = copy.deepcopy(num_frames)
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
                if n_missing < x.shape[1]:
                    x = np.concatenate((x, x[:, :n_missing]), 1)
                else:
                    x = np.concatenate((x, x), 1)
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
            duration = video_end - video_start
            nt = int(duration * video._video_fps)
            video_clip = self.crop_or_pad_video(video_clip, nt, rnd_crop=True)
        return video_clip

    def load_audio_clip(self, video, audio_start, audio_end, i_try, crop_or_pad=False):
        audio_clip = None
        if self._decode_audio:
            audio_clip = video.get_audio_clip(audio_start, audio_end)
            if audio_clip is None:
                logger.debug(
                    "Failed to load clip {}; trial {}".format(video.name, i_try)
                )
                return None, None
            if crop_or_pad:
                duration = audio_end - audio_start
                nt = int(duration * video._audio_fps)
                audio_clip = self.crop_or_pad_audio(audio_clip, nt, rnd_crop=True)
        return audio_clip

    def find_audio_peaks(self, video, i_try):
        audio_start, audio_end = video._audio_start, video._audio_start + video._audio_duration
        audio_clip = self.load_audio_clip(video, audio_start, audio_end, i_try)
        if audio_clip is None:
            return None

        # Generate spectrogram
        assert audio_clip.ndim == 2
        audio_clip_mono = np.mean(audio_clip, axis=0)
        f, t, Sxx_nolog = signal.spectrogram(audio_clip_mono, video._audio_fps)
        Sxx = np.log10(Sxx_nolog + 1e-5)
        Sxx[Sxx == -np.inf] = -25
        Sxx_norm = (Sxx - np.mean(Sxx, axis=1, keepdims=True)) / (np.std(Sxx, axis=1, keepdims=True) + 1e-5)
        Sxx_norm_peaks = signal.find_peaks(Sxx_norm.mean(0), distance=10, prominence=self._audio_event_prominence)

        # peak samples index to time conversion
        # peaks_t_all = np.array([round((t_s / len(t)) * self._video_duration, 2) for t_s in Sxx_norm_peaks[0]])
        # peaks_t = np.array([t_s for t_s in peaks_all_sec
        #                     if t_s >= self.video_delta_time and t_s <= (self._video_duration - self.video_delta_time)])
        peaks_all_sec = np.array([t[peaks_pts] for peaks_pts in Sxx_norm_peaks[0]])
        peaks_t = np.array([t_s for t_s in peaks_all_sec
                            if t_s >= self.delta_non_overlap + self._video_duration and t_s <= (video._video_duration - self.delta_non_overlap - self._video_duration)])
        return peaks_t

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
                traceback.print_exc()
                break
                # print ('here')
                # logger.debug(
                #     "Failed to load video with error: {}; trial {}".format(
                #         e,
                #         i_try,
                #     )
                # )
                # continue

            sample_dict = {
                "video_name": video.name,
                "video_index": video_index,
                **info_dict,
                "video_l": [],
                "video_r": []
            }
            if self._decode_audio:
                sample_dict['audio'] = []

            peaks_t = self.find_audio_peaks(video, i_try)
            if peaks_t is None:
                traceback.print_exc()
                break
            if len(peaks_t) < self._num_clips:
                # Not enough moments of interest found
                video_index = random.sample(range(len(self)), k=1)[0]
                return self[video_index]
            np.random.shuffle(peaks_t)

            for i_clip in range(self._num_clips): # num_clips = 1
                #  Time:  ts-gap-dur   ts-gap      ts       ts+gap        ts+gap+dur
                #  Video: | <----------> | <------> | <------> | <-----------> |
                #              clip_l                                clip_r
                #
                #               ts-audio_dur/2     ts   ts+audio_dur/2
                #  Audio:  --------- | <------------|-------------> | ---------
                #                              clip_audio
                peak_time = peaks_t[i_clip]
                video_clip_l = self.load_video_clip(video,
                                                    peak_time - self.delta_non_overlap - self._video_duration,
                                                    peak_time - self.delta_non_overlap,
                                                    i_try)
                video_clip_r = self.load_video_clip(video,
                                                    peak_time + self.delta_non_overlap,
                                                    peak_time + self.delta_non_overlap + self._video_duration,
                                                    i_try)
                audio_clip_lr = self.load_audio_clip(video,
                                                     peak_time - self._audio_duration / 2.,
                                                     peak_time + self._audio_duration / 2.,
                                                     i_try)
                audio_clip_lr = np.mean(audio_clip_lr, axis=0)
                clips = {'video_l': (video_clip_l, video._video_fps),
                         'video_r': (video_clip_r, video._video_fps),
                         'audio': (audio_clip_lr, video._audio_fps)}

                if self._transform is not None:
                    clips = self._transform(clips)

                sample_dict["video_l"].append(clips['video_l'])     # 3 x 8 x 112 x 112
                sample_dict["video_r"].append(clips['video_r'])
                if self._decode_audio:
                    sample_dict["audio"].append(clips['audio'])

            video.close()
            # if success:
            #     if len(sample_dict['video']) == 1:
            #         sample_dict['video'] = sample_dict['video'][0]
            #     if 'audio' in sample_dict and len(sample_dict['audio']) == 1:
            #         if self._decode_audio:
            #             sample_dict['audio'] = sample_dict['audio'][0]
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


if __name__=='__main__':
    # from data.video_paths import VideoPaths, VideoSegmentsPaths
    import os
    all_files = ['/glusterfs/pmorgado/datasets/epic-kitchens/segments/P01_04_000070_000080.MP4']
    path_prefix = '/glusterfs/pmorgado/datasets/epic-kitchens/segments'

    sample_meta = []
    for fn in all_files:
        sample_meta.append({
            'filename': fn.split('/')[-1],
        })
    video_paths = [(os.path.join(path_prefix, sample_meta[0]['filename']), sample_meta[0])]

    obj = VideoAudioPeakDataset(video_info=video_paths)
    print (obj[0])
