# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from __future__ import annotations

import os
import pathlib
import random
from typing import List, Optional, Tuple

from iopath.common.file_io import g_pathmgr
from torchvision.datasets.folder import make_dataset


class VideoPaths:
    """
    VideoPaths contains pairs of video path and integer index label.
    """

    @classmethod
    def from_csv(cls, file_path: str, path_prefix: str = "") -> VideoPaths:
        """
        Factory function that creates a VideoPaths object by reading a file with the
        following format:
            <path> <integer_label>
            ...
            <path> <integer_label>

        Args:
            file_path (str): The path to the file to be read.
            path_prefix (str): The path to the directory containing the data to be read.
        """
        assert g_pathmgr.exists(file_path), f"{file_path} not found."
        sample_metadata = []
        with g_pathmgr.open(file_path, "r") as f:
            for path_label in f.read().splitlines():
                line_split = path_label.rsplit(None, 1)

                # The video path file may not contain labels (e.g. for a test split). We
                # assume this is the case if only 1 path is found and set the label to
                # -1 if so.
                if len(line_split) == 1:
                    file_path = line_split[0]
                    meta = {'filename': file_path}
                else:
                    file_path, label = line_split
                    meta = {'filename': file_path, 'label': int(label)}

                sample_metadata.append(meta)

        assert (
            len(sample_metadata) > 0
        ), f"Failed to load dataset from {file_path}."
        return cls(sample_metadata, path_prefix)

    @classmethod
    def from_directory(cls, dir_path: str) -> VideoPaths:
        """
        Factory function that creates a VideoPaths object by parsing the structure
        of the given directory's subdirectories into the classification labels. It
        expects the directory format to be the following:
             dir_path/<class_name>/<video_name>.mp4

        Classes are indexed from 0 to the number of classes, alphabetically.

        E.g.
            dir_path/class_x/xxx.ext
            dir_path/class_x/xxy.ext
            dir_path/class_x/xxz.ext
            dir_path/class_y/123.ext
            dir_path/class_y/nsdf3.ext
            dir_path/class_y/asd932_.ext

        Would produce two classes labeled 0 and 1 with 3 videos paths associated with each.

        Args:
            dir_path (str): Root directory to the video class directories .
        """
        assert g_pathmgr.exists(dir_path), f"{dir_path} not found."

        # Find all classes based on directory names. These classes are then sorted and indexed
        # from 0 to the number of classes.
        classes = sorted(
            (f.name for f in pathlib.Path(dir_path).iterdir() if f.is_dir())
        )
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        sample_metadata = make_dataset(
            dir_path, class_to_idx, extensions=("mp4", "avi")
        )
        assert (
            len(sample_metadata) > 0
        ), f"Failed to load dataset from {dir_path}."
        sample_metadata = [{'filename': m[0], 'label': m[1]} for m in sample_metadata]
        return cls(sample_metadata, path_prefix=dir_path)

    def __init__(
        self, sample_metadata: List[dict], path_prefix=""
    ) -> None:
        """
        Args:
            sample_metadata [(str, int)]: a list of tuples containing the video
                path and integer label.
        """
        self._sample_metadata = sample_metadata
        self._path_prefix = path_prefix

        for i, meta in enumerate(self._sample_metadata):
            print(os.path.join(self._path_prefix, meta['filename']))
            if i == 3:
                break

    def path_prefix(self, prefix):
        self._path_prefix = prefix

    path_prefix = property(None, path_prefix)

    def __getitem__(self, index: int) -> Tuple[str, dict]:
        """
        Args:
            index (int): the path and label index.

        Returns:
            The path and label tuple for the given index.
        """
        meta = self._sample_metadata[index]
        return os.path.join(self._path_prefix, meta['filename']), meta

    def __len__(self) -> int:
        """
        Returns:
            The number of video paths and label pairs.
        """
        return len(self._sample_metadata)

    def filter_missing(self):
        available = []
        for meta in self._sample_metadata:
            fn = os.path.join(self._path_prefix, meta['filename'])
            if os.path.isfile(fn):
                available += [meta]
        self._sample_metadata = available


class VideoSegmentsPaths:
    def __init__(
        self, sample_metadata: List[dict], path_prefix="", repeat=1
    ) -> None:
        """
        Args:
            sample_metadata [(str, int)]: a list of tuples containing the video
                path and integer label.
        """
        self._sample_metadata = sample_metadata
        self._path_prefix = path_prefix
        self._repeat = repeat

    def path_prefix(self, prefix):
        self._path_prefix = prefix

    path_prefix = property(None, path_prefix)

    def __getitem__(self, index: int) -> Tuple[str, dict]:
        """
        Args:
            index (int): the path and label index.

        Returns:
            The path and label tuple for the given index.
        """
        index = index % len(self._sample_metadata)
        meta = self._sample_metadata[index]
        fn = random.sample(meta['filenames'], k=1)[0]
        meta = {'filename': fn}
        return os.path.join(self._path_prefix, fn), meta

    def __len__(self) -> int:
        """
        Returns:
            The number of video paths and label pairs.
        """
        return len(self._sample_metadata) * self._repeat

    def filter_missing(self):
        available = []
        for meta in self._sample_metadata:
            fn = os.path.join(self._path_prefix, meta['filename'])
            if os.path.isfile(fn):
                available += [meta]
        self._sample_metadata = available
