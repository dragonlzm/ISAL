#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""CIFAR10 dataset."""

import os
import pickle

import numpy as np
import pycls.core.logging as logging
import torch.utils.data
from pycls.core.config import cfg

logger = logging.get_logger(__name__)

# Per-channel mean and standard deviation values on CIFAR
_MEAN = [129.3, 124.1, 112.4]
_STD = [68.2, 65.4, 70.4]


class SVHN(torch.utils.data.Dataset):
    """CIFAR-10 dataset."""

    def __init__(self, data_path, split, output_id=False):
        assert os.path.exists(data_path), "Data path '{}' not found".format(data_path)
        splits = ["train", "test"]
        assert split in splits, "Split '{}' not supported for cifar".format(split)
        logger.info("Constructing CIFAR-100 {}...".format(split))
        self._data_path, self._split = data_path, split
        self.output_id=output_id #if True dataloader ouput img,label,index
        self._ids = None
        if self.output_id:
            self._inputs, self._labels ,self._ids = self._load_data()
        else:
            self._inputs, self._labels = self._load_data()
        

    def _load_data(self):
        """Loads data into memory."""
        # Compute data batch names
        if self._split == "train":
            batch_path = cfg.TRAIN.DATAPATH
        else:
            batch_path = cfg.TEST.DATAPATH
        logger.info("{} data path: {}".format(self._split, batch_path))
        # Load data batches
        inputs, labels = [], []
        with open(batch_path, "rb") as f:
            data = pickle.load(f, encoding="bytes")
        inputs.append(data[b"data"])
        if b'labels' in data.keys():
            labels += data[b'labels']
        else:
            labels += data[b'fine_labels']
        # Combine and reshape the inputs
        inputs = np.array(inputs, dtype=np.float32)
        inputs = inputs.reshape((-1, 3, cfg.TRAIN.IM_SIZE, cfg.TRAIN.IM_SIZE))
        if self.output_id:
            ids = data[b"img_id"]
            return inputs, labels, ids
        else:
            return inputs, labels

    def _prepare_im(self, im):
        """Prepares the image for network input."""
        for i in range(3):
            # Perform per-channel normalization on CHW image
            im[i] = (im[i] - _MEAN[i]) / _STD[i]
        if self._split == "train":
            # Randomly flip and crop center patch from CHW image
            size = cfg.TRAIN.IM_SIZE
            im = im[:, :, ::-1] if np.random.uniform() < 0.5 else im
            im = np.pad(im, ((0, 0), (4, 4), (4, 4)), mode="constant")
            y = np.random.randint(0, im.shape[1] - size)
            x = np.random.randint(0, im.shape[2] - size)
            im = im[:, y : (y + size), x : (x + size)]
        return im

    def __getitem__(self, index):
        im, label = self._inputs[index, ...].copy(), self._labels[index]
        im = self._prepare_im(im)
        if self.output_id:
            id = self._ids[index]
            return im, label - 1, id
        else:
            return im, label - 1

    def __len__(self):
        return self._inputs.shape[0]
