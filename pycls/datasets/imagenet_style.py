#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Base Class to be used by Datasets stored in an ImageNet-style dataformat"""

import os
import cv2
import numpy as np
import pycls.core.logging as logging
import pycls.datasets.transforms as transforms
import torch.utils.data
from pycls.datasets.memory_buffer.shared_memory_fix_issue39959 import SharedMemory
import array
from skimage.transform import resize

from pycls.core.config import cfg


logger = logging.get_logger(__name__)


class ImageNetStyle(torch.utils.data.Dataset):
    """ImageNet-Style dataset."""

    def __init__(self, name, statistics, data_path, split):
        assert os.path.exists(data_path), "Data path '{}' not found".format(data_path)
        splits = ["train", "val"]
        assert split in splits, "Split '{}' not supported for ImageNet-Style".format(split)
        logger.info("Constructing {} {}...".format(name, split))
        self.name = name.lower()
        self._MEAN, self._SD, self._EIG_VALS, self._EIG_VECS = statistics

        if split == "train":
            self._USED_CONFIG = "TRAIN"
        else:
            self._USED_CONFIG = "TEST"

        if "SHARED_MEM" in cfg[self._USED_CONFIG].keys():
            self.shared_mem = cfg[self._USED_CONFIG].SHARED_MEM
        else:
            self.shared_mem = False

        if "ROTATE" in cfg[self._USED_CONFIG].keys():
            self.rotate = cfg[self._USED_CONFIG].ROTATE
        else:
            self.rotate = False

        if "MASK_CORNERS" in cfg[self._USED_CONFIG].keys():
            self.mask = cfg[self._USED_CONFIG].MASK_CORNERS
        else:
            self.mask = False

        self.LocalProcRandGen = np.random.RandomState()

        self._data_path, self._split = data_path, split
        self.path_name = self._data_path.split("/")[-1]
        self._construct_imdb()

    def _construct_imdb(self):
        """Constructs the imdb."""
        # Compile the split data path
        split_path = os.path.join(self._data_path, self._split)
        logger.info("{} data path: {}".format(self._split, split_path))
        # Images are stored per class in subdirs (format: n<number>)
        split_files = os.listdir(split_path)
        self._class_ids = sorted(split_files)
        # Map ImageNet class ids to contiguous ids
        self._class_id_cont_id = {v: i for i, v in enumerate(self._class_ids)}
        # Construct the image db
        self._imdb = []
        for class_id in self._class_ids:
            cont_id = self._class_id_cont_id[class_id]
            im_dir = os.path.join(split_path, class_id)
            for im_name in os.listdir(im_dir):
                im_path = os.path.join(im_dir, im_name)
                self._imdb.append({"im_path": im_path, "class": cont_id})
        logger.info("Number of images: {}".format(len(self._imdb)))
        logger.info("Number of classes: {}".format(len(self._class_ids)))

        # store entire dataset in mem
        # if self.cache_dataset:
        #     from tqdm import tqdm
        #     self.image_cache = dict()
        #     for ind, imdb_ele in tqdm(enumerate(self._imdb)):
        #             self.image_cache[ind] = cv2.imread(imdb_ele["im_path"])

    def _prepare_im(self, im):
        """Prepares the image for network input."""
        im = im[:, :, ::-1].astype(np.float32) / 255
        # Train and test setups differ
        train_size, test_size = cfg.TRAIN.IM_SIZE, cfg.TEST.IM_SIZE
        if self._split == "train":
            # For training use random_sized_crop, horizontal_flip, augment, lighting
            im = transforms.random_sized_crop(im, train_size)
            im = transforms.horizontal_flip(im, prob=0.5)
            im = transforms.augment(im, cfg.TRAIN.AUGMENT)
            im = transforms.lighting(im, cfg.TRAIN.PCA_STD, self._EIG_VALS, self._EIG_VECS)
        else:
            # Scale and center crop
            im = transforms.scale_and_center_crop(im, cfg.TEST.IM_SIZE, train_size)

        if self.rotate:
            im = transforms.rotate_image(im)

        if self.mask:
            im = transforms.mask_corners(im)

        # from PIL import Image
        # Image.fromarray(((im / np.max(im)) * 255).astype(np.uint8)).show
        # For training and testing use color normalization
        im = transforms.color_norm(im, self._MEAN, self._SD)
        # Convert HWC/RGB/float to CHW/BGR/float format
        im = np.ascontiguousarray(im[:, :, ::-1].transpose([2, 0, 1]))
        return im

    def __getitem__(self, index):
        # Load the image
        if self.shared_mem:

            shape_shm = SharedMemory(name='shape_buf_' + self.path_name + "_" + str(index))
            shape = np.ndarray((3,), dtype=np.int64, buffer=shape_shm.buf)

            img_shm = SharedMemory(name='img_buf_' + self.path_name + "_" + str(index))
            im = np.ndarray(shape, dtype=np.uint8, buffer=img_shm.buf)

            shape = None
            im = im.astype(np.float32, copy=True)
            shape_shm.close()
            img_shm.close()

            label_shm = SharedMemory(name='label_buf_' + self.path_name + "_"+str(index))
            label = bytes(array.array('b', label_shm.buf[:label_shm.size])).decode()
            label = self._class_id_cont_id[label]
            label_shm.close()

        else:
            im = cv2.imread(self._imdb[index]["im_path"])
            im = im.astype(np.float32, copy=False)

            # Retrieve the label np.zeros((3, 224, 224))
            label = self._imdb[index]["class"]
        # im = np.zeros((1280, 1024, 3))
        # Prepare the image for training / testing np.zeros((1280, 1024, 3))

        im = self._prepare_im(im)

        return im, label

    def __len__(self):
        return len(self._imdb)
