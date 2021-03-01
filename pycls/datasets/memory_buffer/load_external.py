#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Program to externally load some datasets"""

from pycls.datasets.memory_buffer.dataset_memorybuffer import DatasetMemoryBuffer
import pycls.datasets.loader as loader
import os
import sys

pickle_caches = loader._DATA_DIR

def main():
    is_done = False
    avail_datasets = os.listdir(pickle_caches)
    avail_datasets = [sys.argv[1]]
    mem_buffers = dict()
    for dataset in avail_datasets:
        print("storing " + dataset + " in shared memory")
        # if not dataset == 'train_imagenet.pickle':
        #     continue
        mem_buffers[dataset] = DatasetMemoryBuffer(dataset)
        mem_buffers[dataset].load(os.path.join(pickle_caches, dataset, "train"))

    while not is_done:
        command = input("To free memory please type <terminate>")
        if command == 'terminate':
            for ds_name, ds_buffer in mem_buffers.items():
                print("unlinking: " + ds_name)
                ds_buffer.unlink()
            is_done = True


if __name__ == "__main__":
    main()
