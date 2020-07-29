#!/usr/bin/env python3

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Check all dataset folders if train and val subfolders exist, if val is missing create one"""

import os
import random
import shutil


val_portion = 0.1
data_dir = "datasets/data"
random.seed = 846

# iterate trough subfolders
subfolders = os.listdir(data_dir)
for folder in subfolders:
    folder_contents = os.listdir(data_dir+"/"+folder)
    if 'val' in folder_contents:
        continue
    os.mkdir(data_dir+"/"+folder+"/val")
    for cla in os.listdir(data_dir+"/"+folder+"/train"):
        # create a corresponding class folder in val and move a portion of the data
        train_dir = data_dir + "/" + folder + "/train/"+cla
        val_dir = data_dir + "/" + folder + "/val/"+cla
        os.mkdir(val_dir)
        samples = os.listdir(train_dir)
        val_size = round(len(samples)*val_portion)
        val_samples = random.sample(samples, val_size)
        for val_sample in val_samples:
            print("move bitch")
            shutil.move(train_dir+"/"+val_sample, val_dir+"/"+val_sample)



