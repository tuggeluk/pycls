#!/usr/bin/env python3

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Convert the dataset structure of HAM10000 to mimic the ImageNet structure"""

import os
import pandas as pd
from shutil import copyfile

home_dir = "/home/tugg/Documents/NNDesignSpaces/Datasets/HAM10000"
target_dir = home_dir+"/Folder_Label"
data_dir = home_dir+"/Images"
metadata = pd.read_csv(home_dir+"/HAM10000_metadata.csv")

os.mkdir(target_dir)
for label in metadata['dx'].unique():
    label_dir = target_dir+"/"+label
    os.mkdir(label_dir)
    for img_id in metadata[metadata['dx'] == label]['image_id']:
        filename = "/"+img_id+".jpg"
        copyfile(data_dir+filename, label_dir+filename)