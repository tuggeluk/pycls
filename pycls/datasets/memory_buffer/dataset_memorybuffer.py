from .shared_memory_fix_issue39959 import SharedMemory
import numpy as np
import pickle
from tqdm import tqdm
import os


"""Loads a complete dataset into memory, in order do reduce disk load during training"""


class DatasetMemoryBuffer:

    def __init__(self, name='anon'):
        self.image_cache = None
        self.IN_USE = False
        self.name = name
        if self.name == 'anon':
            self.name = self.name + str(np.random.randint(low=0, high=1000))

    def load(self, data_path):
        """load pickle into memory"""
        if self.IN_USE:
            raise Exception("already in use - only load one dataset at at time")

        self.IN_USE = True

        if os.path.isfile(data_path):

            with open(data_path, 'rb') as f:
                self.image_cache = pickle.load(f)

            for im_key, [img, label] in tqdm(self.image_cache.items()):
                self.store_item(im_key, img, label)

        else:
            import cv2
            split_files = os.listdir(data_path)
            _class_ids = sorted(split_files)

            self.image_cache = {}
            img_id = 0
            for class_id in _class_ids:
                im_dir = os.path.join(data_path, class_id)
                for im_name in tqdm(os.listdir(im_dir)):
                    im_path = os.path.join(im_dir, im_name)
                    self.store_item(img_id, cv2.imread(im_path), class_id)
                    self.image_cache[img_id] = None
                    img_id += 1

    def store_item(self,im_key,img,label):
        # store image
        self.try_store_nparray(name="img_buf_" + self.name + "_" + str(im_key), input=img)

        # store image shape
        shape = np.array(img.shape)
        self.try_store_nparray(name="shape_buf_" + self.name + "_" + str(im_key), input=shape)

        # store label
        self.try_store_string(name="label_buf_" + self.name + "_" + str(im_key), input=label)

        # free memory:
        self.image_cache[im_key] = None

    def unlink(self):
        """free all the memory used"""

        for id, img in tqdm(enumerate(self.image_cache)):
            img_buf = SharedMemory(name="img_buf_" + self.name + "_" + str(id))
            img_buf.unlink()
            shape_buf = SharedMemory(name="shape_buf_" + self.name + "_" + str(id))
            shape_buf.unlink()
            lab_buf = SharedMemory(name="label_buf_"+ self.name + "_" + str(id))
            lab_buf.unlink()

        self.IN_USE = False


    def try_store_nparray(self, name, input):

        try:
            img_buf = SharedMemory(create=True, name=name, size=input.nbytes)
        except:
            img_buf = SharedMemory(create=False, name=name, size=input.nbytes)

        shared_array = np.ndarray(input.shape, dtype=input.dtype, buffer=img_buf.buf)
        shared_array[:] = input[:]


    def try_store_string(self, name, input):
        encoded = input.encode()
        try:
            img_buf = SharedMemory(create=True, name=name, size=len(encoded))
        except:
            img_buf = SharedMemory(create=False, name=name, size=len(encoded))

        img_buf.buf[:] = encoded[:]


