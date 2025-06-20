'''
MIT License

Copyright (c) 2019 Diana Wofk

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''

import os
import os.path
import numpy as np
import torch.utils.data as data
import h5py
from . import transforms
import random
from collections import namedtuple
import torch

IMG_EXTENSIONS = ['.h5',]
D_MAX_DATASET = 9.9999
def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx

def make_dataset(dir, class_to_idx, type):
    images = []
    dir = os.path.expanduser(dir)
    for target in sorted(os.listdir(dir)):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue
        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    item = (path, class_to_idx[target])
                    images.append(item)
    return images

def h5_loader(path):
    h5f = h5py.File(path, "r")
    rgb = np.array(h5f['rgb'])
    rgb = np.transpose(rgb, (1, 2, 0))
    depth = np.array(h5f['depth'])
    return rgb, depth

def collect_batch_data(**kwargs):
    assert 'rgb' in kwargs, 'rgb should be set for batch data'
    assert 'depth' in kwargs, 'depth should be set for batch data'
    return kwargs

to_tensor = transforms.ToTensor()

class MyDataloader(data.Dataset):
    modality_names = ['rgb', 'rgbd', 'd'] # , 'g', 'gd'
    color_jitter = transforms.ColorJitter(0.4, 0.4, 0.4)

    def __init__(self, root, type, sparsifier=None, modality='rgbd', loader=h5_loader):
        classes, class_to_idx = find_classes(root)
        imgs = make_dataset(root, class_to_idx, type)
        assert len(imgs)>0, "Found 0 images in subfolders of: " + root + "\n"
        print("Found {} images in {} folder.".format(len(imgs), type))
        self.root = root
        self.imgs = imgs
        self.classes = classes
        self.class_to_idx = class_to_idx
        if type == 'train':
            self.transform = self.train_transform
        elif type == 'val':
            self.transform = self.val_transform
        else:
            raise (RuntimeError("Invalid dataset type: " + type + "\n"
                                "Supported dataset types are: train, val"))
        self.loader = loader
        self.sparsifier = sparsifier

        assert (modality in self.modality_names), "Invalid modality type: " + modality + "\n" + \
                                "Supported dataset types are: " + ''.join(self.modality_names)
        self.modality = modality

    def train_transform(self, rgb, depth):
        raise (RuntimeError("train_transform() is not implemented. "))

    def val_transform(rgb, depth):
        raise (RuntimeError("val_transform() is not implemented."))

    # def create_sparse_depth(self, rgb, depth):
    #     if self.sparsifier is None:
    #         print("no sparsifier")
    #         return depth
    #     else:
    #         mask_keep = self.sparsifier.dense_to_sparse(rgb, depth)
    #         sparse_depth = np.zeros(depth.shape)
    #         sparse_depth[mask_keep] = depth[mask_keep]
    #         return sparse_depth

    # def create_rgbd(self, rgb, depth):
    #     sparse_depth = self.create_sparse_depth(rgb, depth)
    #     rgbd = np.append(rgb, np.expand_dims(sparse_depth, axis=2), axis=2)
    #     return rgbd

    # def calculate_dataset_dmax(self):
    #     max_depth = 0
    #     counter = 0
    #     for img in self.imgs:
    #         counter+=1
    #         path, target = img
    #         rgb, depth = self.loader(path)
    #         if np.amax(depth) > max_depth:
    #             max_depth = np.amax(depth)
    #         if counter % 1000 == 0:
    #             print(counter)
    #     print("Max depth of dataset is: " + str(max_depth))
    #     return max_depth

    def __getraw__(self, index):
        return None

    def __getitem__(self, index):
        return None

    def __len__(self):
        return len(self.imgs)
