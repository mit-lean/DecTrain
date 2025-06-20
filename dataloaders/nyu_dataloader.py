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
import numpy as np
from . import transforms
from torchvision import transforms as torch_transforms
import torch 

from .dataloader import MyDataloader
import time # debugging
from dataloaders.dataloader import collect_batch_data

iheight, iwidth = 480, 640 # raw image size


class NYUDataset(MyDataloader):
    def __init__(self, root, type, sparsifier=None, modality='rgb', transform_type="nyu"):
        super(NYUDataset, self).__init__(root, type, sparsifier, modality)
        self.output_size = (224, 224)
        self.transform_type = transform_type
    
    
    def __getraw__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (rgb, depth) the raw data.
        """
        path, target = self.imgs[index]
        rgb, depth = self.loader(path)
        return rgb, depth

    def __getitem__(self, index):
        rgb, depth = self.__getraw__(index)
        if self.transform is not None:
            rgb_np, depth_np = self.transform(rgb, depth)
        else:
            raise(RuntimeError("transform not defined"))

        if self.modality == 'rgb':
            input_np = rgb_np
        to_tensor = transforms.ToTensor()
        input_tensor = torch.from_numpy(input_np).float()
        input_tensor = torch.permute(input_tensor, (2, 0, 1))
        while input_tensor.dim() < 3:
            input_tensor = input_tensor.unsqueeze(0)
        depth_tensor = to_tensor(depth_np)
        depth_tensor = depth_tensor.unsqueeze(0)
        # return input_tensor, depth_tensor
        return collect_batch_data(rgb=input_tensor, depth=depth_tensor)
    
    def train_transform(self, rgb, depth):
        ## original fast depth train transform
        s = np.random.uniform(1.0, 1.5) # random scaling
        depth_np = depth / s
        angle = np.random.uniform(-5.0, 5.0) # random rotation degrees
        do_flip = np.random.uniform(0.0, 1.0) < 0.5 # random horizontal flip

        # perform 1st step of data augmentation
        transform = transforms.Compose([
            transforms.Resize(250.0 / iheight), # this is for computational efficiency, since rotation can be slow
            transforms.Rotate(angle),
            transforms.Resize(s),
            transforms.CenterCrop((228, 304)),
            transforms.HorizontalFlip(do_flip),
            transforms.Resize(self.output_size),
        ])
        rgb_np = transform(rgb)
        rgb_np = self.color_jitter(rgb_np) # random color jittering
        rgb_np = np.asfarray(rgb_np, dtype='float') / 255
        depth_np = transform(depth_np)
        if self.transform_type == "dinov2":
            rgb_t = torch.from_numpy(rgb_np)
            # add additional step on RGB after resizing 
            transform_norm = torch_transforms.Compose([
                        lambda x: 255.0 * x, # Discard alpha component and scale by 255
                        torch_transforms.Normalize(
                            mean=(123.675, 116.28, 103.53),
                            std=(58.395, 57.12, 57.375),
                        ),
                    ])
            rgb_t = torch.permute(rgb_t, (2,0,1))
            rgb_t = transform_norm(rgb_t)
            rgb_np = rgb_t.numpy()
        return rgb_np, depth_np

    def val_transform(self, rgb, depth):
        depth_np = depth
        transform = transforms.Compose([
            transforms.Resize(250.0 / iheight),
            transforms.CenterCrop((228, 304)),
            transforms.Resize(self.output_size),
        ])
        rgb_np = transform(rgb)
        rgb_np = np.asfarray(rgb_np, dtype='float') / 255
        depth_np = transform(depth_np)
        if self.transform_type == "dinov2":
            rgb_t = torch.from_numpy(rgb_np)
            # add additional step on RGB after resizing 
            transform_norm = torch_transforms.Compose([
                        lambda x: 255.0 * x, # Discard alpha component and scale by 255
                        torch_transforms.Normalize(
                            mean=(123.675, 116.28, 103.53),
                            std=(58.395, 57.12, 57.375),
                        ),
                    ])
            rgb_t = torch.permute(rgb_t, (2,0,1))
            rgb_t = transform_norm(rgb_t)
            rgb_np = rgb_t.numpy()
        return rgb_np, depth_np
