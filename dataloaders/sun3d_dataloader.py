import os
import sys
import numpy as np
import torch
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from PIL import Image, ImageFile
# ImageFile.LOAD_TRUNCATED_IMAGES = True
from collections import namedtuple
from dataloaders.dataloader import collect_batch_data
from dataloaders.landmarks import get_landmark_counts
import time # debugging
from . import associate 

class SUN3DMetaDataset:

    def __init__(self, root, mode='val', modality=['rgb', 'd', 'pose'], output_size=(224, 224), val_transform_type = "direct_resizing", split_start = 0, split_end = 1.0, skip_idx = 1):
        self.root = root
        self.pose_file = sorted(os.listdir(os.path.join(root, 'extrinsics')))[-1] # get the latest file in the folder
        self.modality = modality # (rgb, d, pose)
        self.output_size = output_size # (width, height)
        self.iheight, self.iwidth = 480, 640
        self.color_jitter = transforms.ColorJitter(0.4, 0.4, 0.4)
        # skip indices to simulate faster camera
        self.skip_idx = skip_idx
        
        self.matches = self._readDataset() # RGB-D pairs
        self.extrinsics = self._readPose() # camera extrinsics
        # if not (split_start == 0 and split_end == 1.0):
        if split_start > 1.0: # unit in frame
            split_start_idx = int(split_start)
        else:
            split_start_idx = min(round(split_start*len(self.matches)),len(self.matches)-1) # don't give out of range index 
        if split_end > 1.0: # unit in frame
            split_end_idx = int(split_end)
        else:
            split_end_idx = round(split_end*len(self.matches)-1)
        self.matches = self.matches[split_start_idx:split_end_idx:skip_idx]
        print("Found {} images and depth pairs in {}.".format(len(self.matches), self.root))

        self.to_image = transforms.ToPILImage()

        if mode == 'train':
            self.transform = self.train_transform
        elif mode == 'val':
            self.transform = self.val_transform
        else:
            raise (RuntimeError("Invalid dataset type: " + mode + "\n"
                                "Supported dataset types are: train, val"))
        
        # camera intrinsics
        intrinsics = torch.from_numpy(np.loadtxt(os.path.join(self.root, 'intrinsics.txt'))).float().view(3, 3)
        self.intrinsics = {
            'fx': intrinsics[0, 0],
            'fy': intrinsics[1, 1],
            'cx': intrinsics[0, 2],
            'cy': intrinsics[1, 2],
        }
        assert self.intrinsics['cx'] == self.iwidth / 2, "cx: {} != {}".format(self.intrinsics['cx'], self.iwidth / 2)
        assert self.intrinsics['cy'] == self.iheight / 2, "cy: {} != {}".format(self.intrinsics['cy'], self.iheight / 2)
        # val transforms 
        self.val_transform_type = val_transform_type # direct_resizing, resize_centercrop_resize, dinov2



    def __getraw__(self, index):
        raw_items = {}
        skip_index = index*self.skip_idx # skip frames to simulate faster camera
        raw_items['real_idx'] = skip_index # real indices for skipping frames (note, adding sway not enabled here)
        # index = index of trainloader matches where it will be from 0 to length of (skipped) matches
        # skip_index = index of image in full sequence with no skips (useful for indexing pose which is not already skipped like matches)
        timestamp = float(self.matches[index][0].split('-')[0]) # frame index
        raw_items['timestamp'] = timestamp
        if 'rgb' in self.modality:
            rgb_file = os.path.join(self.root, 'image', self.matches[index][0])
            rgb = Image.open(rgb_file)
            rgb_bgr = np.array(rgb)[:, :, [2, 1, 0]] # opencv read in format is BGR, swap for landmark extraction
            raw_items['landmark_count'] = get_landmark_counts(rgb_bgr, detector='SIFT') # get landmark count for rgb image
            raw_items['rgb'] = rgb
        if 'd' in self.modality:
            depth_file = os.path.join(self.root, 'depth', self.matches[index][1])
            depth = Image.open(depth_file)
            raw_items['depth'] = depth
        if 'pose' in self.modality:
            raw_items['rot'] = torch.squeeze(self.extrinsics[skip_index][:, 0:3].view(3, 3).double())
            raw_items['trans'] = torch.squeeze(torch.transpose(self.extrinsics[skip_index][:, 3].view(3, 1).double(), 0, 1))
        if rgb.size != depth.size:
            raise Exception("Color and depth image do not have the same resolution.")
        if rgb.mode != "RGB":
            raise Exception("Color image is not in RGB format")
        if depth.mode not in ["I", "I;16"]:
            raise Exception(f"Depth image is not in intensity format, mode={depth.mode}")

        return raw_items

    def __getitem__(self, index):
        raw_items = self.__getraw__(index)
        items = self.transform(raw_items)
        # 1. they shift up 3-bit for visualization purpose, we shift it back
        # 2. depth is saved in 16-bit PNG in millimeters
        # items['depth'] = (items['depth'].type(torch.FloatTensor) >> 3) * 0.001
        items['depth'] = (items['depth'] >> 3).type(torch.FloatTensor) * 0.001
        return collect_batch_data(**items)

    def __len__(self):
        return len(self.matches)
    
    def _readDataset(self):
        # if associations file exists, read matches from associations.txt
        if os.path.isfile(self.root + "/associations.txt"):
            # read file 
            with open(os.path.join(self.root + "/associations.txt"), 'r') as f:
                matches_filenames = [(line.split()[0], line.split()[1]) for line in f ]
            f.close()
        else: # associations file does not exist, make associations 
            rgb_files = sorted(os.listdir(os.path.join(self.root, 'image')))
            depth_files = sorted(os.listdir(os.path.join(self.root, 'depth')))
            # strip timestamps from filenames 
            rgb_timestamps = [int(x[8:-4]) for x in rgb_files]
            depth_timestamps = [int(x[8:-4]) for x in depth_files]
            # make matches from closest timestamps 
            matches = associate.associate(rgb_timestamps, depth_timestamps, 0, 100000)
            # find index of that timestamp in rgb_files 
            matches_filenames = []
            for i in range(len(matches)):
                # find timestamp of rgb and depth in rgb_files names
                # pad timestamp to correct number of digits 
                for j in rgb_files:
                    if j.find(str(matches[i][0]).zfill(12)) != -1:
                        rgb_file = j
                for j in depth_files:
                    if j.find(str(matches[i][1]).zfill(12)) != -1:
                        depth_file = j
                matches_filenames.append((rgb_file, depth_file))
                # save matches as associations file
                with open(self.root + "/associations.txt", 'a+') as f:
                    f.write(rgb_file + " " + depth_file + "\n")
            f.close()
        return matches_filenames

    def _readPose(self):
        pose_file = os.path.join(self.root, 'extrinsics', self.pose_file)
        extrinsics = np.loadtxt(pose_file)
        extrinsics = torch.from_numpy(extrinsics).float().view(-1, 3, 4) # size: (Nx3, 4), N: num of images
        return extrinsics

    def train_transform(self, items):
        # angle = np.random.uniform(-5.0, 5.0) # random rotation degrees
        # do_flip = np.random.uniform(0.0, 1.0) < 0.5 # random horizontal flip

        rgb, depth = items['rgb'], items['depth']

        # perform 1st step of data augmentation
        to_tensor = transforms.ToTensor()
        transform = transforms.Compose([
            # transforms.Resize(250.0 / self.iheight), # this is for computational efficiency, since rotation can be slow
            transforms.RandomRotation(5.0, interpolation=InterpolationMode.NEAREST),
            # transforms.CenterCrop((228, 304)),
            transforms.RandomHorizontalFlip(0.5),
            transforms.Resize(self.output_size, interpolation=InterpolationMode.NEAREST),
            # transforms.ToTensor()
        ])
        
        rgb_t, depth_t = to_tensor(rgb), to_tensor(depth)
        
        transform_list = [rgb_t, depth_t]
        transform_modality = []

        rgbd_t = transform(torch.cat(transform_list, dim=0))
        for i, m in enumerate(transform_modality):
            data_t = rgbd_t[4+i, :, :].unsqueeze(0)
            items[m] = data_t

        rgb_t = self.color_jitter(rgbd_t[:3, :, :])
        depth_t = rgbd_t[3, :, :].unsqueeze(0)

        items['rgb'] = rgb_t
        items['depth'] = depth_t

        return items

    def val_transform(self, items):
        rgb, depth = items['rgb'], items['depth']
        to_tensor = transforms.ToTensor()
        if self.val_transform_type == "direct_resizing":
            transform = transforms.Compose([
                transforms.Resize(self.output_size, interpolation=InterpolationMode.NEAREST),
            ])            

            rgb_t = transform(to_tensor(rgb))
            depth_t = transform(to_tensor(depth))
        elif self.val_transform_type == "dinov2":
            transform = transforms.Compose([
                transforms.Resize(self.output_size, interpolation=InterpolationMode.NEAREST),
            ])            
            rgb_t = transform(to_tensor(rgb))
            depth_t = transform(to_tensor(depth))
            # add additional step on RGB after resizing 
            transform_norm = transforms.Compose([
                        lambda x: 255.0 * x[:3], # Discard alpha component and scale by 255
                        transforms.Normalize(
                            mean=(123.675, 116.28, 103.53),
                            std=(58.395, 57.12, 57.375),
                        ),
                    ])
            rgb_t = transform_norm(rgb_t)
        else:
            print("This val_transform_type not implemented!")

        items['rgb'] = rgb_t
        items['depth'] = depth_t
        return items
