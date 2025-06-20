import os
import sys
import numpy as np
import torch
from torchvision import transforms as torch_transforms
from torchvision.transforms.functional import InterpolationMode
from PIL import Image, ImageEnhance
from dataloaders.dataloader import collect_batch_data
from dataloaders.landmarks import get_landmark_counts
from . import associate 
import torch.nn.functional as F

# Dataset details: https://www.cvlibs.net/datasets/kitti-360/documentation.php
class KITTI360MetaDataset:

    def __init__(self, root, mode='val', modality=['rgb', 'd', 'pose'], output_size=(224, 224), val_transform_type = "direct_resizing", split_start = 0, split_end = 1.0, skip_idx = 1, add_sway=False, sway_start_idx=0, sway_end_idx=0, num_sway = 5, brightness_scale=1.0):
        self.root = root
        self.pose_root = os.path.join(root, 'pose_per_frame')
        self.modality = modality # (rgb, d, sd, pose)
        self.output_size = output_size # (width, height)
        self.iheight, self.iwidth = 376, 1408
        # self.depth_iheight, self.depth_iwidth = 384, 1408

        self.matches = self._readDataset()
        if not (split_start == 0 and split_end == 1.0):
            if split_start > 1.0: # unit in frame
                split_start_idx = int(split_start)
            else:
                split_start_idx = min(round(split_start*len(self.matches)),len(self.matches)-1) # don't give out of range index 
            if split_end > 1.0: # unit in frame
                split_end_idx = int(split_end)
            else:
                split_end_idx = round(split_end*len(self.matches)-1)
            self.matches = self.matches[split_start_idx:split_end_idx:skip_idx]
        if add_sway:
            self.num_sway = num_sway
            self.sway_start_idx = sway_start_idx
            if (sway_end_idx == -1):
                self.sway_end_idx = len(self.matches)-1  # option to calculate based on length of matches
            else:
                self.sway_end_idx = sway_end_idx
            self.add_sway() 
        else:
            self.real_idx = list(range(0,len(self.matches))) # used for indexing pose, table
        # print(self.real_idx)
        # scale to modify brightness 
        self.brightness_scale = brightness_scale
        print("Brightness scale: " + str(self.brightness_scale))
        print("Found {} images and depth pairs in {}.".format(len(self.matches), self.root))
        if mode == 'val':
            self.transform = self.val_transform
        else:
            raise (RuntimeError("Invalid dataset type: " + mode + "\n"
                                "Supported dataset types are: val"))
        # camera intrinsics
        self.color_intrinsics = torch.from_numpy(np.loadtxt(os.path.join(self.root, 'intrinsic.txt'), delimiter=',')).float().view(1, 4)
        self.intrinsics = {
            'fx': self.color_intrinsics[0, 0],
            'fy': self.color_intrinsics[0, 1],
            'cx': self.color_intrinsics[0, 2],
            'cy': self.color_intrinsics[0, 3],
        }
        # val transforms 
        self.val_transform_type = val_transform_type # direct_resizing, resize_centercrop_resize, dinov2

    def __getraw__(self, index):
        # print("__getraw__ index: " + str(index) + ", self.real_idx[index]: " + str(self.real_idx[index])+ "\n")
        raw_items = {}
        timestamp = float(os.path.splitext(self.matches[index][0].split('.')[0].split('/')[1])[0])
        raw_items['timestamp'] = timestamp
        raw_items['real_idx'] = self.real_idx[index] # real indices for adding sway to frames (note, skipping images not enabled here)
        if 'rgb' in self.modality:
            rgb_file = os.path.join(self.root, self.matches[index][0])
            rgb = Image.open(rgb_file)
            rgb_bgr = np.array(rgb)[:, :, [2, 1, 0]] # opencv read in format is BGR, swap for landmark extraction
            raw_items['landmark_count'] = get_landmark_counts(rgb_bgr, detector='SIFT') # get landmark count for rgb image
            rgb = self.scale_brightness(rgb, factor=self.brightness_scale)
            raw_items['rgb'] = rgb
        if 'd' in self.modality:
            depth_file = os.path.join(self.root, self.matches[index][1])
            depth = Image.open(depth_file)
            raw_items['depth'] = depth
        if 'pose' in self.modality:
            trans_file = os.path.join(self.pose_root, f'{int(timestamp)}', 'translation.pt')
            rot_file = os.path.join(self.pose_root, f'{int(timestamp)}', 'rotation.pt')
            translation = torch.load(trans_file)
            rotation = torch.load(rot_file)
            raw_items['trans'] = translation
            raw_items['rot'] = rotation

        if rgb.mode != "RGB":
            raise Exception("Color image is not in RGB format")
        if depth.mode not in ["I", "I;16"]:
            raise Exception("Depth image is not in intensity format")

        return raw_items

    def __getitem__(self, index):
        raw_items = self.__getraw__(index)
        items = self.transform(raw_items)
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
        else: 
            raise Exception("Associations file not found.")
        return matches_filenames

    def add_sway(self):
        # go through matches and warp back and forth
        matches_before_adding_sway = self.matches[:self.sway_start_idx]
        matches_after_adding_sway = self.matches[self.sway_end_idx:]
        matches_sway = []
        self.real_idx = list(range(0,self.sway_start_idx))
        matches_to_add_sway = self.matches[self.sway_start_idx:self.sway_end_idx]
        # print("****** matches_to_add_sway ******")
        # print("self.sway_start_idx:" + str(self.sway_start_idx))
        # print("self.sway_end_idx-1:" + str(self.sway_end_idx-1))
        matches_sway_middle = self.matches[self.sway_start_idx+1:self.sway_end_idx-1]
        # print("****** matches_sway_middle ******")
        # print("self.sway_start_idx+1:" + str(self.sway_start_idx+1))
        # print("self.sway_end_idx-2:" + str(self.sway_end_idx-2))
        matches_to_add_sway_reversed = matches_sway_middle[::-1]
        for i in range(0,self.num_sway): 
            matches_sway = matches_sway + matches_to_add_sway
            matches_sway = matches_sway + matches_to_add_sway_reversed
            # add real indices (for looking up pose, tables indexed with real sequence idx)
            self.real_idx = self.real_idx + list(range(self.sway_start_idx,self.sway_end_idx)) + list(reversed(range(self.sway_start_idx+1,self.sway_end_idx-1)))
        # add last forward motion again
        matches_sway = matches_sway + matches_to_add_sway
        self.real_idx = self.real_idx + list(range(self.sway_start_idx,self.sway_end_idx)) + list(range(self.sway_end_idx,len(self.matches))) # original length of matches
        # concatenate matches 
        self.matches = matches_before_adding_sway + matches_sway + matches_after_adding_sway
        assert (len(self.real_idx) == len(self.matches)), "# of real idx does not match # of matches."
        return 

    def scale_brightness(self, rgb, factor):        
        if factor != 1.0:
            # object to scale brightness
            enhancer = ImageEnhance.Brightness(rgb)
            rgb = enhancer.enhance(factor)
        return rgb

    def val_transform(self, items):
        rgb, depth = np.array(items['rgb']), items['depth']
        if self.val_transform_type == "direct_resizing":
            rgb_t = F.interpolate(torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float(), self.output_size, mode="bilinear", align_corners=True).squeeze(0) # Tensor 
            items['rgb'] = rgb_t / 255.0

            # Codeps
            # transform_rgb = torch_transforms.Compose([
            #     torch_transforms.Resize(self.output_size, interpolation=torch_transforms.InterpolationMode.LANCZOS),
            #     torch_transforms.ToTensor(),
            # ])
            # rgb_t = transform_rgb(items['rgb'].convert("RGB")) # Tensor, already normalized by 255

            # depth_t = F.interpolate(torch.tensor(depth).unsqueeze(0).unsqueeze(0), size=self.output_size, mode='nearest').squeeze(0).float()
            depth_pil_size = (self.output_size[1], self.output_size[0]) # pil size is (width, height)
            depth_t = torch.tensor(np.array(depth.resize(size=depth_pil_size, resample=Image.NEAREST)).astype(np.int32)).unsqueeze(0)
        elif self.val_transform_type == "dinov2":
            rgb_t = F.interpolate(torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float(), self.output_size, mode="bilinear", align_corners=True).squeeze(0) # Tensor 
            # add additional step on RGB after resizing 
            transform_norm = torch_transforms.Compose([
                        # lambda x: 255.0 * x[:3], # Discard alpha component and scale by 255
                        torch_transforms.Normalize(
                            mean=(123.675, 116.28, 103.53),
                            std=(58.395, 57.12, 57.375),
                        ),
                    ])
            rgb_t = transform_norm(rgb_t)

            # Codeps
            # transform_rgb = torch_transforms.Compose([
            #     torch_transforms.Resize(self.output_size, interpolation=torch_transforms.InterpolationMode.LANCZOS),
            #     torch_transforms.ToTensor(),
            # ])
            # rgb_t = transform_rgb(items['rgb'].convert("RGB"))
            # transform_norm = torch_transforms.Compose([
            #             lambda x: 255.0 * x, # scale by 255
            #             torch_transforms.Normalize(
            #                 mean=(123.675, 116.28, 103.53),
            #                 std=(58.395, 57.12, 57.375),
            #             ),
            #         ])
            # rgb_t = transform_norm(rgb_t)

            # depth_t = torch.tensor(np.array(depth.resize(size=self.output_size, resample=Image.NEAREST)).astype(np.int16)).unsqueeze(0)
            depth_pil_size = (self.output_size[1], self.output_size[0]) # pil size is (width, height)
            depth_t = torch.tensor(np.array(depth.resize(size=depth_pil_size, resample=Image.NEAREST)).astype(np.int32)).unsqueeze(0)
            items['rgb'] = rgb_t
        else:
            print("This val transform not implemented for KITTI-360!")
        if rgb_t.shape[1] != depth_t.shape[1] or rgb_t.shape[2] != depth_t.shape[2]:
            print("RGB shape: " + str(rgb_t.shape) + ", Depth shape: " + str(depth_t.shape))
            raise Exception("Color and depth image do not have the same resolution.")
        items['depth'] = depth_t / 1000.0 # adjust scaling correctly [m]
        return items