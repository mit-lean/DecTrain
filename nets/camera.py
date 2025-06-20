# camera.py
import torch
import torch.nn as nn
from libs.CoDEPS import CameraModel

class Intrinsics:
    def __init__(self, width, height, fu, fv, cu=0, cv=0, device='cpu', gen_cam_mat=False):
        self.height, self.width = height, width
        self.fu, self.fv = fu, fv  # fu, fv: focal length along the horizontal and vertical axes

        # cu, cv: optical center along the horizontal and vertical axes
        self.cu = cu if cu > 0 else (width - 1) / 2.0
        self.cv = cv if cv > 0 else (height - 1) / 2.0

        self.X_cam, self.Y_cam = None, None
        if gen_cam_mat:
            # U, V represent the homogeneous horizontal and vertical coordinates in the pixel space
            self.U = torch.arange(start=0, end=width, device=device).expand(height, width).float()
            self.V = torch.arange(start=0, end=height, device=device).expand(width, height).t().float()

            # X_cam, Y_cam represent the homogeneous x, y coordinates (assuming depth z=1) in the camera coordinate system
            self.X_cam = (self.U - self.cu) / self.fu
            self.Y_cam = (self.V - self.cv) / self.fv

        self.is_cuda = True
        self.device = device

    def to(self, device):
        if self.X_cam is not None:
            self.X_cam.data = self.X_cam.data.to(device)
            self.Y_cam.data = self.Y_cam.data.to(device)
        self.is_cuda = True
        self.device = device
        return self

    def scale(self, height, width, gen_cam_mat=False):
        # return a new set of corresponding intrinsic parameters for the scaled image
        ratio_u = float(width) / self.width
        ratio_v = float(height) / self.height
        fu = ratio_u * self.fu
        fv = ratio_v * self.fv
        cu = ratio_u * self.cu
        cv = ratio_v * self.cv
        new_intrinsics = Intrinsics(width, height, fu, fv, cu, cv, self.device, gen_cam_mat)

        return new_intrinsics
    
    def to_CameraModel(self):
        return CameraModel(self.width, self.height, self.fu, self.fv, self.cu, self.cv)

    def __str__(self):
        # return 'size=({},{})\nfocal length=({},{})\noptical center=({},{})'.format(
        return 'img=({}, {}) | fx={:.4f} | fy={:.4f} | cx={:.4f} | cy={:.4f}'.format(
            self.height, self.width, self.fv, self.fu, self.cv, self.cu)

    def __eq__(self, other):
        return self.height == other.height and self.width == other.width and \
               self.fu == other.fu and self.fv == other.fv and \
               self.cu == other.cu and self.cv == other.cv

class Pose:
    def __init__(self, R, t, device='cpu'):
        self.R = R.view(3, 3).to(device) # (3, 3)
        self.t = t.view(3, 1).to(device) # (3, 1)
        self.device = device

    def __sub__(self, p):
        # Calculate relative pose from pose p to current pose
        rel_R = torch.matmul(torch.transpose(self.R, 0, 1), p.R)
        rel_t = torch.matmul(torch.transpose(self.R, 0, 1), (p.t-self.t))
        return Pose(rel_R, rel_t, device=self.device)

    def __str__(self):
        return f'R = {str(self.R)} | t = {str(self.t)}'

    def to(self, device):
        self.R = self.R.to(device)
        self.t = self.t.to(device)
        return self
    
    def transformation_matrix(self):
        # return transformation matrix (4x4)
        return torch.cat([torch.cat([self.R, self.t], dim=1), torch.tensor([0, 0, 0, 1], device=self.device).view(1, 4)], dim=0)

def multiscale(img):
    avgpool = nn.AvgPool2d(kernel_size=2, stride=2).to(img.device)
    img1 = avgpool(img)
    img2 = avgpool(img1)
    img3 = avgpool(img2)
    img4 = avgpool(img3)
    img5 = avgpool(img4)
    return img5, img4, img3, img2, img1, img

def multiscale_intrinsics(intrinsics, img_size=(224, 224)):
    multiscale_imgs = multiscale(torch.zeros(1, 1, *img_size, device='cuda:0'))
    multiscale_intrinsics = [intrinsics.scale(img.shape[2], img.shape[3], gen_cam_mat=True) for img in multiscale_imgs]
    return multiscale_intrinsics