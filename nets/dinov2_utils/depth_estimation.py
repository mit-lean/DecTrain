''' 
Functions from dinov2/notebooks/depth_estimation.ipynb as .py file to import in rest of the project. 

Credit: DinoV2, Maxime Oquab et al. 
'''
import math
import itertools
from functools import partial

import torch
import torch.nn.functional as F

from libs.dinov2.dinov2.eval.depth.models import build_depther
import urllib


class CenterPadding(torch.nn.Module):
    def __init__(self, multiple):
        super().__init__()
        self.multiple = multiple

    def _get_pad(self, size):
        new_size = math.ceil(size / self.multiple) * self.multiple
        pad_size = new_size - size
        pad_size_left = pad_size // 2
        pad_size_right = pad_size - pad_size_left
        return pad_size_left, pad_size_right

    # @torch.inference_mode()
    def forward(self, x):
        pads = list(itertools.chain.from_iterable(self._get_pad(m) for m in x.shape[:1:-1]))
        output = F.pad(x, pads)
        return output

def create_depther(cfg, backbone_model, backbone_size, head_type):
    train_cfg = cfg.get("train_cfg")
    test_cfg = cfg.get("test_cfg")
    depther = build_depther(cfg.model, train_cfg=train_cfg, test_cfg=test_cfg)

    depther.backbone.forward = partial(
        backbone_model.get_intermediate_layers,
        n=cfg.model.backbone.out_indices,
        reshape=True,
        return_class_token=cfg.model.backbone.output_cls_token,
        norm=cfg.model.backbone.final_norm,
    )

    if hasattr(backbone_model, "patch_size"):
        depther.backbone.register_forward_pre_hook(lambda _, x: CenterPadding(backbone_model.patch_size)(x[0]))

    return depther

def load_config_from_url(url: str) -> str:
    with urllib.request.urlopen(url) as f:
        return f.read().decode()
    
def disp_to_depth(disp: torch.Tensor, min_depth: float = 0.1, max_depth: float = 100) -> torch.Tensor:
    min_disp = 1 / max_depth
    max_disp = 1 / min_depth
    scaled_disp = min_disp + (max_disp - min_disp) * disp
    depth = 1 / scaled_disp
    return depth

def depth_to_disp(depth: torch.Tensor, min_depth: float = 0.1, max_depth: float = 100) -> torch.Tensor:
    min_disp = 1 / max_depth
    max_disp = 1 / min_depth
    scaled_disp = 1 / depth 
    disp = (scaled_disp-min_disp)/(max_disp-min_disp)
    return disp