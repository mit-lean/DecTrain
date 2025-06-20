import torch
import torch.nn as nn
import time
import torch.nn.functional as F
import torchvision
from libs.CoDEPS import DepthHead, ResnetEncoder, UncertaintyHead

class resnet50_monodepth2(nn.Module):
    def __init__(self, freeze_encoder=True):
        super().__init__()
        self.backbone_po_depth = ResnetEncoder(50, True)
        self.depth_head = DepthHead(self.backbone_po_depth.num_ch_enc, use_skips=True)
        self.uncertainty_head = UncertaintyHead(self.backbone_po_depth.num_ch_enc, use_skips=True)

        if freeze_encoder:
            print("[Warning] Freezing resnet50_monodepth2 encoder.")
            self.freeze_encoder()

    def forward(self, x, run_parameters, get_features=False):
        dropout_p = run_parameters['p']
        if 'activate_dropout' in run_parameters:
            # encoder, decoder, last layer
            encoder_dropout_activate = run_parameters['activate_dropout'][0]
            decoder_dropout_activate = run_parameters['activate_dropout'][1]
            last_layer_dropout_activate = run_parameters['activate_dropout'][2]
        else:
            encoder_dropout_activate = False
            decoder_dropout_activate = False
            last_layer_dropout_activate = False
        
        if 'training_dropout' in run_parameters and run_parameters['training_dropout'] == True:
            assert(encoder_dropout_activate == False)
            assert(decoder_dropout_activate == True)
            assert(last_layer_dropout_activate == False)
            assert(dropout_p == 0.2)
        
        # assert(encoder_dropout_activate == False)
        
        x_features = self.backbone_po_depth(x)
        # for feat in x_features: 
        #     print(feat.shape)
        # input("Press enter to continue...")
        x_depth, x_disp = self.depth_head(
            x_features,
            dropout_p=dropout_p,
            activate_dropout=decoder_dropout_activate,
            last_layer_activate_dropout=last_layer_dropout_activate,
            return_disparity=True
        ) # output = depth, disp if return_disparity=True
        x_logvar = self.uncertainty_head(
            x_features,
            dropout_p=dropout_p,
            activate_dropout=decoder_dropout_activate
        )
        out_f = torch.cat((x_depth, x_logvar, x_disp),1) # merge into one x 
        if get_features:
            return out_f, x_features[-1] # only return the last feature, i.e. the bottleneck feature
        else:
            return out_f
    
    def freeze_encoder(self):
        self.backbone_po_depth.eval()
        for param in self.backbone_po_depth.parameters():
            param.requires_grad = False


if __name__ == '__main__':
    run_parameters = {
        'p': 0.2,
        'activate_dropout': [False, False, True]
    }
    # random tensor with shape (3, 480, 640)
    x = torch.rand(1, 3, 480, 640)
    print('resnet50_monodepth2')
    model = resnet50_monodepth2()
    y_depth_unc, y_feats = model(x, run_parameters, True)
    print("Depth+Var Shape : ", y_depth_unc.shape)
    print("Feature Shape: ", [feat.shape for feat in y_feats] )
    # print(model)
    # input("Check the model, press Enter to continue...")