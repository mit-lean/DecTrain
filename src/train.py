import os
import sys
import time
import csv
import numpy as np 
from collections import namedtuple
import torch
import torch.backends.cudnn as cudnn
import torch.optim
from copy import deepcopy

import nets
import nets.criteria
import nets.metrics
import dataloaders.sun3d_dataloader as sun3d_dataloader
import dataloaders.scannet_dataloader as scannet_dataloader
import dataloaders.kitti360_dataloader as kitti360_dataloader

from nets.camera import Intrinsics, multiscale_intrinsics
from . import acquisition
from . import buffer 
from libs.uncertainty_from_motion.src import UncertaintyFromMotion, run_UfM_on_frame
from nets.dinov2_utils import create_depther, load_config_from_url, depth_to_disp

import pickle

# dinov2 imports 
import mmcv
from mmcv.runner import load_checkpoint
'''
Description: sets seeds for reproducibility
Input:       
    seed:                   seed number
''' 
def set_reproducible_seeds(seed):
    import random 
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    return
set_reproducible_seeds(seed=0)

'''
Description: creates data loaders for online training and validation datasets 
Input:       
    run_parameters:         run_parameters (loaded in from yaml) 
Output:
train_loaders, val_target_loaders, train_dataset_length
    train_loaders:          list of data loaders for training 
    val_target_loaders:            list of data loaders for validation
    train_dataset_length:   return sum of all training dataset lengths
''' 
def create_data_loaders(run_parameters):
    # create data_loaders
    train_dirs = run_parameters['train_dir'] 
    val_target_dirs = run_parameters['val_target_dir']
    # initialize empty containers to return 
    train_datasets = []
    val_target_datasets = []
    train_loaders = []
    val_target_loaders = []
    intrinsics_seqs = []
    train_dataset_length = 0
    
    # check run_parameters configurations correct 
    assert len(run_parameters['train_split_start']) == len(train_dirs)
    assert len(run_parameters['train_split_end']) == len(train_dirs)
    assert len(run_parameters['val_target_split_start']) == len(val_target_dirs)
    assert len(run_parameters['val_target_split_end']) == len(val_target_dirs)
    # make train data_loaders for each dataset 
    for i in range(len(train_dirs)):
        train_dir = train_dirs[i]
        # set up training datasets for each sequence in stitched sequence
        print("Training directory: {}".format(train_dir))
        if run_parameters['train_dataset'][i] == 'sun3d_meta':
            # for self-supervised learning, turn-off data augmentation for now (due to pose groundtruth)
            dataset_ = sun3d_dataloader.SUN3DMetaDataset(root=train_dir, output_size=(run_parameters['height'], run_parameters['width']), val_transform_type = run_parameters['val_transform_type'], modality=run_parameters['modality'], split_start=run_parameters['train_split_start'][i], split_end=run_parameters['train_split_end'][i], skip_idx=run_parameters['skip_idx'][i])
            train_datasets.append(dataset_)
            train_dataset_length += len(train_datasets[-1])
            # overwrite intrinsics for sun3d dataset
            print("Writing intrinsics for SUN3D dataset")
            # intrinsics loaded in dataloader from text file
            intrinsics_seqs.append(multiscale_intrinsics(Intrinsics(dataset_.iwidth, dataset_.iheight, dataset_.intrinsics['fx'], dataset_.intrinsics['fy'], dataset_.intrinsics['cx'], dataset_.intrinsics['cy'], device=run_parameters['device'][0]), img_size=(run_parameters['height'], run_parameters['width'])))
        elif run_parameters['train_dataset'][i] == 'scannet_meta':
            # for self-supervised learning, turn-off data augmentation for now (due to pose groundtruth)
            dataset_ = scannet_dataloader.ScanNetMetaDataset(root=train_dir, output_size=(run_parameters['height'], run_parameters['width']), val_transform_type = run_parameters['val_transform_type'], modality=run_parameters['modality'], split_start=run_parameters['train_split_start'][i], split_end=run_parameters['train_split_end'][i], skip_idx=run_parameters['skip_idx'][i], add_sway=run_parameters['add_sway'], sway_start_idx=run_parameters['sway_start_idx'], sway_end_idx=run_parameters['sway_end_idx'], num_sway = run_parameters['num_sway'], brightness_scale=run_parameters['brightness_scale'][i])
            train_datasets.append(dataset_)
            train_dataset_length += len(train_datasets[-1])
            print("Writing intrinsics for ScanNet dataset")
            # intrinsics loaded in dataloader from text file
            intrinsics_seqs.append(multiscale_intrinsics(Intrinsics(dataset_.iwidth, dataset_.iheight, dataset_.intrinsics['fx'], dataset_.intrinsics['fy'], dataset_.intrinsics['cx'], dataset_.intrinsics['cy'], device=run_parameters['device'][0]), img_size=(run_parameters['height'], run_parameters['width'])))
        elif run_parameters['train_dataset'][i] == 'kitti360_meta':
            # for self-supervised learning, turn-off data augmentation for now (due to pose groundtruth)
            dataset_ = kitti360_dataloader.KITTI360MetaDataset(root=train_dir, output_size=(run_parameters['height'], run_parameters['width']), val_transform_type = run_parameters['val_transform_type'], modality=run_parameters['modality'], split_start=run_parameters['train_split_start'][i], split_end=run_parameters['train_split_end'][i], skip_idx=run_parameters['skip_idx'][i], add_sway=run_parameters['add_sway'], sway_start_idx=run_parameters['sway_start_idx'], sway_end_idx=run_parameters['sway_end_idx'], num_sway = run_parameters['num_sway'], brightness_scale=run_parameters['brightness_scale'][i])
            train_datasets.append(dataset_)
            train_dataset_length += len(train_datasets[-1])
            print("Writing intrinsics for KITTI-360 dataset")
            # intrinsics loaded in dataloader from text file
            intrinsics_seqs.append(multiscale_intrinsics(Intrinsics(dataset_.iwidth, dataset_.iheight, dataset_.intrinsics['fx'], dataset_.intrinsics['fy'], dataset_.intrinsics['cx'], dataset_.intrinsics['cy'], device=run_parameters['device'][0]), img_size=(run_parameters['height'], run_parameters['width'])))
        else:
            raise RuntimeError("Dataset not found. The dataset must be either of sun3d_meta, scannet_meta, or kitti360_meta.")
        # convert percent of sequence to actual sequence index 
        if run_parameters['train_split_end'][i] <= 1.0:
            run_parameters['train_split_end'][i] = len(train_datasets[-1])
    # add camera calibration to run parameters 
    run_parameters['fx'] = [intrinsics_seqs[x][-1].fu.item() for x in range(len(train_dirs))]
    run_parameters['fy'] = [intrinsics_seqs[x][-1].fv.item() for x in range(len(train_dirs))]
    run_parameters['cx'] = [intrinsics_seqs[x][-1].cu.item() for x in range(len(train_dirs))]
    run_parameters['cy'] = [intrinsics_seqs[x][-1].cv.item() for x in range(len(train_dirs))]
    # set up validation target datasets
    for i in range(len(val_target_dirs)):
        val_dir = val_target_dirs[i]
        print("Validation target directory: {}".format(val_dir))
        if run_parameters['val_target_dataset'][i] == 'sun3d_meta':
        # for self-supervised learning, turn-off data augmentation for now (due to pose groundtruth)
           val_target_datasets.append(sun3d_dataloader.SUN3DMetaDataset(root=val_dir, output_size=(run_parameters['height'], run_parameters['width']), val_transform_type = run_parameters['val_transform_type'], modality=run_parameters['modality'], split_start=run_parameters['val_target_split_start'][i], split_end=run_parameters['val_target_split_end'][i]))
        elif run_parameters['val_target_dataset'][i] == 'scannet_meta':
            # for self-supervised learning, turn-off data augmentation for now (due to pose groundtruth)
           val_target_datasets.append(scannet_dataloader.ScanNetMetaDataset(root=val_dir, output_size=(run_parameters['height'], run_parameters['width']), val_transform_type = run_parameters['val_transform_type'], modality=run_parameters['modality'], split_start=run_parameters['val_target_split_start'][i], split_end=run_parameters['val_target_split_end'][i]))
        elif run_parameters['val_target_dataset'][i] == 'kitti360_meta':
            # for self-supervised learning, turn-off data augmentation for now (due to pose groundtruth)
           val_target_datasets.append(kitti360_dataloader.KITTI360MetaDataset(root=val_dir, output_size=(run_parameters['height'], run_parameters['width']), val_transform_type = run_parameters['val_transform_type'], modality=run_parameters['modality'], split_start=run_parameters['val_target_split_start'][i], split_end=run_parameters['val_target_split_end'][i]))
        else:
            raise RuntimeError("This target validation dataset not found. The dataset must be either of sun3d_meta, scannet_meta, or kitti360_meta.")
    # construct train loaders, set batch size to be 1 for training -- batch size handled by buffer instead of dataloader 
    for train_dataset in train_datasets:
        train_loaders.append(torch.utils.data.DataLoader(
            train_dataset, batch_size=1, shuffle=run_parameters['shuffle_train'],
            num_workers=run_parameters['workers'], pin_memory=True, sampler=None,
            worker_init_fn=lambda work_id:np.random.seed(work_id))) 
            # worker_init_fn ensures different sampling patterns for each data loading thread
    # construct validation loaders, set batch size to be 1 for validation
    for val_target_dataset in val_target_datasets:
        val_target_loaders.append(torch.utils.data.DataLoader(val_target_dataset,
            batch_size=1, shuffle=False, num_workers=run_parameters['workers'], pin_memory=True))
    print("=> data loaders created.")
    return train_loaders, val_target_loaders, train_dataset_length, intrinsics_seqs, run_parameters

'''
Description: sets loss function based on configuration file 
Input:       
    run_parameters:         run parameters loaded in from yaml 
Output:      
    loss_fn:                loss function
''' 

def make_loss_function(run_parameters):
    if run_parameters['loss'] == "l2":
        loss_fn = nets.criteria.MaskedMSELoss().cuda()   
    elif run_parameters['loss'] == "l1":
        loss_fn = nets.criteria.MaskedL1Loss().cuda()
    elif run_parameters['loss'] == "heteroscedastic":
        loss_fn = nets.criteria.HeteroscedasticLoss().cuda()
    elif run_parameters['loss'] == "multiview_self_supervised":
        loss_fn = nets.criteria.MultiviewSelfSupervisedLoss(
            loss_weights=run_parameters['self_supervised_loss_weights'],
            depth_mode=run_parameters['self_supervised_depth_mode'],
            reproj_mode=run_parameters['self_supervised_reproj_mode'],
            combine=run_parameters['self_supervised_combine'],
            weighted=run_parameters['self_supervised_weighted'],
            sample_settings=run_parameters['self_supervised_sample_settings'],
            enable_nll=run_parameters['self_supervised_enable_nll']
        ).cuda()
    elif run_parameters['loss'] == "codeps_depth_loss":
        loss_fn = nets.criteria.CoDEPS_DepthLoss(
            img_width=run_parameters['width'], 
            img_height=run_parameters['height'], 
            recon_loss_weight=run_parameters['self_supervised_loss_weights'][1],
            smth_loss_weight=run_parameters['self_supervised_loss_weights'][2],
            scales=5, 
            device=run_parameters['device'][0]
        ).cuda()
    elif run_parameters['loss'] == "nll_sep_aleatoric_decoder":
        loss_fn = nets.criteria.NLLSepAleatoricDecoderLoss().cuda()
    else:
        raise NotImplementedError('Loss function: {}'.format(run_parameters['loss']))
    return loss_fn

def freeze_encoder(encoder):
    encoder.eval()
    for param in encoder.parameters():
        param.requires_grad = False
    return encoder

'''
Description: loads trained models in a list 
Input:       
    run_parameters:         run parameters loaded in from yaml 
Output:      
    models:                 list of loaded models
''' 
def load_trained_model(run_parameters):
    models = []
    if run_parameters['arch'] != 'dinov2': 
        for model_counter, model_pth in enumerate(run_parameters['resume']):
            if (model_counter < run_parameters['ensemble_size']): # only add from resume model paths up to ensemble size
                chkpt_path = str(model_pth)
                assert os.path.isfile(chkpt_path), \
                    "=> no checkpoint found at '{}'".format(chkpt_path)
                print("=> loading checkpoint '{}'".format(chkpt_path))
                if run_parameters['arch'] == 'resnet101_monodepth2':
                    model = nets.resnet101_monodepth2(freeze_encoder=run_parameters['freeze_encoder'])
                    model.load_state_dict(torch.load(chkpt_path, map_location=torch.device(run_parameters['device'][model_counter]))['model_state_dict']) 
                elif run_parameters['arch'] == 'resnet18_monodepth2':
                    model = nets.resnet18_monodepth2(freeze_encoder=run_parameters['freeze_encoder'])
                    model.load_state_dict(torch.load(chkpt_path, map_location=torch.device(run_parameters['device'][model_counter]))['model_state_dict']) 
                elif run_parameters['arch'] == 'resnet50_monodepth2':
                    model = nets.resnet50_monodepth2(freeze_encoder=run_parameters['freeze_encoder'])
                    model.load_state_dict(torch.load(chkpt_path, map_location=torch.device(run_parameters['device'][model_counter]))['model_state_dict']) 
                else:
                    print("This architecture not defined!")
                models.append(model)
    else:
        # load in pretrained DinoV2-small with specified encoder and decoder
        backbone_archs = {  "small": "vits14",
                            "base": "vitb14",
                            "large": "vitl14",
                            "giant": "vitg14"}
        
        backbone_arch = backbone_archs[run_parameters['dinov2_backbone_size']]
        backbone_name = f"dinov2_{backbone_arch}"
        
        # load pretrained dinov2 from torch hub in eval mode and cuda
        backbone_model = torch.hub.load(repo_or_dir="facebookresearch/dinov2", model=backbone_name)
        backbone_model.eval()
        backbone_model.cuda()

        # freeze backbone (encoder)
        backbone_model = freeze_encoder(backbone_model)
        # change the following if want to compare to other dinov2 setups
        HEAD_DATASET="nyu" # in ("nyu", "kitti")
        HEAD_TYPE="dpt"    # in ("linear", "linear4", "dpt")

        DINOV2_BASE_URL = "https://dl.fbaipublicfiles.com/dinov2"
        head_config_url = f"{DINOV2_BASE_URL}/{backbone_name}/{backbone_name}_{HEAD_DATASET}_{HEAD_TYPE}_config.py"
        head_checkpoint_url = f"{DINOV2_BASE_URL}/{backbone_name}/{backbone_name}_{HEAD_DATASET}_{HEAD_TYPE}_head.pth"
        cfg_str = load_config_from_url(head_config_url)
        cfg = mmcv.Config.fromstring(cfg_str, file_format=".py")
        # make encoder + decoder 
        model = create_depther(
            cfg,
            backbone_model=backbone_model,
            backbone_size=run_parameters['dinov2_backbone_size'],
            head_type=HEAD_TYPE,
        )

        # load weights for decoder head 
        load_checkpoint(model, head_checkpoint_url, map_location="cuda:0")
        model.eval()
        model.cuda()
        models.append(model)      
    return models 

def set_model_mode(model, mode):
    if not isinstance(model, tuple):
        if mode == 'train':
            model.train()
        elif mode == 'eval':
            model.eval() 
        else:
            print("Invalid mode for model.")
    else: # tuple (depth_model, aleatoric decoder)
        if mode == 'train':
            model[0].train()
            model[1].eval() # separate aleatoric decoder fixed
        elif mode == 'eval':
            model[0].eval() 
            model[1].eval()
        else:
            print("Invalid mode for model.")
    return 

def get_depth_autoencoder_at_idx(models, model_idx):
    if not isinstance(models[model_idx], tuple):
        return models[model_idx]
    else: # tuple (depth_model, aleatoric decoder)
        return models[model_idx][0]
'''
Description: makes new architectures and returns in a list 
Input:       
    run_parameters:         run parameters loaded in from yaml 
Output:      
    models:                 list of initialized new models
''' 
def make_architecture(run_parameters):
    # online training: start from a pre-trained model
    if run_parameters['resume'] or run_parameters['arch'] == "dinov2": # not empty string or dinov2 pretrained model
        models = load_trained_model(run_parameters)
    else: 
        raise RuntimeError("Pre-trained model required for online training. Please specify a pre-trained model in the config file.")
    return models

'''
Description: makes new optimizers for each model in list and returns optimizers
Input:       
    run_parameters:         run parameters loaded in from yaml 
    models:                 list of models
Output:      
    optimizers:             list of optimizers
''' 
def make_optimizers(run_parameters, models, train_dataset_length):
    optimizers = []
    for m in range(run_parameters['ensemble_size']):
        if run_parameters['optimizer'] == "SGD": 
            optimizers.append(torch.optim.SGD(get_depth_autoencoder_at_idx(models, m).parameters(), run_parameters['learning_rate'], \
                momentum=run_parameters['momentum'], weight_decay=run_parameters['weight_decay']))
        elif run_parameters['optimizer'] == "Adam":
            optimizers.append(torch.optim.Adam(get_depth_autoencoder_at_idx(models, m).parameters(), run_parameters['learning_rate']))
        elif run_parameters['optimizer'] == "RMSprop":
            optimizers.append(torch.optim.RMSprop(get_depth_autoencoder_at_idx(models, m).parameters(), run_parameters['learning_rate'], \
                    weight_decay=run_parameters['weight_decay']))
        else:
            print("This optimizer not defined!")
        
    return optimizers

'''
Description: makes new csv files for storing results
Input:       
    run_parameters:         run parameters loaded in from yaml 
    fieldnames:             column names in csv files 
Output:      
    output_dir:             output directory from command line arguments 
    train_csv:              csv file for average training stats 
    train_per_image_csv:       csv file fore training stats per image
    val_target_csv:         csv file for validation stats on target 
''' 
def make_results_files(run_parameters, fieldnames, output_dir):
    # create results folder, if not already exists
    train_csv = os.path.join(output_dir, 'train.csv')
    train_per_image_csv = os.path.join(output_dir, 'train_per_image.csv') # stats per image
    train_all_dnns_csv = []
    train_per_image_all_dnns_csv = []
    for i in range(run_parameters['ensemble_size']):
        train_all_dnns_csv.append(os.path.join(output_dir, 'train_dnn_' + str(i) + '.csv'))
        train_per_image_all_dnns_csv.append(os.path.join(output_dir, 'train_per_image_' + str(i) + '.csv'))
        with open(train_all_dnns_csv[i], 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
        with open(train_per_image_all_dnns_csv[i], 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
    val_target_csv = os.path.join(output_dir, 'val_target.csv')
    # create new csv files with only header
    with open(train_csv, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
    with open(train_per_image_csv, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
    with open(val_target_csv, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
    return train_csv, train_per_image_csv, train_all_dnns_csv, train_per_image_all_dnns_csv, val_target_csv

'''
Description: write out stats to csv
Input:       
    csv_name:               name of csv file writing stats out to 
    avg:                    Result object that stores average stats
    fieldnames:             column names in csv files 
    dataset_idx:            dataset index 
    seq_idx:                sequence index
    num_weight_updates:     number of weight updats to DNN so far
    num_decisions_train:    number of times acquisition function returns decision to train
''' 
def write_avg_metrics_to_file(csv_name, avg, fieldnames, dataset_idx, seq_idx, num_weight_updates, num_decisions_train, metric_gain, epoch):
    with open(csv_name, 'a') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writerow({'epoch': epoch, 'dataset_idx': dataset_idx, 'seq_idx': seq_idx, 'num_weight_updates': num_weight_updates, 'num_decisions_train': num_decisions_train, 'mse': avg.mse, 'rmse': avg.rmse, 'absrel': avg.absrel, 'lg10': avg.lg10,
            'mae': avg.mae, 'delta1': avg.delta1, 'delta2': avg.delta2, 'delta3': avg.delta3, 'delta1_lit': avg.delta1_lit, 'delta2_lit': avg.delta2_lit, 'delta3_lit': avg.delta3_lit,
            'gpu_time': avg.gpu_time, 'data_time': avg.data_time, 'loss': avg.loss, 'nll': avg.nll, 'aleatoric_nll': avg.aleatoric_nll, 'epistemic_nll': avg.epistemic_nll, 'avg_aleatoric_unc': avg.avg_aleatoric_unc, 'avg_epistemic_unc': avg.avg_epistemic_unc, 
            'avg_depth_unmasked': avg.avg_depth_unmasked, 'avg_depth_masked': avg.avg_depth_masked, 'avg_depth_gt_masked': avg.avg_depth_gt_masked, 
            'depth_unmasked_median': avg.depth_unmasked_median, 'depth_masked_median': avg.depth_masked_median, 'depth_gt_masked_median': avg.depth_gt_masked_median, 'aleatoric_unc_unmasked_median': avg.aleatoric_unc_unmasked_median, 'epistemic_unc_unmasked_median': avg.epistemic_unc_unmasked_median,
            'depth_unmasked_max': avg.depth_unmasked_max, 'depth_masked_max': avg.depth_masked_max, 'depth_gt_masked_max': avg.depth_gt_masked_max, 'aleatoric_unc_unmasked_max': avg.aleatoric_unc_unmasked_max, 'epistemic_unc_unmasked_max': avg.epistemic_unc_unmasked_max,
            'depth_unmasked_min': avg.depth_unmasked_min, 'depth_masked_min': avg.depth_masked_min, 'depth_gt_masked_min': avg.depth_gt_masked_min, 'aleatoric_unc_unmasked_min': avg.aleatoric_unc_unmasked_min, 'epistemic_unc_unmasked_min': avg.epistemic_unc_unmasked_min,
            'depth_unmasked_var': avg.depth_unmasked_var, 'depth_masked_var': avg.depth_masked_var, 'depth_gt_masked_var': avg.depth_gt_masked_var, 'aleatoric_unc_unmasked_var': avg.aleatoric_unc_unmasked_var, 'epistemic_unc_unmasked_var': avg.epistemic_unc_unmasked_var,
            'delta1_diff': metric_gain['delta1'], 'delta1_lit_diff': metric_gain['delta1_lit'], 'rmse_diff': metric_gain['rmse'], 'absrel_diff': metric_gain['absrel'], 'loss_diff': metric_gain['loss']
            })
    return 

'''
Description: saves per-frame results
Input:       
    run_parameters:         run parameters loaded in from yaml 
    i:                      sequence index
    input:                  RGB input tensor
    target:                 target tensor
    pred:                   prediction tensor
    aleatoric_unc:          aleatoric variance tensor
    epistemic_unc:          epistemic variance tensor
    output_dir:             output directory 
    all_outputs:            all outputs tensor 
''' 
def save_output_files(run_parameters, i, input, target, pred, aleatoric_unc, epistemic_unc, output_dir, all_outputs = None):
    output_uncertainty_dir = output_dir+"/per_frame_data/"+str(i)
    os.makedirs(os.path.dirname(output_uncertainty_dir+"/gt.npy"), exist_ok=True)
    np.save(output_uncertainty_dir+"/gt.npy",target.data.cpu().data.numpy())
    os.makedirs(os.path.dirname(output_uncertainty_dir+"/rgb_input.npy"), exist_ok=True)
    np.save(output_uncertainty_dir+"/rgb_input.npy",np.squeeze(input.data.cpu().data.numpy())[0:3,:,:])
    # save prediction, aleatoric uncertainty, epistemic uncertainty, error
    os.makedirs(os.path.dirname(output_uncertainty_dir+"/pred.npy"), exist_ok=True)
    np.save(output_uncertainty_dir+"/pred.npy",pred.data.cpu().data.numpy())
    os.makedirs(os.path.dirname(output_uncertainty_dir+"/aleatoric_uncertainty.npy"), exist_ok=True)
    np.save(output_uncertainty_dir+"/aleatoric_uncertainty.npy",aleatoric_unc.data.cpu().data.numpy())
    os.makedirs(os.path.dirname(output_uncertainty_dir+"/epistemic_uncertainty.npy"), exist_ok=True)
    np.save(output_uncertainty_dir+"/epistemic_uncertainty.npy",epistemic_unc.data.cpu().data.numpy())
    os.makedirs(os.path.dirname(output_uncertainty_dir+"/error.npy"), exist_ok=True)
    np.save(output_uncertainty_dir+"/error.npy",(target-pred).abs().data.cpu().data.numpy())
    return 

'''
Description: saves Torch models
Input:       
    models:                 run parameters loaded in from yaml 
    output_dir:             output directory 
    epoch:                  epoch number (not buffer epoch)
''' 
def save_models(models, output_dir, epoch = 1):
    for m, model in enumerate(models):
        checkpoint_filename = os.path.join(output_dir, 'dnn-' + str(m) + '-checkpoint-' + str(epoch) + '.pth')
        torch.save(get_depth_autoencoder_at_idx(models, m).state_dict(), checkpoint_filename)
    return 

'''
Description: prints training stats
Input:       
    dataset_idx:            dataset_index
    epoch:                  epoch number (not buffer epoch)
    i:                      sequence index
    len_train_loader:       number of images in current training dataset 
    data_time:              time for loading in data 
    gpu_time:               time on GPU for training or inference
    result:                 Result object with results
    average_meter:          AverageMeter object with average results
''' 
def print_training_stats(dataset_idx, epoch, i, len_train_loader, result, average_meter):
    output_log = '\r Train dataset idx: {dataset_idx}, Train Epoch: {0} [{1}/{2}]: RMSE={result.rmse:.2f}({average.rmse:.2f}), MAE={result.mae:.2f}({average.mae:.2f}), Delta1={result.delta1:.3f}({average.delta1:.3f}), Delta1_lit={result.delta1_lit:.3f}({average.delta1_lit:.3f}), Loss={result.loss:.3f}({average.loss:.3f})'.format(
                  epoch, i+1, len_train_loader,result=result, average=average_meter.average(), dataset_idx=dataset_idx)
    sys.stdout.write(output_log)
    sys.stdout.flush()
    return

'''
Description: collect data for loss computation
Input:       
    run_parameters          run parameters loaded in from yaml 
    **kwargs                keyword arguments
Output:      
    criteria_data           collected data
''' 
def collect_criteria_data(run_parameters, **kwargs):
    required_keys = []
    if run_parameters['loss'] in ['multiview_self_supervised', 'codeps_depth_loss']:
        required_keys = ['pred', 'buffer', 'intrinsics', 'batch_queries', 'device']
    elif run_parameters['loss'] in ['l2', 'l1', 'heteroscedastic']:
        required_keys = ['pred', 'target', 'device']
    else:
        raise NotImplementedError('Loss function: {}'.format(run_parameters['loss']))
    CriteriaData = namedtuple('CriteriaData', required_keys)
    required_args = {k: v for k, v in kwargs.items() if k in required_keys}
    assert len(required_args.keys()) == len(required_keys), \
        f'Criteria data key mismatch: required keys = {required_keys}, collected keys = {list(required_args.keys())}'
    criteria_data = CriteriaData(**required_args)
    return criteria_data

'''
Description: train on buffer 
Input:       
    run_parameters:         run parameters loaded in from yaml 
    idx:                    sequence index
    model:                  model to train
    loss_fn:                loss function
    optimizer:              optimizer
    online_buffer           Buffer object to train on
Output:      
    loss:                   loss after training on buffer for number of buffer epochs
    gpu_time                time on GPU for training or inference
    num_weight_updates:     number of weight updates to DNN so far
''' 
def train_on_buffer(run_parameters, idx, model, loss_fn, optimizer, train_batch_data, device, intrinsics_seq, policy_data_extractor=None):
    num_weight_updates = 0 
    # set to train/eval mode based on config (default train mode, which will update running statistics for batch norm, and also enable dropout if specified)
    if run_parameters['dnn_mode_for_training'] == 'train':
        set_model_mode(model, mode='train')
    else: # if == 'eval'
        set_model_mode(model, mode='eval')
    # record which images are used for training
    trained_img_idxs = []
    for step in range(run_parameters['buffer_num_epochs']):
        for shuffled_keys_batch, training_data, sample_buffer in train_batch_data[step]:
            training_data, data_time = load_input_target(training_data, device=device)
            # run forward inference on image and parse results
            output, pred, aleatoric_unc, epistemic_unc, inference_time, all_outputs, _, _ = forward_inference(run_parameters, [model], training_data['rgb'], [device])
            # loss computation
            criteria_data = collect_criteria_data(
                run_parameters=run_parameters,
                target=training_data['depth'],
                pred=output,
                buffer=sample_buffer,
                intrinsics=intrinsics_seq,
                batch_queries=shuffled_keys_batch,
                device=run_parameters['device'][0]
            )
            gpu_start = time.time()
            loss = loss_fn(criteria_data)
            
            # train when loss is available (cannot BP when no near view exists)
            if loss != 0:
                optimizer.zero_grad()
                # compute gradient and do SGD step
                loss.backward()
                # gradient clipping, norm of grad < 1
                if not isinstance(model, tuple):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1) 
                else: # separate aleatoric decoder tuple
                    torch.nn.utils.clip_grad_norm_(model[0].parameters(), 1) 
                optimizer.step()
                num_weight_updates+=1
                # for recoding loss
            loss = loss.detach().cpu().item()
            
            trained_img_idxs.extend(shuffled_keys_batch.keys())
            loss_post = None
            if run_parameters['acquisition_type'] == "balance_compute_accuracy_gain_policy" or policy_data_extractor.stats_collection_on or run_parameters['record_groundtruth_metric_gain']:
                with torch.no_grad():
                    output_post, pred_post, aleatoric_unc_post, epistemic_unc_post, inference_time_post, all_outputs_post, _, _ = forward_inference(run_parameters, [model], training_data['rgb'], [device])
                    # loss computation
                    criteria_data_post = collect_criteria_data(
                        run_parameters=run_parameters,
                        target=training_data['depth'],
                        pred=output_post,
                        buffer=sample_buffer,
                        intrinsics=intrinsics_seq,
                        batch_queries=shuffled_keys_batch,
                        device=run_parameters['device'][0]
                    )
                    loss_post = loss_fn(criteria_data_post)
                    loss_post = loss_post.detach().cpu().item()
                
    # set to eval mode, as externally eval mode
    set_model_mode(model, mode='eval')
    torch.cuda.synchronize()
    gpu_time = time.time() - gpu_start
    return loss, gpu_time, num_weight_updates, trained_img_idxs, loss_post

'''
Description: parse aleatoric variance from DNN output
Input:       
    run_parameters:         run parameters loaded in from yaml 
    output:                 output tensor
Output:      
    aleatoric_unc:          aleatoric variance tensor
'''
def parse_aleatoric_variance(run_parameters, output):
    if run_parameters['arch'] != "dinov2":
        if run_parameters['loss'] in ['multiview_self_supervised', 'codeps_depth_loss', 'l1']:
            log_var = torch.reshape(output[:,1,:,:],(output.shape[0],1,output.shape[2],output.shape[3]))
            aleatoric_unc = torch.exp(log_var)
    else:
        aleatoric_unc = torch.reshape(output[:,1,:,:],(output.shape[0],1,output.shape[2],output.shape[3]))
    return aleatoric_unc

def check_which_run_parameters_to_use(run_parameters, run_parameters_no_dropout_one_inf):
    # check which run_parameters to use -- specifically, for MCDropout, need to run no dropout and only one inference for main prediction. For all other methods, can run with given configuration
    if run_parameters['unc_method'] == 'mc' or run_parameters['unc_method'] == 'mc-last' or run_parameters['unc_method'] == 'mc-ufm':
        run_parameters_to_use = run_parameters_no_dropout_one_inf
    else:
        run_parameters_to_use = run_parameters
    return run_parameters_to_use
'''
Description: run forward inference on DNN
Input:       
    run_parameters:         run parameters loaded in from yaml 
    models:                 list of loaded models
    input:                  RGB input tensor
Output:      
    output:                 DNN output tensor of last model in list
    pred:                   depth output tensor
    aleatoric_unc:          aleatoric variance tensor
    epistemic_unc:          epistemic variance tensor  
    inference_time:         DNN inference time 
    all_outputs:            all DNN outputs for all models in list
'''
def forward_inference(run_parameters, models, input, devices):
    inference_start = time.time()
    all_outputs = [] # return each ensemble member's individual prediction
    all_features = [] # return each ensemble member's individual features
    n = 0 # counter for number of inferences 
    depth_sum, aleatoric_var_sum = 0, 0 # initialize sums for mean calculation
    for m, (model, device) in enumerate(zip(models, devices)): # compute average prediction, epistemic uncertainty, and aleatoric uncertainty 
        input = input.to(device)
        for i in range(run_parameters['num_inferences']):
            start_inf = time.time()
            if run_parameters['arch'] != "dinov2":
                output, features = model(input, run_parameters, get_features=True) # passing in run_parameters for dropout and loss parameters
            else: # dinov2 specific forward pass 
                # use dinov2 for inference only
                depth_pred, aleatoric_var, features = model.whole_inference_with_features_and_aleatoric_var(input, img_meta=None, rescale=True, mcd_p=run_parameters['p'], profile_target=run_parameters["profile_target"])
                output = torch.concat((depth_pred, aleatoric_var, depth_to_disp(depth_pred)),axis=1) # depth, aleatoric var, disp
                # change shape of features (used for cosine similarity for buffer addition/removal policy)
                features = torch.concat((features[0][0], features[1][0],features[2][0],features[3][0]),axis=1)
            torch.cuda.synchronize()
            end_inf = time.time() 
            output = output.to(run_parameters['device'][0]) # return result to main device
            features = features.to(run_parameters['device'][0]) # return result to main device
            if n > 0:
                assert (torch.allclose(output, all_outputs[-1]) == False), "Current inference output same as previous inference output"
            all_outputs.append(output)
            all_features.append(features)
            pred = torch.reshape(output[:,0,:,:],(output.shape[0],1,output.shape[2],output.shape[3])) 
            if n == 0: # initialize old pred
                old_depth_mean = torch.zeros(pred.shape, device=torch.device(run_parameters['device'][0]))
                epistemic_unc = torch.zeros(output[:,0,:,:].shape, device=torch.device(run_parameters['device'][0]))
            aleatoric_unc = parse_aleatoric_variance(run_parameters,output)  
            # calculate average depth prediction so far 
            depth_sum+=pred 
            depth_mean = depth_sum/(n+1)
            # calculate sum of aleatoric variance prediction so far
            aleatoric_var_sum+=aleatoric_unc
            aleatoric_var_mean = aleatoric_var_sum/(n+1)
            # calculate iterative epistemic variance so far
            epistemic_unc = compute_iterative_epistemic_uncertainty(pred, epistemic_unc, depth_mean, old_depth_mean, n+1, run_parameters, biased=True)  
            old_depth_mean = depth_mean
            # update sample index
            n+=1 
    assert (torch.count_nonzero(aleatoric_var_mean < 0) == 0), "Aleatoric variance has negative values: " + str(torch.count_nonzero(aleatoric_var_mean < 0))
    assert (torch.count_nonzero(epistemic_unc < 0) == 0), "Epistemic variance has negative values: " + str(torch.count_nonzero(epistemic_unc < 0))
    torch.cuda.synchronize()
    inference_time = time.time() - inference_start
    return output, depth_mean, aleatoric_var_mean, epistemic_unc, inference_time, all_outputs, features, all_features

def compute_iterative_epistemic_uncertainty(pred, epistemic_unc, new_mean, old_mean, k, run_parameters, biased = True):
    if biased: 
        Sn = epistemic_unc*(k-1) # previous Sn 
        Sn = Sn + (pred - old_mean)*(pred - new_mean)
        epistemic_unc = Sn/(k) # variance
    else:
        Sn = epistemic_unc*(k-2) # previous Sn 
        Sn = Sn + (pred - old_mean)*(pred - new_mean)
        epistemic_unc = Sn/(k-1) # variance
        if k == 1:# only one number, return 0 for sample variance instead of NaN
            epistemic_unc = torch.zeros(pred.shape, device=pred.get_device())  
    return epistemic_unc

'''
Description: record accuracy and loss statistics to file 
Input:       
    run_parameters:         run parameters loaded in from yaml 
    result:                 Result object with results
    loss:                   loss after training on buffer for number of buffer epochs
    average_meter:          AverageMeter object with average results
    gpu_time                time on GPU for training or inference
    data_time:              time for loading in data 
    input:                  RGB input tensor
    i:                      sequence index
    len_train_loader:       number of images in current trainin dataset 
    dataset_idx:            dataset index
    epoch:                  epoch number (not buffer epoch)
    csv_name:               name of csv file writing stats out to 
    per_image_csv_name:        csv file for stats per image
    fieldnames:             column names in csv files
    num_weight_updates:     number of weight updats to DNN so far
    num_decisions_train:    number of times acquisition function returns decision to train
    
Output:      
    average_meter:          AverageMeter object with updated average results
'''
def record_acc_loss(run_parameters, result, average_meter, gpu_time, data_time, input, i, len_train_loader, dataset_idx, epoch, csv_name, per_image_csv_name,  fieldnames, num_weight_updates, num_decisions_train, metric_gain):
    # measure accuracy and record loss, weighted by batch size 
    average_meter.update(result, gpu_time, data_time, input.size(0))
    if (i % run_parameters['print_freq'] == 0): 
        print_training_stats(dataset_idx, epoch, i, len_train_loader, result, average_meter)
    # store training metrics 
    avg = average_meter.average()
    write_avg_metrics_to_file(csv_name, avg, fieldnames, dataset_idx, i, num_weight_updates, num_decisions_train, metric_gain, epoch)
    write_avg_metrics_to_file(per_image_csv_name, result, fieldnames, dataset_idx, i, num_weight_updates, num_decisions_train, metric_gain, epoch)
    return average_meter

'''
Description: loads input and target on to CUDA device
Input:       
    input:                  RGB input tensor
    target:                 target tensor    
Output:      
    input:                  RGB input tensor
    target:                 target tensor    
    data_time:              time for loading in data 
'''
def load_input_target(metadata, device):
    data_start = time.time()
    metadata = {k: v.to(device) for k, v in metadata.items()}
    torch.cuda.synchronize()
    data_time = time.time() - data_start
    return metadata, data_time

'''
Description: initiates variables needed at start of training
Input:       
    models:                 list of loaded models
Output:      
    average_meter:          AverageMeter object initialized
    models:                 list of loaded models
    epoch_start_time:       start time of training
    num_weight_updates:     number of weight updates to DNN initialized to 0
    num_decisions_train:    number of times acquisition function returns decision to train
'''
def initiate_train_epoch(models):
    average_meter = nets.metrics.AverageMeter()
    average_meter_all_dnns = [] 
    for m, model in enumerate(models):
        set_model_mode(model, mode='eval')
        average_meter_all_dnns.append(nets.metrics.AverageMeter())
    epoch_start_time = time.time()
    num_weight_updates = 0 # counter for number of weight updates
    num_decisions_train = 0 # counter for number of decisions to train
    return average_meter, average_meter_all_dnns, models, epoch_start_time, num_weight_updates, num_decisions_train

'''
Description: turn off MC-dropout in copy of configuration file
Input:       
    run_parameters:         run parameters loaded in from yaml 
Output:      
    run_parameters_copy:    run parameters modified with p = 0 and number of inferences = 1 
'''
def get_no_dropout_one_inference_run_parameters(run_parameters):
    run_parameters_copy = run_parameters.copy()
    keys_to_change=['p', 'num_inferences']
    vals_to_change=[0,1]
    for key, val in zip(keys_to_change, vals_to_change):
        run_parameters_copy[key] = val
    return run_parameters_copy

'''
Description: run forward inference on image and parse results without gradients to compute epistemic variance (without exceeding CUDA memory)
Input:       
    run_parameters:         run parameters loaded in from yaml
    target:                 target tensor 
Output:      
    target:                 target tensor 
''' 
def get_sampled_bnn_epistemic_var(run_parameters, models, input):
    # run forward inference on image and parse results without gradients
    with torch.no_grad():
        output_bnn, pred_bnn, aleatoric_unc_bnn, epistemic_unc_bnn, inference_time_bnn, all_outputs_bnn, _, _ = forward_inference(run_parameters, models, input, run_parameters['device'])
    return epistemic_unc_bnn
    

'''
Description: queries current lr 
Input:       
    optimizer:              optimizer
Output:      
    param_group['lr']:      learning rate
'''
def get_current_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

from src.policyDataExtractor import policyDataExtractor
def initialize_policy_data_extractor(run_parameters, train_dataset_idx):
    stats_collection_on = run_parameters['record_policy_training_data']
    update_pattern = run_parameters['record_policy_training_data_pattern']
    policy_data_extractor = policyDataExtractor(
        train_dir=run_parameters['train_dir'][train_dataset_idx], 
        update_pattern=update_pattern, 
        stats_collection_on=stats_collection_on,
        record_per_img_all_outputs=False,
        record_all_trained_combined_stats=False,
        record_all_trained_per_img_stats=True,
        all_trained_sample_count=run_parameters['buffer_window_size']+1
    )
    return policy_data_extractor

"""
Description: setup ablation study for on-the-fly training
"""
def ablation_study_setup(run_parameters, policy_feat):
    if policy_feat is None:
        return None
    if "ablation_target" not in run_parameters:
        return policy_feat
    if run_parameters["ablation_target"] == None:
        return policy_feat
    if "epistemic_unc" in run_parameters["ablation_target"]:
        policy_feat["epistemic_avg"] = 0
        policy_feat["epistemic_var"] = 0
        policy_feat["epistemic_mid"] = 0
        policy_feat["epistemic_max"] = 0
    if "aleatoric_unc" in run_parameters["ablation_target"]:
        for key in policy_feat.keys():
            if "aleatoric" in key:
                policy_feat[key] = 0
    if "depth" in run_parameters["ablation_target"]:
        for key in policy_feat.keys():
            if "depth" in key:
                policy_feat[key] = 0
    if "loss" in run_parameters["ablation_target"]:
        policy_feat["prev_loss_curr"] = 0
    if "pose" in run_parameters["ablation_target"]:
        for key in policy_feat.keys():
            if "translation" in key or "rotation" in key:
                policy_feat[key] = 0
    if "landmark" in run_parameters["ablation_target"]:
        for key in policy_feat.keys():
            if "landmark" in key:
                policy_feat[key] = 0
    return policy_feat


"""
Description: get training buffer (current img + replay buffer random selection) and_policy input features
"""
def get_training_buffer_and_policy_input(run_parameters, idx, model, loss_fn, online_buffer, device, intrinsics_seq, policy_data_extractor, epistemic_unc_curr, prev_loss_curr, all_outputs_curr):
    set_model_mode(model, mode='eval')
    policy_feat, current_online_stats = None, None
    train_batch_data = {} # {step: [ (shuffled_keys_batch, training_data, sample_buffer) ]}
    with torch.no_grad():
        for step in range(run_parameters['buffer_num_epochs']):
            train_batch_data[step] = []
            for shuffled_keys_batch, training_data, sample_buffer in online_buffer.gen_random_metadata(run_parameters['batch_size']):
                train_batch_data[step].append(tuple([shuffled_keys_batch, training_data, sample_buffer]))
                training_data, data_time = load_input_target(training_data, device=device)

                # decide whether to train on this batch (need to have replay buffer stats)
                if not policy_data_extractor.stats_collection_on and run_parameters['acquisition_type'] == "balance_compute_accuracy_gain_policy":
                    assert run_parameters['buffer_num_epochs'] == 1, "Current policy is 1-step lookahead, so buffer_num_epochs should be 1, as the model is not trained to predict the gain after multiple DNN updates"
                    # epistemic only related to current frame, and only using current frame stats + landmark stats from replay buffer
                    assert len(all_outputs_curr) == 1, "Should only have one depth prediction"
                    policy_feat, current_online_stats = policy_data_extractor.extract_policy_input(seq_idx=idx, queries=shuffled_keys_batch, all_outputs=[torch.cat(tuple([all_outputs_curr[0]]*3), 0)], epistemic_unc=epistemic_unc_curr, sample_buffer=sample_buffer, prev_loss=prev_loss_curr)
                # log for policy training
                if policy_data_extractor.stats_collection_on:
                    # run forward inference on image and parse results
                    output, pred, aleatoric_unc, epistemic_unc, inference_time, all_outputs, _, _ = forward_inference(run_parameters, [model], training_data['rgb'], [device])
                    # loss computation
                    criteria_data = collect_criteria_data(
                        run_parameters=run_parameters,
                        target=training_data['depth'],
                        pred=output,
                        buffer=sample_buffer,
                        intrinsics=intrinsics_seq,
                        batch_queries=shuffled_keys_batch,
                        device=run_parameters['device'][0]
                    )
                    loss = loss_fn(criteria_data)
                    policy_data_extractor.record_online_stats_all_trained(seq_idx=idx, queries_dict=shuffled_keys_batch, all_outputs=all_outputs, epistemic_unc=epistemic_unc, sample_buffer=sample_buffer)
    # ablation study setup
    policy_feat = ablation_study_setup(run_parameters, policy_feat)
    return train_batch_data, policy_feat, current_online_stats

"""
Description: get groundtruth metric gain between current model and updated model
"""
from dataclasses import dataclass
@dataclass
class NULL_POLICY_DATA_EXTRACTOR():
    stats_collection_on: bool = False

def get_groundtruth_metric_gain(
        # flow control
        acq_decision, 
        # train on buffer
        run_parameters, run_parameters_no_dropout_one_inf, models, i, seq_idx_bias, loss_fn, optimizers, train_batch_data, intrinsics_seq, 
        # forward inference
        metadata,
        # compute gain
        pred_prev, losses_prev, losses_post,
        # zero-regret policy
        acq
        ):
    null_policy_data_extractor = NULL_POLICY_DATA_EXTRACTOR(stats_collection_on=True)
    # if not trained, train once; else, skip training
    snapshot_state = {}
    if not acq_decision:
        # snapshot model
        if not isinstance(models[0], tuple):
            snapshot_state['models'] = [ deepcopy(model.state_dict()) for model in models ]
        else:
            snapshot_state['models'] = [ deepcopy(model[0].state_dict()) for model in models ]
        # train once
        all_losses_prev, all_losses_post = [], []
        for m, (model, device) in enumerate(zip(models, run_parameters['device'])): 
            loss_per_model, model_train_time, num_weight_updates_per_model, trained_img_idxs, loss_post_per_model = train_on_buffer(run_parameters_no_dropout_one_inf, i+seq_idx_bias, models[m], loss_fn, optimizers[m], train_batch_data, device, intrinsics_seq, null_policy_data_extractor)
            all_losses_prev.append(loss_per_model)
            all_losses_post.append(loss_post_per_model)
        loss_prev = sum(all_losses_prev) / len(all_losses_prev)
        loss_update = sum(all_losses_post) / len(all_losses_post)
    else:
        # if trained, skip training, and use passed-in losses
        loss_prev = sum(losses_prev) / len(losses_prev)
        loss_update = sum(losses_post) / len(losses_post)
    # evaluate the updated model on current frame
    # select whether or not to force no dropout and one inference (relevant for MCDropout)
    run_parameters_main_inf = check_which_run_parameters_to_use(run_parameters, run_parameters_no_dropout_one_inf)
    # select which models/devices to run main inference (non-trivial if Ensemble-UfM, otherwise returns back list of models)
    if run_parameters['unc_method'] == 'ensemble-ufm':
        ensemble_member_number = i % run_parameters['ensemble_size']
        models_main_inf = [models[ensemble_member_number]] 
        devices_main_inf = [run_parameters['device'][ensemble_member_number]]
    else:
        models_main_inf = models 
        devices_main_inf = run_parameters['device']
    with torch.no_grad():
        output_update, pred_update, aleatoric_unc_update, epistemic_unc_update, inference_time_update, all_outputs_update, features_update, all_features_update = forward_inference(run_parameters_main_inf, models_main_inf, metadata['rgb'], devices_main_inf)
    # compute the gain (acc, error, loss)
    result_prev = nets.metrics.Result()
    result_prev.evaluate(pred_prev, metadata['depth'], loss_prev, None, None)
    result_updated = nets.metrics.Result()
    result_updated.evaluate(pred_update, metadata['depth'], loss_update, None, None)
    metric_gain = {
        'rmse'      : result_updated.rmse - result_prev.rmse,
        'absrel'    : result_updated.absrel - result_prev.absrel,
        'delta1'    : result_updated.delta1 - result_prev.delta1,
        'delta1_lit': result_updated.delta1_lit - result_prev.delta1_lit,
        'loss'      : loss_update - loss_prev,
        'loss_rate' : (loss_update - loss_prev) / loss_prev if loss_prev != 0 else 0,
    }
    
    # model recover stage
    post_decision = None
    if run_parameters['acquisition_type'] in ["balance_compute_accuracy_gain_zero_regret"]:
        # post decision with zero regret policy
        post_decision = acq.decide_to_train(seq_idx=i+seq_idx_bias, this_seq_idx=i, metric_gain=metric_gain)
        if not post_decision:
            # recover model
            for i in range(len(models)):
                get_depth_autoencoder_at_idx(models, i).load_state_dict(snapshot_state['models'][i]) 
    else:
        # no post-decision included, recover model to original state if originally not trained
        if not acq_decision:
            # recover model
            print("[Groundtruth Metric Gain] Recovering model to original state...")
            for i in range(len(models)):
                get_depth_autoencoder_at_idx(models, i).load_state_dict(snapshot_state['models'][i])
    return metric_gain, post_decision

def online_policy_adapt_reinference(
    # flow control
    ADAPT_DECISION,
    # updated policy
    acq,
    # parameters required for reinference method
    run_parameters,
    run_parameters_no_dropout_one_inf,
    models,
    loss_fn,
    metadata,
    online_buffer,
    intrinsics_seq,
    # parameters required for policy update
    curr_loss_prev,
    seq_idx,
    policy_feat
):
    loss_diff_rate = None
    curr_loss_post = None
    if ADAPT_DECISION:
        # 1 additional depth inference required for getting current loss gain rate
        with torch.no_grad():
            output_post, pred_post, _, _, _, all_outputs_post, _, _ = forward_inference(run_parameters_no_dropout_one_inf, models, metadata['rgb'], run_parameters['device'])
            # for recording current image loss after model traininig
            criteria_data = collect_criteria_data(
                    run_parameters=run_parameters,
                    target=metadata['depth'],
                    pred=output_post,
                    buffer=online_buffer.running_buffer,
                    intrinsics=intrinsics_seq,
                    batch_queries={seq_idx: list(online_buffer.running_buffer.keys())},
                    device=run_parameters['device'][0]
                )
            curr_loss_post = loss_fn(criteria_data).item()
        # loss_diff_rate = (all_losses_post[0]-all_losses_prev[0])/all_losses_prev[0]
        # print("\n***" + str(seq_idx) + ": Reinference: current_loss_post: " + str(curr_loss_post))
        loss_diff_rate = (curr_loss_post - curr_loss_prev)/curr_loss_prev # using current loss only
        acq.update_online_policy(train_decision=True, seq_idx=seq_idx, x_stats=policy_feat, y_stats=loss_diff_rate, ablation_targets=run_parameters["ablation_target"])
    return acq, curr_loss_post, loss_diff_rate

def online_policy_adapt_warped(
    # flow control
    ADAPT_DECISION,
    # updated policy
    acq,
    # parameters required for warped method
    run_parameters,
    loss_fn,
    online_buffer,
    intrinsics_seq,
    curr_output,
    prev_running_buffer,
    # parameters required for policy update
    seq_idx,
    curr_loss_prev_d1,
    seq_idx_d1,
    policy_feat_d1,
):
    loss_diff_rate = None
    mimic_loss_post = None
    if ADAPT_DECISION:
        print(f"[Policy Update] Update policy based on warped loss rate")
        # get loss: L(D_t, X_t, Buf_{t-1}| M_v)
        mimic_running_buffer = {k: v for k, v in prev_running_buffer.items() if k != seq_idx_d1}
        mimic_running_buffer[seq_idx] = online_buffer.running_buffer[seq_idx]  # only keep {curr_idx, prev_r1, prev_r2}
        assert len(mimic_running_buffer) == 3, "Mimic buffer should have 3 elements only: [curr_idx, prev_replay_idx1, prev_replay_idx2]"
        # print(mimic_running_buffer.keys())
        criteria_data = collect_criteria_data(
            run_parameters=run_parameters,
            target=None, # not used
            pred=curr_output,
            buffer=mimic_running_buffer,
            intrinsics=intrinsics_seq,
            batch_queries={seq_idx: list(mimic_running_buffer.keys())}, # calculate loss using D_(t) -> {RGB_(r1@t-1), RGB_(r2@t-1)}
            device=run_parameters['device'][0]
        )
        with torch.no_grad():
            mimic_loss_post = loss_fn(criteria_data).item()

        # get loss_diff_rate by L(D_t, X_t, Buf_{t-1}| M_v) - L(D_(t-1), X_(t-1), Buf_(t-1)| M_(v-1))
        loss_diff_rate = (mimic_loss_post - curr_loss_prev_d1)/curr_loss_prev_d1 # using current loss only
        # update policy based on info from previous frame
        acq.update_online_policy(train_decision=True, seq_idx=seq_idx_d1, x_stats=policy_feat_d1, y_stats=loss_diff_rate, ablation_targets=run_parameters["ablation_target"])
    return acq, mimic_loss_post, loss_diff_rate


def online_policy_adapt_next_inference(acq, loss_previous_dnn_previous_img, loss_current_dnn_current_img, seq_idx_d1, policy_feat_d1):
    loss_diff_rate = (loss_current_dnn_current_img - loss_previous_dnn_previous_img)/loss_previous_dnn_previous_img 
    # update policy based l(X_{t-1:t+1}, D(X_{t+1}, theta_{k+1})) - l(X_{t-2:t}, D(X_{t}, theta_{k}))
    acq.update_online_policy(train_decision=True, seq_idx=seq_idx_d1, x_stats=policy_feat_d1, y_stats=loss_diff_rate, ablation_targets=run_parameters["ablation_target"])
    return acq

'''
Description: run online training 
Input:       
    run_parameters:         run parameters loaded in from yaml
    run_parameters_no_dropout_one_inf:    run parameters modified with p = 0 and number of inferences = 1 
    train_loader:           dataset loader for current training dataset 
    models:                 list of loaded models
    loss_fn:                loss function
    optimizers:             list of optimizers
    epoch:                  epoch number (not buffer epoch)
    train_csv:              csv file for average training stats 
    train_per_image_csv:       csv file fore training stats per image
    fieldnames:             column names in csv files 
    acq:                    Acquisition object for active learning selection
    train_dataset_idx:      index of training dataset in dataloaders list
    output_dir:             output directory from command line arguments 
    train_dataset_length:   number of images in current training dataset 
Output:      
    num_weight_updates:     number of weight updates to DNN so far (increases with ensemble size)
    num_decisions_train:    number of decisions to train 
'''

def train_online(run_parameters, run_parameters_no_dropout_one_inf, train_loader, models, loss_fn, optimizers, 
                 epoch, train_csv, train_per_image_csv, train_all_dnns_csv, train_per_image_all_dnns_csv, fieldnames, acq, 
                 train_dataset_idx, output_dir, seq_length, seq_idx_bias, intrinsics_seq, online_buffer, average_meter, 
                 average_meter_all_dnns, epoch_start_time, num_weight_updates, num_decisions_train, 
                 val_target_loaders, val_target_csv):
    # initialize policy data extractor
    policy_data_extractor = initialize_policy_data_extractor(run_parameters, train_dataset_idx)
    # reset running buffer (for stitched sequences)
    online_buffer.reset_running_buffer()
    # for delayed policy adaptation
    ACQ_DECISION_D1, PU_running_buffer_d1, PU_curr_loss_prev_d1, PU_seq_idx_d1, PU_policy_feat_d1 = False, None, None, None, None
    gt_loss_rate_gain, gt_loss_post = 0, 0 # debug
    # make empty UfM point cloud if using UfM
    if run_parameters['unc_method'] == 'mc-ufm' or run_parameters['unc_method'] == 'ensemble-ufm':
        # print('Using UfM')
        ufm_run_parameters = make_ufm_config_file(run_parameters)
        ufm_device = 'cuda:0'
        UfM_pc = UncertaintyFromMotion(ufm_run_parameters, torch.device(ufm_device)) # initialize point cloud
    # post analysis
    if run_parameters['save_snapshot']:
        save_snapshot(seq_idx=-1, buffer=online_buffer, models=models, output_dir=output_dir)

    # simulate seeing sequence in order 
    for i, metadata in enumerate(train_loader):
        # load in data 
        metadata, data_time = load_input_target(metadata, run_parameters['device'][0])
        # squeeze pose tensors 
        metadata['rot'] = torch.squeeze(metadata['rot']) # current rotation  # 3x3
        metadata['trans'] = torch.squeeze(torch.transpose(metadata['trans'], 0, 1)) # current translation # 3x1
        # check if valid pose 
        invalid_pose = torch.any(torch.isinf(metadata['rot'])) or torch.any(torch.isinf(metadata['trans']))
        if invalid_pose:
            print("\nRecognized invalid pose on this_seq_idx " + str(i) + " and stitched seq_idx " + str(i + seq_idx_bias) + ". Will run inference and log result, but will not run training, add to buffer, or add to data collection for linear regression/oracle.")
        # select whether or not to force no dropout and one inference (relevant for MCDropout)
        run_parameters_main_inf = check_which_run_parameters_to_use(run_parameters, run_parameters_no_dropout_one_inf)
        # select which models/devices to run main inference (non-trivial if Ensemble-UfM, otherwise returns back list of models)
        if run_parameters['unc_method'] == 'ensemble-ufm':
            ensemble_member_number = i % run_parameters['ensemble_size']
            models_main_inf = [models[ensemble_member_number]] # 
            devices_main_inf = [run_parameters['device'][ensemble_member_number]]
        else:
            models_main_inf = models 
            devices_main_inf = run_parameters['device']
        
        # DNN inference
        with torch.no_grad():
            # Profiling: encoder-inference, depth-inference
            output, pred, aleatoric_unc, epistemic_unc, inference_time, all_outputs, features, all_features = forward_inference(run_parameters_main_inf, models_main_inf, metadata['rgb'], devices_main_inf)
            # feature vector use for CoDEPs method
            mean_features = sum(all_features) / len(all_features)
        if not invalid_pose: # valid pose 
            # Get epistemic uncertainty
            # if doing sampled BNN (e.g, MC-Dropout) for epistemic variance, run with no grad for dropout value p and multiple forward inference (otherwise limited by CUDA memory) 
            if run_parameters['unc_method'] == 'mc' or run_parameters['unc_method'] == 'mc-last': 
                epistemic_unc = get_sampled_bnn_epistemic_var(run_parameters, models, metadata['rgb']) # computes dropout passes with no gradients (otherwise, storing an ensemble of gradients)
            elif run_parameters['unc_method'] == 'mc-ufm' or run_parameters['unc_method'] == 'ensemble-ufm':
                # compute UfM 
                if run_parameters['unc_method'] == 'mc-ufm':
                    # run an additional inference with dropout 
                    output_dropout, pred_dropout, aleatoric_unc_dropout, epistemic_unc_dropout, inference_time_dropout, all_outputs_dropout, features_dropout, all_features_dropout = forward_inference(run_parameters, models_main_inf, metadata['rgb'], devices_main_inf)
                    pred_detached = pred_dropout.detach().to(ufm_device)
                else:
                    pred_detached = pred.detach().to(ufm_device)
                metadata['trans'] = metadata['trans'].to(ufm_device)
                metadata['rot'] = metadata['rot'].to(ufm_device)
                if run_parameters['ufm_input_unc'] == "none":
                    epistemic_unc = run_UfM_on_frame(pred_detached, torch.zeros(aleatoric_unc.shape, device=torch.device(ufm_device)), metadata['trans'], metadata['rot'], UfM_pc, metadata['real_idx'])
                elif run_parameters['ufm_input_unc'] == "aleatoric":
                    aleatoric_unc_detached = aleatoric_unc.detach().to(ufm_device)
                    assert torch.all(aleatoric_unc_detached >= 0), "Negative elements in aleatoric var"
                    epistemic_unc = run_UfM_on_frame(pred_detached, aleatoric_unc_detached, metadata['trans'], metadata['rot'], UfM_pc, metadata['real_idx'])
                epistemic_unc = epistemic_unc.to(run_parameters['device'][0])
                metadata['trans'] = metadata['trans'].to(run_parameters['device'][0])
                metadata['rot'] = metadata['rot'].to(run_parameters['device'][0])
            # calculate loss for logging into result (not currently used for deciding whether to train or not)
            online_buffer.push_metadata(i+seq_idx_bias, metadata, intrinsics_seq[-1])
            criteria_data = collect_criteria_data(
                    run_parameters=run_parameters,
                    target=metadata['depth'],
                    pred=output,
                    buffer=online_buffer.running_buffer,
                    intrinsics=intrinsics_seq,
                    batch_queries={i+seq_idx_bias: list(online_buffer.running_buffer.keys())},
                    device=run_parameters['device'][0]
                )
            loss = loss_fn(criteria_data).item()
            print("\n" + str(i+seq_idx_bias) + ": train online: loss: " + str(loss))
            lr_logger_curr_loss_prev = loss # for policy_data_extractor online stats
            models_train_time = 0 # initialize online training time
            # [Policy Data Extractor] if collect policy training dataset, record snapshot and record online stats
            policy_data_extractor.record_snapshot(pred=pred, metadata=metadata, models=models, buffer=online_buffer, acq_meta=acq, all_outputs=all_outputs)
            policy_data_extractor.record_online_stats(seq_idx=i+seq_idx_bias, all_outputs=all_outputs, epistemic_unc=epistemic_unc, running_buffer=online_buffer.running_buffer)
            
            # Making policy depend on the replay buffer images, so need to sample the image before going into the decision
            # decision made purely on the first model (ensemble member 0) inference 
            train_batch_data, policy_feat, current_online_stats = get_training_buffer_and_policy_input(run_parameters_no_dropout_one_inf, i+seq_idx_bias, models[0], loss_fn, online_buffer, run_parameters['device'][0], intrinsics_seq, policy_data_extractor, epistemic_unc, lr_logger_curr_loss_prev, all_outputs)
            # if acquisition function returns true, train all models on image 
            acq_cfgs = (i+seq_idx_bias, aleatoric_unc, epistemic_unc, pred, metadata['depth'], metadata['rgb'], online_buffer.running_buffer, loss, metadata['rot'], metadata['trans'], intrinsics_seq, metadata['real_idx'], mean_features, train_dataset_idx, policy_feat)
            
            # make decision
            ACQ_DECISION = acq.decide_to_train(*acq_cfgs)

            # Subprocess 1: Run training if acquisition function returns true
            all_losses_prev, all_losses_post = [], []
            if ACQ_DECISION:
                models_train_time = 0
                num_decisions_train+=1
                # add to buffer 
                online_buffer.push_sample(query=i+seq_idx_bias) # legacy code, not used in CoDEPS buffer
                for m, (model, device) in enumerate(zip(models, run_parameters['device'])): 
                    # train on buffer 
                    loss_per_model, model_train_time, num_weight_updates_per_model, trained_img_idxs, loss_post_per_model = train_on_buffer(run_parameters_no_dropout_one_inf, i+seq_idx_bias, models[m], loss_fn, optimizers[m], train_batch_data, device, intrinsics_seq, policy_data_extractor)
                    models_train_time += model_train_time # compute cumulative training time across all model(s)
                    num_weight_updates+=num_weight_updates_per_model # add to number of weight updates counter 
                    all_losses_prev.append(loss_per_model)
                    all_losses_post.append(loss_post_per_model)
                # post analysis1
                if run_parameters['save_snapshot']:
                    save_snapshot(seq_idx=i+seq_idx_bias, buffer=online_buffer, models=models, output_dir=output_dir)
                
                # [Policy Data Extractor] update DNN based on fixed pattern, record training stats
                if policy_data_extractor.stats_collection_on:
                    with torch.no_grad():
                        output_post, pred_post, _, _, _, all_outputs_post, _, _ = forward_inference(run_parameters_no_dropout_one_inf, models, metadata['rgb'], run_parameters['device'])
                        # for recording current image loss after model traininig
                        criteria_data = collect_criteria_data(
                                run_parameters=run_parameters,
                                target=metadata['depth'],
                                pred=output_post,
                                buffer=online_buffer.running_buffer,
                                intrinsics=intrinsics_seq,
                                batch_queries={i+seq_idx_bias: list(online_buffer.running_buffer.keys())},
                                device=run_parameters['device'][0]
                            )
                        lr_logger_curr_loss_post = loss_fn(criteria_data).item()
                    policy_data_extractor.update_decision_and_record_status(seq_idx=i+seq_idx_bias, pred_post=pred_post, metadata=metadata, models=models, buffer=online_buffer, acq_meta=acq, trained_img_idxs=trained_img_idxs, all_outputs_post=all_outputs_post, all_losses_prev=all_losses_prev, all_losses_post=all_losses_post, curr_loss_prev=lr_logger_curr_loss_prev, curr_loss_post=lr_logger_curr_loss_post)

                # [Policy Adapt] Post-training policy update: for direct policy update method
                if run_parameters['acquisition_type'] in ["balance_compute_accuracy_gain_policy"]:
                    acq, gt_loss_post, gt_loss_rate_gain = online_policy_adapt_reinference(
                        ACQ_DECISION, acq,
                        # parameters required for reinference method
                        run_parameters, run_parameters_no_dropout_one_inf, models, loss_fn, metadata, online_buffer, intrinsics_seq,
                        # parameters required for policy update
                        lr_logger_curr_loss_prev, i+seq_idx_bias, policy_feat
                    )

            # Subprocess 2: Get groundtruth metric gain
            if run_parameters['record_groundtruth_metric_gain']:
                metric_gain, post_decision = get_groundtruth_metric_gain(
                    ACQ_DECISION, 
                    run_parameters, run_parameters_no_dropout_one_inf, models, i, seq_idx_bias, loss_fn, optimizers, train_batch_data, intrinsics_seq, 
                    metadata,
                    pred, all_losses_prev, all_losses_post,
                    acq
                )
                if post_decision is not None:
                    if post_decision:
                        num_decisions_train+=1
            else:
                metric_gain = {'rmse': 0, 'absrel': 0, 'delta1': 0, 'delta1_lit': 0, 'loss': 0, 'loss_rate': 0} # placeholder (=0) for all metric gain if we are not recording groundtruth metric gain
            
            if len(online_buffer.running_buffer) > run_parameters['buffer_window_size']:
                # buffer update, push images that can conduct self-supervised learning
                pushed_to_buffer, pop_index = online_buffer.update_buffer(query=i+seq_idx_bias, features=mean_features, online_stats=current_online_stats)
                
        else: # invalid pose
            models_train_time = 0 # set training time to 0 when pose is invalid
            loss = float("nan") # set NaN for loss 
            metric_gain = {'rmse': float("nan"), 'absrel': float("nan"), 'delta1': float("nan"), 'delta1_lit': float("nan"), 'loss': float("nan"), 'loss_rate': float("nan")} # set NaN for all metric gain if the pose is invalid
        
        # compute result and record performance metrics 
        result = nets.metrics.Result()
        result.evaluate(pred, metadata['depth'], loss, aleatoric_unc, epistemic_unc)
        average_meter = record_acc_loss(run_parameters, result, average_meter, models_train_time, data_time, metadata['rgb'], i+seq_idx_bias, len(train_loader), train_dataset_idx, epoch, train_csv, train_per_image_csv, fieldnames, num_weight_updates, num_decisions_train, metric_gain)
        # compute result of individual dnn and record performance metrics 
        for d in range(len(all_outputs)):
            result_per_dnn = nets.metrics.Result()
            aleatoric_unc_per_dnn = parse_aleatoric_variance(run_parameters,all_outputs[d]) 
            epistemic_unc_per_dnn = torch.zeros(all_outputs[d][:,0,:,:].shape, device=torch.device(run_parameters['device'][0]))
            result_per_dnn.evaluate(torch.squeeze(all_outputs[d][:,0,:,:]), metadata['depth'], loss, aleatoric_unc_per_dnn, epistemic_unc_per_dnn)
            average_meter_all_dnns[d] = record_acc_loss(run_parameters, result_per_dnn, average_meter_all_dnns[d], models_train_time, data_time, metadata['rgb'], i+seq_idx_bias, len(train_loader), train_dataset_idx, epoch, train_all_dnns_csv[d], train_per_image_all_dnns_csv[d], fieldnames, num_weight_updates, num_decisions_train, metric_gain)

        # save files
        if run_parameters['save_files']:
            save_output_files(run_parameters, i+seq_idx_bias, metadata['rgb'], metadata['depth'], pred, aleatoric_unc, epistemic_unc, output_dir, all_outputs)

        # [Policy Adapt] record prev step for delayed policy adaptation
        ACQ_DECISION_D1, PU_running_buffer_d1, PU_curr_loss_prev_d1, PU_seq_idx_d1, PU_policy_feat_d1 = ACQ_DECISION, deepcopy(online_buffer.running_buffer), lr_logger_curr_loss_prev, i+seq_idx_bias, deepcopy(policy_feat)
    # validation 
    # validate if last trajectory of entire stitched sequence 
    if train_dataset_idx == (len(run_parameters['train_env'])-1):
        # validate on that environment's validation sequence  
        current_env = run_parameters['train_env'][train_dataset_idx]
        # get index of that env in val
        val_this_env_idx = run_parameters['val_env'].index(current_env)
        # validate on that target sequence for this environment + source validation  
        print("\n**********************" + "Validating on sequence: " + run_parameters['val_target_dir'][val_this_env_idx] + "***************************")
        run_validation(run_parameters,run_parameters_no_dropout_one_inf,val_target_loaders[val_this_env_idx], models, loss_fn, True, val_target_csv,fieldnames, output_dir, val_this_env_idx, i+seq_idx_bias, num_weight_updates, num_decisions_train, epoch) # evaluate on target validation set
    else: 
        # validate if last trajectory in this environment 
        # check if end of environment
        current_env = run_parameters['train_env'][train_dataset_idx]
        next_env = run_parameters['train_env'][train_dataset_idx+1]
        if current_env != next_env:
            # validate on that environment's validation sequence 
            # get index of that env in val
            val_this_env_idx = run_parameters['val_env'].index(current_env)
            print("\n**********************" + "Validating on sequence: " + run_parameters['val_target_dir'][val_this_env_idx] + "***************************")
            run_validation(run_parameters,run_parameters_no_dropout_one_inf,val_target_loaders[val_this_env_idx], models, loss_fn, True, val_target_csv,fieldnames, output_dir, val_this_env_idx, i+seq_idx_bias, num_weight_updates, num_decisions_train, epoch) # evaluate on target validation set
    # '''
    if run_parameters['save_models_epoch']:
        save_models(models, output_dir)

    # [Policy Data Extractor] Writeout training stats metadata
    policy_data_extractor.write_metadata(output_dir)

    print('Finished epoch {}: time {}s'.format(epoch, time.time() - epoch_start_time))
    return num_weight_updates, num_decisions_train

'''
Description: save post analysis files
Input:

'''
def save_snapshot(seq_idx, buffer, models, output_dir):
    output_snapshot_dir = output_dir+"/snapshots/"+str(seq_idx)
    os.makedirs(output_snapshot_dir, exist_ok=True)
    # save models
    save_models(models, output_snapshot_dir)
    # save buffer
    pickle.dump(buffer, open(output_snapshot_dir+"/buffer.pickle", "wb"))
    return

'''
Description: run validation
Input:       
    run_parameters:         run parameters
    val_loader:             data loader for validation
    models:                 list of loaded models
    loss_fn:                loss function
    write_to_file:          flag to indicate whether to write out validation stats to csv
    val_target_csv:         csv file for validation stats on target 
    fieldnames:             column names in csv files 
    output_dir:             output directory from command line arguments 
    val_dataset_idx:        index of validation dataset in dataloaders list
    train_seq_idx:          sequence index
    num_weight_updates:     number of weight updats to DNN so far
    num_decisions_train:    number of times acquisition function returns decision to train
'''
def validate(run_parameters, val_loader, models, write_to_file, test_csv, fieldnames, output_dir, val_dataset_idx, train_seq_idx, num_weight_updates, num_decisions_train, train_epoch): 
    average_meter_val = nets.metrics.AverageMeter()
    for m, model in enumerate(models):
        set_model_mode(model, mode='eval')
    times = [] # holder for DNN time
    for i, metadata in enumerate(val_loader):
        # load in data
        metadata, data_time = load_input_target(metadata, device=run_parameters['device'][0])
        # run forward inference on image and parse results
        with torch.no_grad():
            output, pred, aleatoric_unc, epistemic_unc, inference_time, all_outputs, _, _ = forward_inference(run_parameters, models, metadata['rgb'], run_parameters['device'])
        result = nets.metrics.Result()
        criteria_data = collect_criteria_data(run_parameters=run_parameters, target=metadata['depth'], pred=output, buffer='', intrinsics='', batch_queries='', device=run_parameters['device'][0])
        pred = pred.to(run_parameters['device'][0]) 
        result.evaluate(pred, metadata['depth'], 0, aleatoric_unc, epistemic_unc)
        average_meter_val.update(result, inference_time, data_time, metadata['rgb'].size(0))
        # print stats
        if (i+1) % run_parameters['print_freq'] == 0:
            print('Test: [{0}/{1}]\t'
                't_GPU={gpu_time:.3f}({average.gpu_time:.3f})\n\t'
                'RMSE={result.rmse:.2f}({average.rmse:.2f}) '
                'MAE={result.mae:.2f}({average.mae:.2f}) '
                'Delta1={result.delta1:.3f}({average.delta1:.3f}) '
                'Delta1_lit={result.delta1_lit:.3f}({average.delta1_lit:.3f}) '
                'REL={result.absrel:.3f}({average.absrel:.3f}) '
                'Lg10={result.lg10:.3f}({average.lg10:.3f}) '.format(
                i+1, len(val_loader), gpu_time=inference_time, result=result, average=average_meter_val.average()))
    avg = average_meter_val.average()

    print('\n*\n'
        'RMSE={average.rmse:.3f}\n'
        'MAE={average.mae:.3f}\n'
        'Delta1={average.delta1:.3f}\n'
        'Delta1_lit={average.delta1_lit:.3f}\n'
        'REL={average.absrel:.3f}\n'
        'Lg10={average.lg10:.3f}\n'
        't_GPU={time:.3f}\n'.format(
        average=avg, time=avg.gpu_time))

    if write_to_file:
        metric_gain = {'rmse': 0, 'absrel': 0, 'delta1': 0, 'delta1_lit': 0, 'loss': 0}
        write_avg_metrics_to_file(test_csv, avg, fieldnames, val_dataset_idx, train_seq_idx, num_weight_updates, num_decisions_train, metric_gain, epoch=train_epoch)
    torch.cuda.empty_cache()
    return

def check_valid_config(run_parameters):
    # check that if we're running dropout in main prediction (not MCD methods), it is with one inference
    if run_parameters['unc_method'] != 'mc' and run_parameters['unc_method'] != 'mc-last':
        assert run_parameters['num_inferences'] == 1, "Running multiple inferences with tracking gradients, will store ensemble of gradients"
    # check if we're running UfM, it is with 1 inference 
    if run_parameters['unc_method'] == 'mc-ufm' or run_parameters['unc_method'] == 'ensemble-ufm':
        assert run_parameters['num_inferences'] == 1, "Running UfM with multiple inferences; incorrect setting."
    # check if we're running main prediction without dropout and only one inference, we are running MCDropout without UfM; only needed for that case. 
    if run_parameters['unc_method'] == 'mc' or run_parameters['unc_method'] == 'mc-last':
        assert (run_parameters['p'] > 0 and run_parameters['num_inferences'] > 1), "Running main prediction without dropout and only one inference, but not running MCDropout methods"
    if run_parameters['num_inferences'] > 1:
        assert (run_parameters['ensemble_size'] == 1), "Trying to do multiple inferences on each ensemble member for ensemble_size > 1"
        assert (run_parameters['p'] > 0), "Trying to do multiple inferences with zero dropout"

def make_ufm_config_file(run_parameters):
    ufm_run_parameters = {}
    ufm_run_parameters['input_height'] = run_parameters['height']
    ufm_run_parameters['input_width'] = run_parameters['width']
    ufm_run_parameters['fx'] = run_parameters['fx'][0]
    ufm_run_parameters['fy'] = run_parameters['fy'][0]
    ufm_run_parameters['cx'] = run_parameters['cx'][0]
    ufm_run_parameters['cy'] = run_parameters['cy'][0]
    ufm_run_parameters['max_cloud_size'] = run_parameters['max_cloud_size']
    return ufm_run_parameters

'''
Description: initialize variables to start online training 
Input:       
    run_parameters:         run parameters
Ouput:
    loss_fn:                loss function
    train_loaders:          list of data loaders for training
    val_target_loaders:            list of data loaders for validation
    models:                 list of loaded models
    optimizers:             list of optimizers
    acq:                    Acquisition object for active learning selection
    fieldnames:             column names in csv files 
    output_dir:             output directory from command line arguments 
    train_csv:              csv file for average training stats 
    train_per_image_csv:       csv file fore training stats per image
    train_first_dnn_csv:    csv file for average training stats for first DNN in ensemble
    train_per_image_first_dnn_csv: csv file fore training stats per image for first DNN in ensemble
    val_target_csv:         csv file for validation stats on target 
    train_dataset_length:   number of images in current training dataset 
    run_parameters_no_dropout_one_inf:    run parameters modified with p = 0 and number of inferences = 1 
    run_parameters:         run_parameters updated to be backward compatible and computed budget 
    intrinsics_seqs:        list of intrinsics for different train_loader sequences 
'''
def initialize_training(run_parameters, output_dir):
    # define loss function 
    loss_fn = make_loss_function(run_parameters)
    # make data loaders
    train_loaders, val_target_loaders, train_dataset_length, intrinsics_seqs, run_parameters = create_data_loaders(run_parameters)
    # create architecture
    print("=> creating Model ({}) ...".format(run_parameters['arch']))
    models = make_architecture(run_parameters)
    print("=> model created.")
    # make optimizer 
    optimizers = make_optimizers(run_parameters, models, train_dataset_length)
    # check device list
    assert len(run_parameters['device']) == len(models)
    for model, device in zip(models, run_parameters['device']):
        if not isinstance(model, tuple):
            model = model.to(device)
        else: # tuple of depth autoencoder and separate aleatoric decoder
            model = (model[0].to(device), model[1].to(device))
    # make acquisition function 
    acq = acquisition.AcquisitionFunction(run_parameters, train_dataset_length, output_dir)
    # define some global data values and results files 
    fieldnames = ['epoch', 'dataset_idx', 'seq_idx', 'num_weight_updates', 'num_decisions_train', 'mse', 'rmse', 'absrel', 'lg10', 'mae', 'delta1', 'delta2', 'delta3', 'delta1_lit', 'delta2_lit', 'delta3_lit', 'data_time', 'gpu_time', 'loss', 'nll', 'aleatoric_nll', 'epistemic_nll', 'avg_aleatoric_unc', 'avg_epistemic_unc', 'avg_depth_unmasked', 'avg_depth_masked', 'avg_depth_gt_masked',
                'depth_unmasked_median','depth_masked_median','depth_gt_masked_median', 'aleatoric_unc_unmasked_median', 'epistemic_unc_unmasked_median','depth_unmasked_max', 'depth_masked_max', 'depth_gt_masked_max', 'aleatoric_unc_unmasked_max', 'epistemic_unc_unmasked_max', 'depth_unmasked_min', 'depth_masked_min', 'depth_gt_masked_min', 'aleatoric_unc_unmasked_min', 'epistemic_unc_unmasked_min',
                'depth_unmasked_var', 'depth_masked_var', 'depth_gt_masked_var', 'aleatoric_unc_unmasked_var', 'epistemic_unc_unmasked_var',
                'delta1_diff', 'delta1_lit_diff', 'rmse_diff', 'absrel_diff', 'loss_diff'
                ]
    train_csv, train_per_image_csv, train_all_dnns_csv, train_per_image_all_dnns_csv, val_target_csv = make_results_files(run_parameters, fieldnames, output_dir)
    # make copy of run parameters that has no dropout and only one inference (needed so that we can run grad only on no dropout/one inference and with grad with the sampled BNNs)
    run_parameters_no_dropout_one_inf = get_no_dropout_one_inference_run_parameters(run_parameters)
    return loss_fn, train_loaders, val_target_loaders, models, optimizers, acq, fieldnames, output_dir, train_csv, train_per_image_csv, train_all_dnns_csv, train_per_image_all_dnns_csv, val_target_csv, train_dataset_length, run_parameters_no_dropout_one_inf, run_parameters, intrinsics_seqs

'''
Description: helper function to run validation with or without dropout based on configuration file 
Input:       
    run_parameters:         run parameters
    run_parameters_no_dropout_one_inf:    run parameters modified with p = 0 and number of inferences = 1 
    val_loader:             data loader for validation
    models:                 list of loaded models
    loss_fn:                loss function
    write_to_file:          flag to indicate whether to write out validation stats to csv
    val_target_csv:         csv file for validation stats on target 
    fieldnames:             column names in csv files 
    output_dir:             output directory from command line arguments 
    val_dataset_idx:        index of validation dataset in dataloaders list
    train_seq_idx:          sequence index
    num_weight_updates:     number of weight updats to DNN so far
    num_decisions_train:    number of times acquisition function returns decision to train
'''
def run_validation(run_parameters, run_parameters_no_dropout_one_inf, val_loader, models, loss_fn, write_to_file, test_csv, fieldnames, output_dir, val_dataset_idx, train_seq_idx, num_weight_updates, num_decisions_train, train_epoch):
    # enables toggling between running with MC-dropout or not for validation
    if run_parameters['validate'] == "validate": # evaluate on validation sets before training, using set parameters in run_parameters
        validate(run_parameters, val_loader, models, write_to_file, test_csv, fieldnames, output_dir, val_dataset_idx, train_seq_idx, num_weight_updates, num_decisions_train, train_epoch) # evaluate on validation set 
    elif run_parameters['validate'] == "validate_no_dropout": # set run parameters to non-dropout mode with one inference for actual validation
        validate(run_parameters_no_dropout_one_inf,val_loader, models, write_to_file, test_csv, fieldnames, output_dir, val_dataset_idx, train_seq_idx, num_weight_updates, num_decisions_train, train_epoch) # evaluate on validation set
    return

'''
Description: runs training loop (called by ./examples/train_dnn.py)
Input:       
    run_parameters:         run parameters
Output:
    run_parameters:         updated run parameters
'''

def train(run_parameters, output_dir):
    # check valid configuration 
    check_valid_config(run_parameters)
    # initialize models, loss function, dataloaders, optimizers, csv files, dataset length, backward compatibility for run_parameters for online training 
    loss_fn, train_loaders, val_target_loaders, models, optimizers, acq, fieldnames, output_dir, train_csv, train_per_image_csv, train_all_dnns_csv, train_per_image_all_dnns_csv, val_target_csv, train_dataset_length, run_parameters_no_dropout_one_inf, run_parameters, intrinsics_seqs = initialize_training(run_parameters, output_dir)
    print("Starting online training with the following run_parameters: ")
    print(run_parameters)
    # Initialize online buffer
    online_buffer = buffer.CoDEPS_Buffer(run_parameters)
    # train online on datasets
    seq_idx_bias = 0
    assert len(train_loaders) == len(intrinsics_seqs)
    average_meter, average_meter_all_dnns, models, epoch_start_time, num_weight_updates, num_decisions_train = initiate_train_epoch(models) # intialize metrics, put DNN in train mode
    for train_dataset_idx in range(len(train_loaders)): # iterate through each sequence in potentially stitched sequence 
        print(f"Training sequence {train_dataset_idx}...")
        for epoch in range(0, run_parameters['epochs']): # number of online training runs (for online training, only one epoch)
            seq_length = len(train_loaders[train_dataset_idx].dataset) # length of this sequence in stitched sequence
            print("Stitched sequence length: " + str(train_dataset_length) + ", seq. length in this segment of stitched sequence: " + str(seq_length))
            print("**********************" + "Training on sequence: " + run_parameters['train_dir'][train_dataset_idx] + "***************************")
            num_weight_updates, num_decisions_train = train_online(
                run_parameters,run_parameters_no_dropout_one_inf, train_loaders[train_dataset_idx], models, loss_fn, optimizers, 
                epoch, train_csv, train_per_image_csv, train_all_dnns_csv, train_per_image_all_dnns_csv, fieldnames, 
                acq, train_dataset_idx, output_dir, seq_length, seq_idx_bias, intrinsics_seqs[train_dataset_idx], online_buffer, 
                average_meter, average_meter_all_dnns, epoch_start_time, num_weight_updates, num_decisions_train,
                val_target_loaders, val_target_csv
            ) # train for one epoch
            seq_idx_bias += seq_length
            print("Number of backprop passes on epoch " + str(epoch) + ": " + str(num_weight_updates))
    torch.cuda.empty_cache()
    return run_parameters