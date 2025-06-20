import numpy as np 
import torch
from scipy.stats import norm
from scipy.stats import kstest

import math
import time # debugging
import bisect # sorted list insertion  
from . import train as trainer
import sys 
import os
import csv
import pickle
import torch.nn.functional as F
# from pypapi import papi_high
# from pypapi import events as papi_events

# superpixel imports
from skimage.segmentation import slic
import matplotlib.pyplot as plt
import scipy 

from nets.camera import Pose
from skimage.metrics import structural_similarity as ssim
from sklearn.kernel_ridge import KernelRidge
from policy.PolicyTrainer import OnlinePolicyTrainer

class AcquisitionFunction():
    def __init__(self, run_parameters, seq_length, output_dir):
        # track number of times decided to train 
        self.num_train_decisions = 0 # initialize to 0
        self.seq_length = seq_length # length of full online training sequence (relevant for buffer adaptive allocation)
        # writing stats out to file parameters 
        self.acquisition_csv = os.path.join(output_dir, 'acquisition.csv') 
        # baseline acqusisition functions 
        self.run_parameters = run_parameters
        if run_parameters['acquisition_type'] == "none": 
            self.decide_to_train = self.return_false
        elif run_parameters['acquisition_type'] == "all": 
            self.decide_to_train = self.return_true
        elif run_parameters['acquisition_type'] == "random":
            self.decide_to_train = self.select_random
        elif run_parameters['acquisition_type'] == "index":
            self.decide_to_train = self.select_by_idx
        elif run_parameters['acquisition_type'] == "interval":
            self.decide_to_train = self.select_by_interval
        elif run_parameters['acquisition_type'] == "pattern":
            self.decide_to_train = self.select_by_pattern
        elif run_parameters['acquisition_type'] == "balance_compute_accuracy_gain_zero_regret":
            self.decide_to_train = self.balance_compute_accuracy_gain_zero_regret
            self.alpha = self.run_parameters['alpha'] # weight on predicted accuracy gain
            self.regret_metric = run_parameters['regret_metric']
        elif run_parameters['acquisition_type'] == "balance_compute_accuracy_gain_policy":
            self.decide_to_train = self.balance_compute_accuracy_gain_policy
            self.alpha = self.run_parameters['alpha'] # weight on predicted accuracy gain
            # load in policy trained weights
            assert self.run_parameters['policy_model_path'].endswith(".pkl")
            online_policy_configs = {
                "biased"                   : run_parameters["online_policy_biased"],
                "norm"                     : run_parameters["online_policy_norm"],
                "online_train_method"      : run_parameters["online_policy_train_method"],
                "output_type"              : run_parameters["online_policy_output_type"],
                "online_training_configs"  : run_parameters["online_training_configs"],
                "online_buffer_configs"    : run_parameters["online_buffer_configs"],
            }
            self.policy_model = OnlinePolicyTrainer(online_policy_configs)
            self.policy_model.load_model(self.run_parameters['policy_model_path'])
        return           

    def return_false(self, *args):
        # write out to files
        seq_idx = int(args[0])
        stats = {'seq_idx': seq_idx, 'train_decision': False, 'decision_type': "not_active"}
        self.write_stats_to_file(stats)
        return False 
    
    def return_true(self, *args):
        # read in args 
        seq_idx = int(args[0]) # index of current image in full (stitched) sequence
        this_seq_idx = int(args[11]) # index of this sequence in stitched sequence         
        if this_seq_idx < 2: # first two images of a sequence has empty policy_feat and not enough near views for loss calculation
            train_decision = False            
        else:
            train_decision = True 
            self.num_train_decisions+=1 
        # write out to file
        stats = {'seq_idx': seq_idx, 'train_decision': train_decision, 'decision_type': "not_active"}
        self.write_stats_to_file(stats)
        return train_decision 

    def select_random(self, *args):
        seq_idx = int(args[0])
        if np.random.random() < self.run_parameters['random_percentage']: 
            train_decision = True
        else: 
            train_decision = False 
        # write out to file
        stats = {'seq_idx': seq_idx, 'train_decision': train_decision, 'decision_type': "not_active"}
        self.write_stats_to_file(stats)
        return train_decision 

    def select_by_idx(self, *args):
        seq_idx = args[0]
        if seq_idx in self.run_parameters['seq_idx_train']:
            train_decision = True
        else:
            train_decision = False
        # write out to file
        seq_idx = int(args[0])
        stats = {'seq_idx': seq_idx, 'train_decision': train_decision, 'decision_type': "not_active"}
        self.write_stats_to_file(stats)
        return train_decision 

    def select_by_interval(self, *args):
        seq_idx = args[0]
        if (seq_idx % self.run_parameters['interval'] == 0):
            train_decision = True
        else:
            train_decision = False
        # write out to file
        stats = {'seq_idx': seq_idx, 'train_decision': train_decision, 'decision_type': "not_active"}
        self.write_stats_to_file(stats)
        return train_decision 

    def select_by_pattern(self, *args):
        seq_idx = args[0]
        selected_pattern = self.run_parameters['select_pattern'] # list, in form of [0, 1, 0, 0, ...], which indicates whether to train at each time step in a pattern
        if selected_pattern[seq_idx % len(selected_pattern)] == 1:
            train_decision = True
        else:
            train_decision = False
        # write out to file
        stats = {'seq_idx': seq_idx, 'train_decision': train_decision, 'decision_type': "not_active"}
        self.write_stats_to_file(stats)
        return train_decision 
    
    def write_stats_to_file(self, stats, add_header=False):
        with open(self.acquisition_csv, 'a') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=stats.keys())
            # add header if first time 
            if stats['seq_idx'] == 0: # or add_header:
                writer.writeheader()
            writer.writerow(stats)
        return 

    def balance_compute_accuracy_gain_zero_regret(self, *args, **kwargs):
        if 'metric_gain' not in kwargs:
            return False # need to return false, otherwise will have no "recovered model" when running metric_gain
        else:
            # read in args
            seq_idx = int(kwargs['seq_idx']) # index of current image in full (stitched) sequence
            this_seq_idx = int(kwargs['this_seq_idx']) # index of this sequence in stitched sequence 
            
            if this_seq_idx < 2: # first two images of a sequence has empty policy_feat and not enough near views for loss calculation
                stats = {'seq_idx': seq_idx, 'train_decision': "False", 'groundtruth_gain': None, 
                        'decision_type': "not_active"}
                self.write_stats_to_file(stats)
                return False
            # decide to train based on groundtruth accuracy gain
            metric_gain = kwargs['metric_gain'][self.regret_metric] if self.regret_metric in ['delta1', 'delta1_lit'] else -kwargs['metric_gain'][self.regret_metric]
            if -1+self.alpha*metric_gain > 0:
                self.num_train_decisions+=1
                train_decision = True
            else:
                train_decision = False
            # write out to file values
            stats = {'seq_idx': seq_idx, 'train_decision': str(train_decision), 'groundtruth_gain': kwargs['metric_gain'][self.regret_metric], 'train_reward': -1+self.alpha*metric_gain, 'decision_type': "active"}
            self.write_stats_to_file(stats)
            return train_decision

    def balance_compute_accuracy_gain_policy(self, *args, **kwargs):
        # read in args 
        seq_idx = int(args[0]) # index of current image in full (stitched) sequence
        this_seq_idx = int(args[11]) # index of this sequence in stitched sequence 
        policy_feat = args[14] # dictionary of features used in policy
        
        if this_seq_idx < 2: # first two images of a sequence has empty policy_feat and not enough near views for loss calculation
            stats = {'seq_idx': seq_idx, 'train_decision': "False", 'predicted_acc_gain': None, 
                     'decision_type': "not_active"}
            self.write_stats_to_file(stats)
            return False
        # decide to train or not
        predicted_acc_gain = self.run_policy(policy_feat)
        # compute whether predicted accuracy gain is worth computational energy trade-off 
        if -1+self.alpha*predicted_acc_gain > 0:
            self.num_train_decisions+=1
            train_decision = True
        else:
            train_decision = False
        
        # write out to acq stats values
        stats = {'seq_idx': seq_idx,
                'train_decision': str(train_decision), 
                'predicted_acc_gain': predicted_acc_gain, 
                'train_reward': -1+self.alpha*predicted_acc_gain,
                'decision_type': "active"}
        stats.update(policy_feat)
        self.write_stats_to_file(stats) #, add_header=(this_seq_idx==2))
        return train_decision
    
    def run_policy(self, policy_feat):
        predictions = self.policy_model.predict(policy_feat) # (B,)
        # print(predictions) 
        return predictions[0] # loss reduction

    def update_online_policy(self, train_decision, seq_idx, x_stats, y_stats, ablation_targets=None):
        # called when decided to train, and has updated input/output data
        if not train_decision:
            return
        self.policy_model.update_policy(seq_idx=seq_idx, x=x_stats, y=y_stats, ablation_targets=ablation_targets)