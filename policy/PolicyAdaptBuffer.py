import numpy as np 
import random

from policy.onlineStatsLoader import onlineStatsLoader

class PolicyAdaptBuffer:
    # goal:
    # 1. load source data when initializing
    # 2. store target data in structure when pushed
    # 3. sample data (from source, target buffer) for policy adaptation
    # issue to be considered:
    # 1. which source data to load
    # 2. imbalancing issue: target data pushed only when it predicted "train"

    def __init__(self, configs):
        self.configs = configs
        self.init_model_configs = None # configs from the initial model
        self.source_buffer_load_path   = configs['source_buffer_load_path'] # path to source data buffer
        self.source_buffer_max_size    = configs['source_buffer_max_size']
        self.target_buffer_max_size    = configs['target_buffer_max_size']
        self.source_buffer_push_policy = configs['source_buffer_load_policy'] # all, random
        self.target_buffer_pop_policy  = configs['target_buffer_pop_policy'] # fifo, random
        self.activate_source_buffer    = configs['activate_source_buffer']

        # source data buffer
        self.source_buffer_input  = []
        self.source_buffer_output = []
        # target data buffer
        self.target_buffer_input  = []
        self.target_buffer_output = []
        self.target_buffer_query  = []

    def _load_source_data(self):
        # load source training data using same setup as init model offline training
        print("[Decision DNN Adapt Info] Load source data for policy adaptation: ", self.init_model_configs['dataset_cfgs']['train']['seq_names'])
        model_cfgs, output_cols, input_cols = \
            self.init_model_configs['model_cfgs'], self.init_model_configs["dataset_cfgs"]["output_cols"], self.init_model_configs["dataset_cfgs"]["input_cols"]
        train_seq_names    = self.init_model_configs['dataset_cfgs']['train']['seq_names']
        train_model_groups = self.init_model_configs['dataset_cfgs']['train']['model_groups']
        train_model_idx    = self.init_model_configs['dataset_cfgs']['train']['model_idx']
        train_input_rows   = self.init_model_configs['dataset_cfgs']['train']['input_rows']
        train_pattern_idx  = self.init_model_configs['dataset_cfgs']['train']['pattern_idx']
        dataloader = onlineStatsLoader(self.source_buffer_load_path, self.init_model_configs['dataset_cfgs']['sma_window'], self.init_model_configs['dataset_cfgs']['ewm_alpha'])
        src_x_train, src_y_train, norm_meta_train = dataloader.get_online_stats(train_seq_names, train_model_groups, train_model_idx, train_pattern_idx, output_cols, train_input_rows, model_cfgs["biased"], {'method': model_cfgs['normalize']}, input_cols)
        # select source data to store online
        if self.source_buffer_push_policy == 'all':
            print("[Decision DNN Adapt Info] Load all source data to buffer, total size: ", src_x_train.shape[0])
            self.source_buffer_input = src_x_train
            self.source_buffer_output = src_y_train
        elif self.source_buffer_push_policy == 'random':
            idx = random.sample(range(len(src_x_train)), self.source_buffer_max_size)
            self.source_buffer_input = [src_x_train[i] for i in idx]
            self.source_buffer_output = [src_y_train[i] for i in idx]
        else:
            raise ValueError('Invalid source buffer push policy')
    
    def _pop_target_data(self):
        if self.target_buffer_pop_policy == 'fifo':
            self.target_buffer_input.pop(0)
            self.target_buffer_output.pop(0)
            self.target_buffer_query.pop(0)
        elif self.target_buffer_pop_policy == 'random':
            idx = random.randint(0, len(self.target_buffer_input)-1)
            self.target_buffer_input.pop(idx)
            self.target_buffer_output.pop(idx)
            self.target_buffer_query.pop(idx)
        else:
            raise ValueError('Invalid target buffer pop policy')

    ############################################################################################################
    # Public methods
    ############################################################################################################
    def setup_source_buffer(self, init_model_configs):
        assert self.activate_source_buffer is not None
        self.init_model_configs = init_model_configs
        if self.activate_source_buffer:
            self._load_source_data()
        else:
            print("[Decision DNN Adapt Info] Source buffer is not activated")
        
    def push_target_data(self, seq_idx, x, y):
        self.target_buffer_query.append(seq_idx)
        self.target_buffer_input.append(x)
        self.target_buffer_output.append(y)
        while len(self.target_buffer_input) > self.target_buffer_max_size:
            self._pop_target_data()
        return

    def sample_data(self, target_replay, source_replay):
        # target_replay: number of target data to sample
        # source_replay: number of source data to sample, ignore when source buffer is not activated
        sampled_input, sampled_output = [], []
        # sample target data
        if len(self.target_buffer_input) < target_replay:
            print(f"[Decision DNN Adapt Info] Target buffer size is less than target replay size ==> ({len(self.target_buffer_input)}/{target_replay})")
            target_replay = len(self.target_buffer_input)
        idx = random.sample(range(len(self.target_buffer_input)), target_replay)
        sampled_input += [self.target_buffer_input[i] for i in idx]
        sampled_output += [self.target_buffer_output[i] for i in idx]
        # sample source data
        if self.activate_source_buffer:
            assert len(self.source_buffer_input) > source_replay
            idx = random.sample(range(len(self.source_buffer_input)), source_replay)
            sampled_input += [self.source_buffer_input[i] for i in idx]
            sampled_output += [self.source_buffer_output[i] for i in idx]
        sampled_x, sampled_y = np.vstack(sampled_input), np.vstack(sampled_output)
        return sampled_x, sampled_y