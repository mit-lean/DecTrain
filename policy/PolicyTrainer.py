import os
import time
import csv
import numpy as np 

import yaml 
import math
import copy
import pickle

from policy.onlineStatsLoader import onlineStatsLoader
from policy.DNNPolicyTrainer import DNNPolicyTrainer
from policy.PolicyAdaptBuffer import PolicyAdaptBuffer

class PolicyMetricRecorder:

    def __init__(self, result_dir=None):
        self.result_dir = result_dir
        self.results = {}
        os.makedirs(self.result_dir, exist_ok=True)

    def record_metrics(self, tag, metrics):
        if tag not in self.results:
            self.results[tag] = {}
        self.results[tag].update(metrics)
        return
    
    def load_results_csv(self, csv_file):
        if self.result_dir is None:
            print("[Result Recorder] No result directory specified. Skip reading results from csv.")
            return
        
        assert csv_file.endswith(".csv")
        _csv_file = os.path.join(self.result_dir, csv_file)
        print("[Result Recorder] Loading results from: ", _csv_file)
        with open(_csv_file, 'r') as f:
            reader = csv.reader(f)
            header = next(reader)
            for row in reader:
                tag = row[0]
                metrics = {header[i]: float(row[i]) for i in range(1, len(row))}
                self.results[tag] = metrics
        return self.results

    def write_results_csv(self, csv_file):
        if self.result_dir is None:
            print("[Result Recorder] No result directory specified. Skip writing results to csv.")
            return
    
        assert csv_file.endswith(".csv")
        _csv_file = os.path.join(self.result_dir, csv_file)
        print("[Result Recorder] Writing results to: ", _csv_file)
        with open(_csv_file, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(["tag"] + list(self.results[list(self.results.keys())[0]].keys()))
            for tag, metrics in self.results.items():
                writer.writerow([tag] + list(metrics.values()))
        return
    
    def plot_results(self, fig_dir):
        pass

class PolicyTrainer:

    def __init__(self, configs, output_dir):
        self.configs = configs
        self.dataset_dir = self.configs['dataset_dir']
        self.output_dir = output_dir
        self.model_dir = self.output_dir
        self.model = None
        self.norm_meta = None
        self.dataloader = onlineStatsLoader(self.dataset_dir, self.configs['dataset_cfgs']['sma_window'], self.configs['dataset_cfgs']['ewm_alpha'])
        self.result_recorder = PolicyMetricRecorder(self.output_dir)
        self.use_transform = self.configs['dnn_train_cfgs']['use_transform']
        self.show_configs()
        self.w, self.b = 1.0, 0.0
        print(f"[Decision DNN Offline Training] Output directory: {self.output_dir}")
        os.makedirs(self.model_dir, exist_ok=True)

    @property
    def expname(self):
        expname = ".".join([f"{k}={v}" for k, v in self.configs['model_cfgs'].items()])
        expname += ".out=" + "_".join(self.configs['dataset_cfgs']['output_cols'])
        return expname

    def save_model(self, model_file):
        assert model_file.endswith(".pkl")
        model_path = os.path.join(self.model_dir, model_file)
        save_obj = {
            "model": self.model,
            "norm_meta": self.norm_meta,
            "config": self.configs,
            "transform": {'w': self.w, 'b': self.b}
        }
        print("[Decision DNN Offline Training] Saving model to: ", model_path)
        with open(model_path, 'wb') as f:
            pickle.dump(save_obj, f)
        with open(model_path.replace(".pkl", ".yaml"), 'w') as f:
            yaml.dump(self.configs, f)
        return

    def load_model(self, model_file):
        assert model_file.endswith(".pkl")
        model_path = os.path.join(self.model_dir, model_file)
        print("[Decision DNN Offline Training] Loading model from: ", model_path)
        with open(model_path, 'rb') as f:
            load_obj = pickle.load(f)
        self.model = load_obj['model']
        self.norm_meta = load_obj['norm_meta']
        self.configs = load_obj['config']
        self.w = load_obj['transform']['w']
        self.b = load_obj['transform']['b']
        return
    
    def get_policy_model(self, x_train, y_train, model_cfgs, x_val, y_val):
        best_loss = math.inf
        # setup decision DNN model architecture and train model offline
        dims = "-".join([str(model_cfgs['input_dim'])] + [ str(x) for x in model_cfgs['hidden_dims']] + [str(model_cfgs['output_dim'])])
        print(f"[Decision DNN Offline Training] Training model: {model_cfgs['model']} ({dims})")
        exec_time, best_loss = self.get_dnn_model(x_train, y_train, x_val, y_val, self.configs)
        print(f"[Decision DNN Offline Training] Training time: {exec_time:.4f} seconds")
        return exec_time, best_loss
    
    def get_dnn_model(self, x_train, y_train, x_val, y_val, all_configs):
        if self.model is None:
            self.model = DNNPolicyTrainer(all_configs)
        start_time = time.time()
        best_loss = self.model.fit(x_train, y_train, x_val, y_val)
        exec_time = time.time() - start_time
        return exec_time, best_loss
    
    def get_precision_recall(self, y_test, y_pred):
        y_test_bin = np.sign(y_test)
        y_pred_bin = np.sign(y_pred)
        # predict increase, gt increase
        TP = (y_pred_bin[y_test_bin == 1] == 1).sum()
        TN = (y_pred_bin[y_test_bin != 1] != 1).sum()
        FP = (y_test_bin[y_pred_bin == 1] != 1).sum()
        FN = (y_test_bin[y_pred_bin != 1] == 1).sum()
        assert TP + FP == (y_pred_bin == 1).sum()
        assert TP + FN == (y_test_bin == 1).sum()
        assert TN + FP == (y_test_bin != 1).sum()
        assert TN + FN == (y_pred_bin != 1).sum()
        precision_1 = 1.0 * TP / (TP + FP)
        recall_1 = 1.0 * TP / (TP + FN)
        precision_0 = 1.0 * TN / (TN + FN)
        recall_0 = 1.0 * TN / (TN + FP)
        return precision_1, recall_1, precision_0, recall_0
    
    def get_test_label_balance(self, y_test):
        y_test_bin = np.sign(y_test)
        label_1 = (y_test_bin == 1).sum()
        label_0 = (y_test_bin != 1).sum()
        return label_1, label_0
    
    def evaluate_model(self, x_test, y_test, output_cols=None, scatter_plot_fname=None):
        predictions = self.model.predict(x_test)
        if predictions.ndim == 1: # for SVR
            predictions = predictions.reshape(-1, 1)
        if output_cols is None:
            output_cols = self.dataloader._get_input_output_cols()[1] # get all output cols
        metrics = {}
        for i, output_col in enumerate(output_cols):
            p1, r1, p0, r0, = self.get_precision_recall(y_test=y_test[:, i], y_pred=predictions[:, i])
            metrics[f"{output_col}_precision (1)"] = p1
            metrics[f"{output_col}_recall (1)"] = r1
            metrics[f"{output_col}_precision (0)"] = p0
            metrics[f"{output_col}_recall (0)"] = r0
        correlation = np.corrcoef(predictions, y_test, rowvar=False)
        for i, output_col in enumerate(output_cols):
            metrics[f"cor ({output_col})"] = correlation[i, len(output_cols)+i]
            if i == 0:
                scatter_correration = metrics[f"cor ({output_col})"]
        metrics["l1-loss"] = np.mean(np.abs(predictions - y_test))

        print("[Decision DNN Offline Training] Evaluation results: ")
        for k, v in metrics.items():
            print(f"\t{k}: {v:.4f}")
        
        return metrics

    def get_transform(self, y_pred, y_test):
        if not self.use_transform:
            return
        # setup transform s.t. y_test = a * y_pred + b
        # record a, b
        assert y_pred.ndim == 1
        assert y_test.ndim == 1
        assert y_pred.shape == y_test.shape
        self.w = np.cov(y_pred, y_test)[0, 1] / np.var(y_pred)
        self.b = np.mean(y_test) - self.w * np.mean(y_pred)
        print(f"[Decision DNN Offline Training] Get transform: y_test = {self.w:.4f} * y_pred + {self.b:.4f}")
        return
    
    def transform(self, y_pred):
        if not self.use_transform:
            return y_pred
        assert hasattr(self, 'w') and hasattr(self, 'b')
        return self.w * y_pred + self.b
    
    def show_configs(self):
        shown_cfgs = {
            "dataset_dir": self.configs["dataset_dir"],
            "dataset_cfgs": {
                "output_cols": self.configs["dataset_cfgs"]["output_cols"],
                "train": self.configs["dataset_cfgs"]["train"]["seq_names"],
                "tests": {f"test_{i}": self.configs["dataset_cfgs"]["tests"][i]["seq_names"] for i in range(len(self.configs["dataset_cfgs"]["tests"]))} 
            },
            "model_cfgs": self.configs['model_cfgs'],
            "train_cfgs": self.configs['dnn_train_cfgs']
        }
        print(yaml.dump(shown_cfgs, default_flow_style=False)) # default_flow_style=None
    
    ############################################################################################################
    # Public methods
    ############################################################################################################
    """
    Offline Policy Training Input Constraints (on current state)
        1. Input features can only be collected from the current state (i.e. no temporal features)
        2. Input features (of training) can come from different sequences
        3. Predictions can include groundtruth (e.g. accuracy, error of depth)
    """
    """
    Online Policy Training Scenario
        1. Collect train data (warm up or whenever the initial policy determines to train): for every step, collect the a. input statistics b. loss reduction after conducting training
        2. Deploy: use the updated policy for the upcoming steps
    Limitations of Online Policy Training (on current state)
        1. Input features can only be collected from the current state (i.e. no temporal features)
        2. Input features (of training) can only come from one setup (same initial model, same seq, same pattern)
        2. Predictions cannot include groundtruth (e.g. self-supervised loss)
    """
    def train_offline_model(self):
        # config parsing
        model_cfgs, output_cols, input_cols = \
            self.configs['model_cfgs'], self.configs["dataset_cfgs"]["output_cols"], self.configs["dataset_cfgs"]["input_cols"]
        # training data
        train_seq_names    = self.configs['dataset_cfgs']['train']['seq_names']
        train_model_groups = self.configs['dataset_cfgs']['train']['model_groups']
        train_model_idx    = self.configs['dataset_cfgs']['train']['model_idx']
        train_input_rows   = self.configs['dataset_cfgs']['train']['input_rows']
        train_pattern_idx  = self.configs['dataset_cfgs']['train']['pattern_idx']

        # load data
        x_train, y_train, x_val, y_val, norm_meta_train = self.dataloader.get_online_stats(train_seq_names, train_model_groups, train_model_idx, train_pattern_idx, output_cols, train_input_rows, model_cfgs["biased"], {'method': model_cfgs['normalize']}, input_cols, val_split=0.2)
        self.norm_meta = norm_meta_train
        
        best_loss = math.inf
        best_model = None
        max_iteration = 20
        for i in range(max_iteration):
            print(f"[Decision DNN Offline Training] Iteration: {i+1}/{max_iteration}")
            # train decision DNN model
            exec_time, _best_loss = self.get_policy_model(x_train, y_train, self.configs['model_cfgs'], x_val=x_val, y_val=y_val)
            # get transform
            predictions = self.model.predict(x_train)
            self.get_transform(predictions[:, 0], y_train[:, 0])
            # evaluate model
            print(f"[Decision DNN Offline Training] Evaluating model: training ({self.expname})")
            metrics_train = self.evaluate_model(x_train, y_train, output_cols)
            # record results
            self.result_recorder.record_metrics(tag=f"train", metrics=metrics_train)
            # save model
            self.save_model(f"policy_model_{i}.pkl")
            # save results
            self.result_recorder.write_results_csv(f"train_report_{i}.csv")
            # validation
            self.test_offline_model(tag=f"{i}")
            # record best model
            if best_loss > _best_loss:
                print(f"[Decision DNN Offline Training] Loss: {best_loss:.4f} --> {_best_loss:.4f}")
                best_loss = _best_loss
                best_model = copy.deepcopy(self.model)
        self.model = best_model
        self.save_model(f"policy_model_best_val.pkl")
        return

    def test_offline_model(self, model_name=None, tag="final"):
        # config parsing
        model_cfgs, output_cols, input_cols = \
            self.configs['model_cfgs'], self.configs['dataset_cfgs']["output_cols"], self.configs['dataset_cfgs']['input_cols']
        # load trained model
        if self.model is None:
            if model_name is None:
                model_name = f"policy_model.pkl"
            self.load_model(model_name)
        
        # training data
        train_seq_names    = self.configs['dataset_cfgs']['train']['seq_names']
        train_model_groups = self.configs['dataset_cfgs']['train']['model_groups']
        train_model_idx    = self.configs['dataset_cfgs']['train']['model_idx']
        train_input_rows   = self.configs['dataset_cfgs']['train']['input_rows']
        train_pattern_idx  = self.configs['dataset_cfgs']['train']['pattern_idx']
        # testing data
        tests_seq_names    = [ test['seq_names'] for test in self.configs['dataset_cfgs']['tests']]
        tests_model_groups = [ test['model_groups'] for test in self.configs['dataset_cfgs']['tests']]
        tests_model_idx    = [ test['model_idx'] for test in self.configs['dataset_cfgs']['tests']]
        tests_input_rows   = [ test['input_rows'] for test in self.configs['dataset_cfgs']['tests']]
        tests_pattern_idx  = [ test['pattern_idx'] for test in self.configs['dataset_cfgs']['tests']]

        # load data and evaluate
        x_train, y_train, _ = self.dataloader.get_online_stats(train_seq_names, train_model_groups, train_model_idx, train_pattern_idx, output_cols, train_input_rows, model_cfgs["biased"], self.norm_meta, input_cols)
        print(f"[Decision DNN Offline Training] Evaluating model: training ({self.expname})")
        metrics_train = self.evaluate_model(x_train, y_train, output_cols, scatter_plot_fname=os.path.join(self.output_dir, f"scatter_plot_train_{tag}.png"))
        self.result_recorder.record_metrics(tag=f"train", metrics=metrics_train)

        for i in range(len(tests_seq_names)):
            x_test, y_test, _ = self.dataloader.get_online_stats(tests_seq_names[i], tests_model_groups[i], tests_model_idx[i], tests_pattern_idx[i], output_cols, tests_input_rows[i], model_cfgs["biased"], self.norm_meta, input_cols)
            # x_test, y_test, _ = self.dataloader.get_online_stats(tests_seq_names[i], tests_model_groups[i], tests_model_idx[i], tests_pattern_idx[i], output_cols, tests_input_rows[i], model_cfgs["biased"], {'method': model_cfgs['normalize']})
            print(f"[Decision DNN Offline Training] Evaluating model: test {i+1} ({self.expname})")
            metrics_test = self.evaluate_model(x_test, y_test, output_cols, scatter_plot_fname=os.path.join(self.output_dir, f"scatter_plot_test{i}_{tag}.png"))
            self.result_recorder.record_metrics(tag=f"test{i+1}", metrics=metrics_test)

        # save results
        self.result_recorder.write_results_csv(f"test_report_{tag}.csv")

        return

class OnlinePolicyTrainer:

    def __init__(self, configs):
        self.configs = configs
        self.model = None
        self.norm_meta = None
        self.dataloader = onlineStatsLoader(None)

        # prediction type
        self.input_biased = self.configs['biased']
        self.output_type = self.configs['output_type']
        assert self.output_type in ['accuracy', 'error']
        self.w = 1
        self.b = 0

        # init model
        self.init_model = None
        self.init_norm_meta = None

        # online training setup
        self.online_train_method = self.configs['online_train_method']
        if self.online_train_method != 'none':
            self.online_training_configs = self.configs['online_training_configs']
        self.replay_buffer = PolicyAdaptBuffer(configs['online_buffer_configs'])

        # ablation study
        self.ablation_idxs = None

    @property
    def expname(self):
        excluded_keys = ['dataset_dir', 'model_dir', 'result_dir', 'model_file', 'result_file']
        expname = ".".join([f"{k}={v}" for k, v in self.configs.items() if k not in excluded_keys])
        return expname
    
    def save_model(self, model_path):
        assert model_path.endswith(".pkl")
        save_obj = {
            "model": self.model,
            "norm_meta": self.norm_meta,
            "config": self.configs,
            "transform": {'w': self.w, 'b': self.b}
        }
        print("[Decision DNN Info] Saving model to: ", model_path)
        with open(model_path, 'wb') as f:
            pickle.dump(save_obj, f)
        return

    def load_model(self, model_path):
        assert model_path.endswith(".pkl")
        print("[Decision DNN Info] Loading model from: ", model_path)
        with open(model_path, 'rb') as f:
            load_obj = pickle.load(f)
        self.model = load_obj['model']
        self.norm_meta = load_obj['norm_meta']
        self.offline_cofigs = load_obj['config']
        self.train_input_cols = self.offline_cofigs['dataset_cfgs']['input_cols']
        if "use_transform" in self.offline_cofigs['dnn_train_cfgs']:
            self.use_transform = self.offline_cofigs['dnn_train_cfgs']['use_transform']
        else:
            self.use_transform = True
        if 'transform' in load_obj:
            self.w = load_obj['transform']['w']
            self.b = load_obj['transform']['b']
        else:
            self.w = 1
            self.b = 0
        # initialize source buffer (if activated)
        self.replay_buffer.setup_source_buffer(self.offline_cofigs)
        return
    
    def get_transform(self, y_pred, y_test):
        if not self.use_transform:
            return
        # setup transform s.t. y_test = a * y_pred + b
        # record a, b
        assert y_pred.ndim == 1
        assert y_test.ndim == 1
        assert y_pred.shape == y_test.shape
        self.w = np.cov(y_pred, y_test)[0, 1] / np.var(y_pred)
        self.b = np.mean(y_test) - self.w * np.mean(y_pred)
        return
    
    def transform(self, y_pred):
        if not self.use_transform:
            return y_pred
        return self.w * y_pred + self.b
    
    # utils: data preparation
    def _status_to_features(self, stats):
        return self.dataloader.stats_to_feat(stats, biased=self.input_biased, norm_meta=self.norm_meta, train_input_cols=self.train_input_cols)
    
    def _get_data_from_buffer(self, target_replay, source_replay):
        sampled_inputs, sampled_outputs = self.replay_buffer.sample_data(target_replay, source_replay)
        return sampled_inputs, sampled_outputs

    def _prepare_data(self, x_curr_stat, y_curr_stat):
        # setup
        train_target_replay = self.online_training_configs['train_target_replay']
        train_source_replay = self.online_training_configs['train_source_replay']
        val_target_replay   = self.online_training_configs['val_target_replay']
        val_source_replay   = self.online_training_configs['val_source_replay']
        # current data
        x_curr = self._status_to_features(x_curr_stat)
        # training data from target buffer and source buffer
        x_train, y_train = self._get_data_from_buffer(target_replay=train_target_replay, source_replay=train_source_replay)
        # add current data to training data
        x_train = np.vstack([x_curr     , x_train])
        y_train = np.vstack([y_curr_stat, y_train])
        # validation data from target buffer and source buffer
        x_val, y_val = self._get_data_from_buffer(target_replay=val_target_replay, source_replay=val_source_replay)
        return x_train, y_train, x_val, y_val

    # ablation target zero-out
    def _ablate_features(self, x, ablation_targets):
        # find out ablation feature index
        if self.ablation_idxs is None:
            ignore_keywords = []
            if "epistemic_unc" in ablation_targets:
                ignore_keywords.append("epistemic")
            if "aleatoric_unc" in ablation_targets:
                ignore_keywords.append("aleatoric")
            if "depth" in ablation_targets:
                ignore_keywords.append("depth")
            if "loss" in ablation_targets:
                ignore_keywords.append("prev_loss_curr")
            if "pose" in ablation_targets:
                ignore_keywords.append("translation")
                ignore_keywords.append("rotation")
            if "landmark" in ablation_targets:
                ignore_keywords.append("landmark")
            self.ablation_idxs = [i for i, col in enumerate(self.train_input_cols) if any([kw in col for kw in ignore_keywords])]
        # zero-out the target feature
        for idx in self.ablation_idxs:
            x[:, idx] = 0
        return x
    
    # define model update method
    def _update_model(self, x_curr_stat, y_curr_stat, ablation_targets=None):
        if self.online_train_method == 'none':
            # always use initial model
            return
        elif self.online_train_method == 'dnn-adapt':
            # adapt the current model with the online train data
            # adaptation setups
            batch_size = self.online_training_configs['batch_size']
            epochs = self.online_training_configs['epochs']
            if 'early_stop_epochs' in self.online_training_configs:
                early_stop_epochs = self.online_training_configs['early_stop_epochs']
            else:
                early_stop_epochs = None
            # data preparation
            x_train, y_train, x_val, y_val = self._prepare_data(x_curr_stat, y_curr_stat)
            # ablation study
            if ablation_targets not in [None, "none"]:
                x_train = self._ablate_features(x_train, ablation_targets)
                x_val = self._ablate_features(x_val, ablation_targets)
            # dnn adaptation
            self.model.adapt(x_train, y_train, x_val, y_val, batch_size, epochs, early_stop_epochs)
            # update transform
            x_all, y_all = np.vstack([x_train, x_val]), np.vstack([y_train, y_val])
            predictions = self.model.predict(x_all)
            self.get_transform(predictions[:, 0], y_all[:, 0])
        return
    
    ############################################################################################################
    # Public methods
    ############################################################################################################
    def predict(self, x_stats):
        # transfer input stats to input features
        x_test_norm = self.dataloader.stats_to_feat(x_stats, biased=self.input_biased, norm_meta=self.norm_meta, train_input_cols=self.train_input_cols)
        predictions = self.model.predict(x_test_norm)
        # transform predictions to gain
        predictions = self.transform(predictions)
        if self.output_type == 'error':
            predictions = -predictions
        return predictions # return loss reduction (so higher reward when predict minus)
    
    def update_policy(self, seq_idx, x, y, ablation_targets=None):
        # update model based on current data, target replay, and source replay
        self._update_model(x, y, ablation_targets=ablation_targets)
        # push data to buffer
        x_feat = self._status_to_features(x)
        self.replay_buffer.push_target_data(seq_idx, x_feat, y)
        return
    