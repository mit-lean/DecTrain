import os
import numpy as np 
import pandas as pd

class onlineStatsLoader:

    def __init__(self, online_stats_dir, sma_window=None, ewm_alpha=None):
        self.online_stats_dir = online_stats_dir
        self.sma_window = sma_window # 30
        self.ewm_alpha = ewm_alpha # 0.1
    
    def _load_online_stats(self, loaded_cfgs, loaded_rows):
        # loaded_cfgs: list of cfgs to be loaded
        assert len(loaded_cfgs) == len(loaded_rows)
        files = [ f"cfg_{i}.csv" for i in loaded_cfgs ]
        online_stats_dfs = []
        for file, rows in zip(files, loaded_rows):
            online_stats_df = pd.read_csv(os.path.join(self.online_stats_dir, file))
            
            # find out the current stats
            current_stats_cols = ['depth_avg', 'depth_var', 'depth_mid', 'depth_max'] 
            for col in current_stats_cols:
                online_stats_df[col] = np.where(
                                online_stats_df['seq_idx'] == online_stats_df['query_2'], online_stats_df[f'{col}_2'], np.where(
                                online_stats_df['seq_idx'] == online_stats_df['query_1'], online_stats_df[f'{col}_1'], np.where(
                                online_stats_df['seq_idx'] == online_stats_df['query_0'], online_stats_df[f'{col}_0'], np.nan)))
            online_stats_df.dropna(inplace=True)
            
            if self.sma_window is not None or self.ewm_alpha is not None:
                for col in current_stats_cols:
                    if self.sma_window is not None:
                        online_stats_df[f"{col}_sma{self.sma_window}"] = online_stats_df[col].to_frame().rolling(self.sma_window).mean()
                    if self.ewm_alpha is not None:
                        online_stats_df[f"{col}_ewm{self.ewm_alpha}"] = online_stats_df[col].to_frame().ewm(alpha=self.ewm_alpha, adjust=False).mean()
                online_stats_df.dropna(inplace=True)
            # add loss diff rate (loss diff / prev_loss_avg)
            # online_stats_df['loss_diff_rate'] = online_stats_df['loss_diff'] / online_stats_df['prev_loss_avg']
            online_stats_df['loss_diff_rate'] = online_stats_df['loss_diff_curr'] / online_stats_df['prev_loss_curr']
            if rows is not None:
                assert len(rows) == 2
                row_start, row_end = rows[0], rows[1]
                online_stats_df = \
                    online_stats_df.loc[(online_stats_df['seq_idx'] >= row_start) & (online_stats_df['seq_idx'] <= row_end)]
            online_stats_dfs.append(online_stats_df)
        online_stats_row_concat = pd.concat(online_stats_dfs, axis=0)
        return online_stats_row_concat
    
    def _normalize_input(self, train_input, normalization, verbose=False):
        norm_metadata = normalization
        norm_method = normalization['method']
        if norm_method == 'none':
            return train_input, norm_metadata
        if norm_method == 'standard':
            if 'std_mean' in normalization:
                if verbose:
                    print(f"[StatsLoader] Using pre-defined mean and std statistics: mean, std = {normalization['std_mean']}, {normalization['std_std']}")
                mean = normalization['std_mean']
                std = normalization['std_std']
            else:
                if verbose:
                    print("[StatsLoader] Calculating mean and std from input data")
                mean = np.mean(train_input, axis=0)
                std = np.std(train_input, axis=0)
                assert np.all(mean[std == 0] == 0)
                std[std == 0] = 1e-6 # avoid division by zero
            norm_metadata['std_mean'] = mean
            norm_metadata['std_std'] = std
            train_input = (train_input - mean[np.newaxis, :]) / std[np.newaxis, :]
        elif norm_method == 'minmax':
            if 'minmax_min' in normalization:
                if verbose:
                    print(f"[StatsLoader] Using pre-defined min and max statistics: min, max = {normalization['minmax_min']}, {normalization['minmax_max']}")
                min_val = normalization['minmax_min']
                max_val = normalization['minmax_max']
            else:
                if verbose:
                    print("[StatsLoader] Calculating min and max from input data")
                min_val = np.min(train_input, axis=0)
                max_val = np.max(train_input, axis=0)
            norm_metadata['minmax_min'] = min_val
            norm_metadata['minmax_max'] = max_val
            train_input = (train_input - min_val) / (max_val - min_val)
        else:
            raise NotImplementedError
        return train_input, norm_metadata
        
    def _get_input_output_cols(self, ignore_input_cols=None):
        """
        Returns the input and output columns of the online stats dataframe
        """
        # train_input_cols = ['depth_avg', 'depth_var', 'depth_mid', 'depth_max', 
        #                     'aleatoric_avg', 'aleatoric_var', 'aleatoric_mid', 'aleatoric_max', 
        #                     'landmarks_avg',
        #                     'translation_avg', 'translation_max', 'translation_min', 
        #                     'rotation_avg', 'rotation_max', 'rotation_min']
        train_input_cols = ['depth_avg', 'depth_var', 'depth_mid', 'depth_max']
        train_input_cols += ['depth_avg_0', 'depth_var_0', 'depth_mid_0', 'depth_max_0', 
                            'aleatoric_avg_0', 'aleatoric_var_0', 'aleatoric_mid_0', 'aleatoric_max_0', 
                            'translation_avg_0', 'translation_max_0', 'translation_min_0', 
                            'rotation_avg_0', 'rotation_max_0', 'rotation_min_0',
                            'landmark_cnt_avg_0']
        train_input_cols += ['depth_avg_1', 'depth_var_1', 'depth_mid_1', 'depth_max_1', 
                            'aleatoric_avg_1', 'aleatoric_var_1', 'aleatoric_mid_1', 'aleatoric_max_1', 
                            'translation_avg_1', 'translation_max_1', 'translation_min_1', 
                            'rotation_avg_1', 'rotation_max_1', 'rotation_min_1',
                            'landmark_cnt_avg_1']
        train_input_cols += ['depth_avg_2', 'depth_var_2', 'depth_mid_2', 'depth_max_2', 
                            'aleatoric_avg_2', 'aleatoric_var_2', 'aleatoric_mid_2', 'aleatoric_max_2', 
                            'translation_avg_2', 'translation_max_2', 'translation_min_2', 
                            'rotation_avg_2', 'rotation_max_2', 'rotation_min_2',
                            'landmark_cnt_avg_2']
        train_input_cols += ['epistemic_avg', 'epistemic_var', 'epistemic_mid', 'epistemic_max']
        train_input_cols += ['prev_loss_curr']
        if self.sma_window is not None:
            train_input_cols += [f"{col}_sma{self.sma_window}" for col in ['depth_avg', 'depth_var', 'depth_mid', 'depth_max', 
                            'aleatoric_avg', 'aleatoric_var', 'aleatoric_mid', 'aleatoric_max', 
                            'translation_avg', 'translation_max', 'translation_min', 
                            'rotation_avg', 'rotation_max', 'rotation_min',
                            'landmark_cnt_avg']]
        if self.ewm_alpha is not None:
            train_input_cols += [f"{col}_ewm{self.ewm_alpha}" for col in ['depth_avg', 'depth_var', 'depth_mid', 'depth_max', 
                            'aleatoric_avg', 'aleatoric_var', 'aleatoric_mid', 'aleatoric_max', 
                            'translation_avg', 'translation_max', 'translation_min', 
                            'rotation_avg', 'rotation_max', 'rotation_min',
                            'landmark_cnt_avg']]
        train_output_cols = ['delta1_diff', 'delta1_lit_diff', 'rmse_diff', 'absrel_diff', 'loss_diff_curr', 'loss_diff_rate']
        if ignore_input_cols is not None:
            train_input_cols = [ x for x in train_input_cols if x not in ignore_input_cols ]
        return train_input_cols, train_output_cols

    def _get_input_output(self, online_stats_df, biased=True, normalization={'method': 'none'}, output_cols=None, input_cols=None):
        """
        biased: if True, add 1s to the input matrix to add a global bias to prediction
        """
        train_input_cols, train_output_cols = self._get_input_output_cols(ignore_input_cols=[])
        if output_cols is not None:
            assert len(output_cols) > 0
            assert all([col in train_output_cols for col in output_cols])
            train_output_cols = output_cols
        if input_cols is not None:
            assert len(input_cols) > 0
            assert all([col in train_input_cols for col in input_cols])
            train_input_cols = input_cols
        train_input = online_stats_df.loc[:, train_input_cols].to_numpy()
        train_output = online_stats_df.loc[:, train_output_cols].to_numpy()
        train_input, norm_meta = self._normalize_input(train_input, normalization=normalization)
        if biased:
            train_input = np.hstack((np.ones((train_input.shape[0], 1)), train_input))
        return train_input, train_output, norm_meta
    
    def _get_config_idxs(self, seq_names, model_groups, model_idx, pattern_idx, rows=None):
        # matching decision dnn statistics of each experiment to the corresponding config idxs
        # model group: 0 (model 1-10) - 4 (model 41 - 50)
        # model idx: 0 - 9
        # pattern idx: 0 - 5
        if rows is None:
            rows = [None]*len(seq_names)
        seq_name_to_start_gidx = {
            'scene0101_00': 6, 'scene0101_05': 8,
            'scene0673_00': 10, 'scene0673_05': 12,
            'scene0451_01': 14, 'scene0451_05': 16,
            'scene0092_00': 18, 'scene0092_04': 20,
            'scene0141_00': 22, 'scene0362_00': 24,
            'scene0181_00': 26, 'scene0320_00': 28,
            'scene0286_00': 30, "scene0142_00": 32,
            "scene0166_00": 34, "scene0571_01": 36,
            "scene0544_00": 38
        }
        model_per_group = 6 # 60
        cfgs, cfg_rows = [], []
        for seq_name, gidx, midx, pidx, row in zip(seq_names, model_groups, model_idx, pattern_idx, rows):
            _gidx = seq_name_to_start_gidx[seq_name] + gidx
            if pidx is None:
                _cfgs = [(model_per_group*_gidx)+(6*midx)+1+x for x in range(6)]
            else:
                _cfgs = [(model_per_group*_gidx)+(6*midx)+1+pidx]
            cfgs += _cfgs
            cfg_rows += [row]*len(_cfgs)
        return cfgs, cfg_rows
    
    ############################################################################################################
    # Public methods
    ############################################################################################################
    def normalize(self, x_feat, norm_meta):
        return self._normalize_input(x_feat, norm_meta)
    
    def stats_to_feat(self, x_stats, biased=True, norm_meta=None, train_input_cols=None):
        # input lr-logger stats dict (csv row format)
        # output: input features for LR-model
        # train_input_cols, train_output_cols = self._get_input_output_cols(ignore_input_cols=ignore_input_cols)
        stats_dict = {k: [v] for k, v in x_stats.items()}
        online_stats_df = pd.DataFrame(stats_dict)
        input_feat = online_stats_df.loc[:, train_input_cols].to_numpy()
        if norm_meta is not None:
            input_feat, _ = self._normalize_input(input_feat, normalization=norm_meta, verbose=False)
        if biased:
            input_feat = np.hstack((np.ones((input_feat.shape[0], 1)), input_feat))
        return input_feat

    def get_online_stats(
            self, 
            seq_names,                    # list of seq_name of stats to be loaded
            model_groups,                 # list of model group of stats to be loaded
            model_idx,                    # list of model idx of stats to be loaded
            pattern_idx=None,             # list of pattern idx of stats to be loaded
            output_cols=None,             # list of output cols to be loaded
            rows=None,                    # list of rows to be loaded
            biased=True,                  # if True, return input_dim+1 features/input data
            normalize={'method': 'none'}, # normalize data, options: 'none', 'standard', 'minmax'
            input_cols=None,              # input cols name (order)
            val_split=None                # split data into train and val, if none, return only train
        ):
        # get all config idxs
        cfg_idxs, cfg_rows = self._get_config_idxs(seq_names, model_groups, model_idx, pattern_idx, rows)
        # load input and output data
        online_stats_df = self._load_online_stats(cfg_idxs, cfg_rows)
        x_feats, y_feats, norm_meta = self._get_input_output(online_stats_df, biased=biased, normalization=normalize, output_cols=output_cols, input_cols=input_cols)
        if val_split is None:
            return x_feats, y_feats, norm_meta
        else:
            # determine train and val split
            train_ratio = 1 - val_split
            n_train = int(train_ratio * x_feats.shape[0])
            # shuffle data
            n_x_cols = x_feats.shape[1]
            xy_stack = np.hstack((x_feats, y_feats))
            np.random.shuffle(xy_stack)
            # split data into train and val
            x_train, y_train = xy_stack[:n_train, :n_x_cols], xy_stack[:n_train, n_x_cols:]
            x_val, y_val = xy_stack[n_train:, :n_x_cols], xy_stack[n_train:, n_x_cols:]
            return x_train, y_train, x_val, y_val, norm_meta