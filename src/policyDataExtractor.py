import os
import sys
import time
import csv
import numpy as np 
from collections import namedtuple
import torch
import torch.backends.cudnn as cudnn
import torch.optim

import matplotlib.pyplot as plt 
import pandas as pd
import yaml 
import math
import matplotlib.style as mplstyle
import copy
import pickle
mplstyle.use('fast')
from scipy.spatial.transform import Rotation
from statistics import median

import nets.metrics

default_lrlogger_config = {
    "update_pattern": [0]*10, # never update after training
    "mode":['delta1', 'delta1_lit', 'rmse', 'absrel'], # recorded gain on metrics
    "display_score": 'delta1', # training pre/post score to show
    #################### Other settings ########################
    "record_per_img_all_outputs": False, # store metadata for each image pair
    "record_all_trained_combined_stats": False, # store all trained online stats (e.g. mean of all predicted depth, etc.)
    "record_all_trained_per_img_stats": False, # store trained image stats, but per image
    "all_trained_sample_count": 3, # all trained image count per seq idx
}

class policyDataExtractor:

    def __init__(self, train_dir, stats_collection_on=False, **kwargs):
        self.stats_collection_on = stats_collection_on
        self._update_configs(kwargs)

        ##########################################################################################
        # Recorded metadata
        ##########################################################################################
        self.per_img_metadata = {} # the improvement in accuracy/rmse/absrel for each image pair

        self.recorded_stats = {} # temp record of snapshot
        self.online_stats = {} # record of all online statistics
        self.online_stats_all_trained = {} # record of all online statistics for all trained images
        self.online_all_trained_per_img_stats = {}

    def record_snapshot(self, pred, metadata, models, buffer, acq_meta, all_outputs):
        # acq_meta may not be used as it should always be train on all when running data collection
        if not self.stats_collection_on:
            return
        ##########################################################################################
        # snapshot state before training
        ##########################################################################################
        recorded_scores = self._evaluate_results(pred_depth=pred, metadata=metadata)
        recorded_snapshot = self._record_snapshot(models=models, buffer=buffer, acq_meta=acq_meta)
        self.recorded_stats = {
            "scores": recorded_scores,
            "snapshot": recorded_snapshot,
            "all_outputs": all_outputs
        }
        return
    
    def update_decision_and_record_status(self, seq_idx, pred_post, metadata, models, buffer, acq_meta, trained_img_idxs, all_outputs_post, all_losses_prev, all_losses_post, curr_loss_prev, curr_loss_post):
        if not self.stats_collection_on:
            return
        ##########################################################################################
        # Record the training stats, and update DNNs based on pattern
        ##########################################################################################
        post_scores = self._evaluate_results(pred_depth=pred_post, metadata=metadata)
        recorded_scores, recorded_snapshot = self.recorded_stats['scores'], self.recorded_stats['snapshot']
        recorded_eval_score, post_eval_score = recorded_scores[self.display_score], post_scores[self.display_score]
        print(f"[{self.display_score}] Pre-score: {recorded_eval_score:.4f} | Post-score: {post_eval_score:.4f}")

        if seq_idx >= 2:
            prev_loss = self._extract_stats_loss(all_losses_prev)
            post_loss = self._extract_stats_loss(all_losses_post)
            train_decision = self._update_decision(seq_idx, prev_loss, post_loss)
            self.per_img_metadata[seq_idx] = {
                "train": train_decision,
                "trained_img_idxs": trained_img_idxs,
                "prev_scores": recorded_scores,
                "post_scores": post_scores,
                "prev_loss": prev_loss, # loss before training, include replay
                "post_loss": post_loss, # loss after training, include replay
                "prev_loss_curr": curr_loss_prev, # loss before training, only current image
                "post_loss_curr": curr_loss_post, # loss after training, only current image
            }
            if self.record_per_img_all_outputs:
                self.per_img_metadata[seq_idx]["prev_outputs"] = [x.cpu().detach() for x in self.recorded_stats["all_outputs"]],
                self.per_img_metadata[seq_idx]["post_outputs"] = [x.cpu().detach() for x in all_outputs_post]

        if not train_decision:
            print('[Decision DNN online statistics logger] Reverting to previous state')
            models, buffer, acq_meta = self._retrieve_snapshot(recorded_snapshot, models, buffer, acq_meta)
        else:
            print('[Decision DNN online statistics logger] Update DNNs based on pattern')
        return models, buffer, acq_meta
    
    def record_online_stats(self, seq_idx, all_outputs, epistemic_unc, running_buffer):
        if not self.stats_collection_on:
            return
        if running_buffer[seq_idx]['real_idx'] < 2: # first two images of a sequence
            return
        # avg, var, mid, max of depth
        depth_stats = self._extract_stats_depth(all_outputs)
        # avg, var, mid, max of aleatoric uncertainty 
        aleatoric_stats = self._extract_stats_aleatoric(all_outputs)
        # avg, var, mid, max of epistemic uncertainty
        epistemic_stats = self._extract_stats_epistemic(epistemic_unc)
        # num of landmarks (from feature tracking)
        landmarks_stats = self._extract_stats_landmarks(seq_idx, running_buffer)
        # avg, max, min of translation of running window
        translation_stats = self._extract_stats_translation(seq_idx, running_buffer)
        # avg, max, min of rotation of running window
        rotation_stats = self._extract_stats_rotation(seq_idx, running_buffer)

        self.online_stats[seq_idx] = {
            "depth": depth_stats,
            "aleatoric": aleatoric_stats,
            "epistemic": epistemic_stats,
            "landmarks": landmarks_stats,
            "translation": translation_stats,
            "rotation": rotation_stats
        }

        return self.online_stats[seq_idx]
    
    def record_online_stats_all_trained(self, seq_idx, queries_dict, all_outputs, epistemic_unc, sample_buffer):
        # record stats for all trained images (including selected from replay buffer and the original ones)
        # queries: the queries of the data in the replay buffer (align with batch order of all_outputs)
        def remove_invalid_index(queries, all_outputs):
            if all([x >= 2 for x in queries]):
                return queries, all_outputs
            valid_queries = []
            valid_outputs = []
            valid_outputs_batched = []
            for i, q in enumerate(queries):
                if q >= 2:
                    valid_queries.append(q)
                    valid_outputs.append([x[i].unsqueeze(0) for x in all_outputs])
            for midx in range(len(all_outputs)):
                valid_outputs_batched.append(torch.cat([x[midx] for x in valid_outputs], dim=0))
            return valid_queries, valid_outputs_batched
            
        # if not self.stats_collection_on or seq_idx < 2:
        if not self.stats_collection_on or sample_buffer[seq_idx]['real_idx'] < 2:
            return
        _queries, _all_outputs = remove_invalid_index(queries_dict.keys(), all_outputs)
        if self.record_all_trained_combined_stats:
            assert seq_idx not in self.online_stats_all_trained, "Multiple mini-batch training not supported yet."
            # avg, var, mid, max of depth
            depth_stats = self._extract_stats_depth(_all_outputs)
            # avg, var, mid, max of aleatoric uncertainty 
            aleatoric_stats = self._extract_stats_aleatoric(_all_outputs)
            # avg, var, mid, max of epistemic uncertainty
            epistemic_stats = self._extract_stats_epistemic(epistemic_unc)
            # num of landmarks (from feature tracking)
            landmarks_stats = self._extract_stats_landmarks_all(_queries, sample_buffer)
            # avg, max, min of translation of running window
            translation_stats = self._extract_stats_translation_all(_queries, sample_buffer)
            # avg, max, min of rotation of running window
            rotation_stats = self._extract_stats_rotation_all(_queries, sample_buffer)
            self.online_stats_all_trained[seq_idx] = {
                "depth": depth_stats,
                "aleatoric": aleatoric_stats,
                "epistemic": epistemic_stats,
                "landmarks": landmarks_stats,
                "translation": translation_stats,
                "rotation": rotation_stats
            }
        if self.record_all_trained_per_img_stats:
            assert seq_idx not in self.online_all_trained_per_img_stats, "Multiple mini-batch training not supported yet."
            self.online_all_trained_per_img_stats[seq_idx] = {} # dict seq_idx -> q -> stats
            for i, q in enumerate(_queries):
                # avg, var, mid, max of depth: no access, but record for now
                depth_stats = self._extract_stats_depth([x[i].unsqueeze(0) for x in _all_outputs])
                # avg, var, mid, max of aleatoric uncertainty: available as we don't change aleatoric uncertainty
                aleatoric_stats = self._extract_stats_aleatoric([x[i].unsqueeze(0) for x in _all_outputs])
                # not recording epistemic uncertainty as not running inference before training
                # running_buffer_q = {idx: sample_buffer[idx] for idx in range(q-2, q+1)}
                # near_queries = self._search_near_view(query=q, buffer=sample_buffer, window_max_size=2)
                near_queries = queries_dict[q]
                running_buffer_q = {idx: sample_buffer[idx] for idx in near_queries}
                # num of landmarks (from feature tracking)
                landmarks_stats = self._extract_stats_landmarks(q, running_buffer_q)
                # avg, max, min of translation of running window
                translation_stats = self._extract_stats_translation(q, running_buffer_q)
                # avg, max, min of rotation of running window
                rotation_stats = self._extract_stats_rotation(q, running_buffer_q)
                self.online_all_trained_per_img_stats[seq_idx][i] = {
                    "query": q,
                    "depth": depth_stats,
                    "aleatoric": aleatoric_stats,
                    "landmarks": landmarks_stats,
                    "translation": translation_stats,
                    "rotation": rotation_stats
                }

        return
    
    def extract_policy_input(self, seq_idx, queries, all_outputs, epistemic_unc, sample_buffer, prev_loss):
        # Interface for extracting policy inputs, must run with debug mode off
        assert not self.stats_collection_on, "input extraction mode can only be used when debug is off"
        if sample_buffer[seq_idx]['real_idx'] < 2:
            return None, None
        online_all_trained_per_img_stats, current_online_stats = self._extract_online_stats_all_trained(seq_idx, queries, all_outputs, sample_buffer)
        # transform the stats into inputs for policy
        stats_dict = self._get_online_stats_as_csv_dict(mode="online_all_trained_per_img_stats", stats=online_all_trained_per_img_stats)
        # epistemic only relate to current
        epistemic_stats = self._extract_stats_epistemic(epistemic_unc)
        epistemic_stats_flatten = {
            "epistemic_avg": epistemic_stats["avg"],
            "epistemic_var": epistemic_stats["var"],
            "epistemic_mid": epistemic_stats["mid"],
            "epistemic_max": epistemic_stats["max"]
        }
        # add current depth info (when doing policy inference, should not use depth info of replay buffer)
        curr_depth_stats_flatten = {
            "depth_avg": current_online_stats["depth"]["avg"],
            "depth_var": current_online_stats["depth"]["var"],
            "depth_mid": current_online_stats["depth"]["mid"],
            "depth_max": current_online_stats["depth"]["max"]
        }
        # use current loss info only (not use replayed loss to avoid compute cost on depth inference)
        input_stats = {**stats_dict, **epistemic_stats_flatten, **curr_depth_stats_flatten, "prev_loss_curr": prev_loss}
        return input_stats, current_online_stats

    def write_metadata(self, output_dir):
        if not self.stats_collection_on:
            return
        ##########################################################################################
        # Writeout metadata
        ##########################################################################################
        # csv out online stats
        self._write_online_stats_csv(os.path.join(output_dir, 'lrlogger_online_stats.csv'))
        # csv out online stats
        if self.record_all_trained_combined_stats:
            self._write_online_stats_all_trained_csv(os.path.join(output_dir, 'lrlogger_online_stats_all_trained.csv'))
        if self.record_all_trained_per_img_stats:
            self._write_online_all_trained_per_img_stats_csv(os.path.join(output_dir, 'lrlogger_online_all_trained_per_img_stats.csv'))
        # pickle output per_img_metadata
        if self.record_per_img_all_outputs:
            with open(os.path.join(output_dir, 'lrlogger_per_img_metadata.pkl'), 'wb') as fo:
                pickle.dump(self.per_img_metadata, fo)
        return
    
    ##########################################################################################
    # Private utils
    ##########################################################################################
    def _extract_online_stats_all_trained(self, seq_idx, queries_dict, all_outputs, sample_buffer):
        # record stats for all trained images (including selected from replay buffer and the original ones)
        # queries_dixt: the query -> near views of the data in the replay buffer (align with batch order of all_outputs)
        def remove_invalid_index(queries, all_outputs):
            if all([x >= 2 for x in queries]):
                return queries, all_outputs
            valid_queries = []
            valid_outputs = []
            valid_outputs_batched = []
            for i, q in enumerate(queries):
                if q >= 2:
                    valid_queries.append(q)
                    valid_outputs.append([x[i].unsqueeze(0) for x in all_outputs])
            for midx in range(len(all_outputs)):
                valid_outputs_batched.append(torch.cat([x[midx] for x in valid_outputs], dim=0))
            return valid_queries, valid_outputs_batched

        _queries, _all_outputs = remove_invalid_index(queries_dict.keys(), all_outputs)
        online_all_trained_per_img_stats = {} # dict seq_idx -> q -> stats
        current_online_stats = None
        for i, q in enumerate(_queries):
            # avg, var, mid, max of depth: no access, but record for now
            depth_stats = self._extract_stats_depth([x[i].unsqueeze(0) for x in _all_outputs])
            # avg, var, mid, max of aleatoric uncertainty: available as we don't change aleatoric uncertainty
            aleatoric_stats = self._extract_stats_aleatoric([x[i].unsqueeze(0) for x in _all_outputs])
            near_queries = queries_dict[q]
            running_buffer_q = {idx: sample_buffer[idx] for idx in near_queries}
            # num of landmarks (from feature tracking)
            if q == seq_idx:
                landmarks_stats = self._extract_stats_landmarks(q, running_buffer_q)
            else:
                landmarks_stats = {
                    "landmark_cnt_avg": sample_buffer[q]["online_stats"]["landmarks"]["landmark_cnt_avg"],
                }
            # avg, max, min of translation of running window
            translation_stats = self._extract_stats_translation(q, running_buffer_q)
            # avg, max, min of rotation of running window
            rotation_stats = self._extract_stats_rotation(q, running_buffer_q)
            online_all_trained_per_img_stats[i] = {
                "query": q,
                "depth": depth_stats,
                "aleatoric": aleatoric_stats,
                "landmarks": landmarks_stats,
                "translation": translation_stats,
                "rotation": rotation_stats
            }
            if q == seq_idx:
                current_online_stats = copy.deepcopy(online_all_trained_per_img_stats[i])
        return online_all_trained_per_img_stats, current_online_stats
    
    def _update_configs(self, config):
        self.config = default_lrlogger_config
        self.config.update(config)
        # extract settings
        assert set(self.config["mode"]).issubset(set(['delta1', 'delta1_lit', 'rmse', 'absrel']))
        self.update_method = "pattern"
        self.update_pattern = self.config["update_pattern"]
        self.mode = self.config["mode"]
        self.display_score = self.config["display_score"]

        self.record_per_img_all_outputs = self.config["record_per_img_all_outputs"]
        self.record_all_trained_combined_stats = self.config["record_all_trained_combined_stats"]
        self.record_all_trained_per_img_stats = self.config["record_all_trained_per_img_stats" ]
        self.all_trained_sample_count = self.config["all_trained_sample_count"]
        return

    def _evaluate_results(self, pred_depth, metadata):
        # assert set(modes).issubset(set(['delta1', 'delta1_lit', 'rmse', 'absrel']))
        result = nets.metrics.Result()
        result.evaluate(pred_depth, metadata['depth'], loss=0)
        scores = {}
        for mode in self.mode:
            if mode in ['delta1', 'delta1_lit', 'rmse', 'absrel']:
                scores[mode] = getattr(result, mode)
        return scores
    
    def _record_snapshot(self, models, buffer, acq_meta):
        record = {}
        if isinstance(models[0], tuple):
            # record['models'] = [ [ copy.deepcopy(model.state_dict()) for model in models_t ] for models_t in models ]
            record['models'] = [ [ copy.deepcopy(models_t[0].state_dict()) ] for models_t in models ] # only record depth
        else:
            record['models'] = [ copy.deepcopy(model.state_dict()) for model in models ]
        record['buffer'] = copy.deepcopy(buffer)
        record['acq_meta'] = copy.deepcopy(acq_meta)
        return record

    def _update_decision(self, seq_idx, prev_loss, post_loss):
        # always collect training statistics with a given fixed pattern
        if self.update_method == "pattern":
            if self.update_pattern[int(seq_idx%len(self.update_pattern))] == 0:
                return False
            return True
        else:
            raise ValueError(f"Unknown update method: {self.update_method}. Supported methods: pattern, oracle_loss_rate.")

    def _retrieve_snapshot(self, record, models, buffer, acq_meta):
        for i in range(len(models)):
            if isinstance(models[i], tuple):
                models[i][0].load_state_dict(record['models'][i][0]) # only restore depth
            else:
                models[i].load_state_dict(record['models'][i])
        buffer = record['buffer']
        # acq: when decide to train, will update the following acq metadata:
        #   - num_train_decisions
        #   - update_f2_variables:
        #       - DfM: self.last_trained_rotation, self.last_trained_translation, self.last_trained_depth_pred
        #       - cosine_distance: self.features_buffer
        # other acq updates are not related to decision of training = True
        acq_meta.num_train_decisions = record['acq_meta'].num_train_decisions
        if 'last_trained_rotation' in dir(acq_meta):
            acq_meta.last_trained_rotation = record['acq_meta'].last_trained_rotation
            acq_meta.last_trained_translation = record['acq_meta'].last_trained_translation
            acq_meta.last_trained_depth_pred = record['acq_meta'].last_trained_depth_pred
        if 'features_buffer' in dir(acq_meta):
            acq_meta.features_buffer = record['acq_meta'].features_buffer
        return models, buffer, acq_meta

    def _extract_stats_depth(self, all_outputs):
        depths = [x[:,0,:,:].unsqueeze(1) for x in all_outputs]
        all_depth = torch.cat(depths, dim=0).cpu().detach()
        return {
            "avg": all_depth.mean().item(),
            "var": all_depth.var().item(),
            "mid": all_depth.median().item(),
            "max": all_depth.max().item()
        }
    
    def _extract_stats_aleatoric(self, all_outputs):
        aleatorics = [x[:,1,:,:].unsqueeze(1) for x in all_outputs]
        all_aleatorics = torch.cat(aleatorics, dim=0).cpu().detach()
        return {
            "avg": all_aleatorics.mean().item(),
            "var": all_aleatorics.var().item(),
            "mid": all_aleatorics.median().item(),
            "max": all_aleatorics.max().item()
        }
    
    def _extract_stats_epistemic(self, epistemic_unc):
        epistemic_unc_t = epistemic_unc.cpu().detach()
        return {
            "avg": epistemic_unc_t.mean().item(),
            "var": epistemic_unc_t.var().item(),
            "mid": epistemic_unc_t.median().item(),
            "max": epistemic_unc_t.max().item()
        }
    
    def _extract_stats_translation(self, seq_idx, running_buffer):
        # assert seq_idx in running_buffer
        # assert seq_idx-1 in running_buffer
        # assert seq_idx-2 in running_buffer
        # trans = [ running_buffer[seq_idx-i]['trans'] for i in range(0, 3) ]
        trans = [ v['trans'] for k, v in running_buffer.items() ]
        trans1 = torch.linalg.norm(trans[0] - trans[1]).cpu().detach().item()
        trans2 = torch.linalg.norm(trans[0] - trans[2]).cpu().detach().item()
        return {
            "avg": (trans1+trans2)/2,
            "max": max(trans1, trans2),
            "min": min(trans1, trans2)
        }
    
    def _extract_stats_rotation(self, seq_idx, running_buffer):
        def get_rotation(R1, R2):
            r =  Rotation.from_matrix(torch.matmul(torch.transpose(R1, 0, 1), R2))
            angles = r.as_euler("zyx", degrees=True)
            # current use sum of abs angles of each direction as metric
            angle_sum = torch.sum(torch.abs(torch.tensor(angles)))
            return angle_sum
        # assert seq_idx in running_buffer
        # assert seq_idx-1 in running_buffer
        # assert seq_idx-2 in running_buffer
        # rot = [ running_buffer[seq_idx-i]['rot'].cpu().detach() for i in range(0, 3) ]
        rot = [ v['rot'].cpu().detach() for k, v in running_buffer.items() ]
        rot1 = get_rotation(rot[0], rot[1]).cpu().detach().item()
        rot2 = get_rotation(rot[0], rot[2]).cpu().detach().item()
        return {
            "avg": (rot1+rot2)/2,
            "max": max(rot1, rot2),
            "min": min(rot1, rot2)
        }
    
    def _extract_stats_landmarks(self, seq_idx, running_buffer):
        # per image feature cnt: for the current image
        per_img_feat_cnt_q = [ v['landmark_count'].item() for k, v in running_buffer.items() ]
        return {
            "landmark_cnt_avg": sum(per_img_feat_cnt_q)/len(per_img_feat_cnt_q)
        }

    def _extract_stats_loss(self, all_losses):
        _all_losses = all_losses
        return {
            "avg": sum(_all_losses)/len(_all_losses),
            "var": sum([x**2 for x in _all_losses])/len(_all_losses) - (sum(_all_losses)/len(_all_losses))**2,
            "mid": median(_all_losses),
            "max": max(_all_losses)
        }
    
    def _extract_stats_translation_all(self, queries, sample_buffer):
        trans_stats = None
        b = 0
        for q_idx in queries:
            if q_idx < 2:
                continue
            near_queries = queries[q_idx]
            q_running_buffer = {idx: sample_buffer[idx] for idx in near_queries}
            q_stats = self._extract_stats_translation(q_idx, q_running_buffer)
            if trans_stats is None:
                trans_stats = q_stats
            else:
                trans_stats = {
                    "avg": trans_stats["avg"]+q_stats["avg"],
                    "max": max(trans_stats["max"], q_stats["max"]),
                    "min": min(trans_stats["min"], q_stats["min"])
                }
            b += 1
        trans_stats["avg"] = trans_stats["avg"]/b
        return trans_stats
    
    def _extract_stats_rotation_all(self, queries, sample_buffer):
        rot_stats = None
        b = 0
        for q_idx in queries:
            if q_idx < 2:
                continue
            near_queries = queries[q_idx]
            q_running_buffer = {idx: sample_buffer[idx] for idx in near_queries}
            q_stats = self._extract_stats_rotation(q_idx, q_running_buffer)
            if rot_stats is None:
                rot_stats = q_stats
            else:
                rot_stats = {
                    "avg": rot_stats["avg"]+q_stats["avg"],
                    "max": max(rot_stats["max"], q_stats["max"]),
                    "min": min(rot_stats["min"], q_stats["min"])
                }
            b += 1
        rot_stats["avg"] = rot_stats["avg"]/b
        return rot_stats
    
    def _extract_stats_landmarks_all(self, queries, sample_buffer):
        landmarks_stats = None
        b = 0
        for q_idx in queries:
            if q_idx < 2:
                continue
            near_queries = queries[q_idx]
            q_running_buffer = {idx: sample_buffer[idx] for idx in near_queries}
            q_stats = self._extract_stats_landmarks(q_idx, q_running_buffer)
            if landmarks_stats is None:
                landmarks_stats = q_stats
            else:
                landmarks_stats = {
                    "landmark_cnt_avg": landmarks_stats["landmark_cnt_avg"]+q_stats["landmark_cnt_avg"]
                }
            b += 1
        landmarks_stats["landmark_cnt_avg"] = landmarks_stats["landmark_cnt_avg"]/b
        return landmarks_stats
    
    def _get_online_stats_as_csv_dict(self, mode, stats):
        # meta_stats: all stats data related to this seq_idx
        # fieldnames: the fieldnames of the csv
        assert mode in ["online_stats", "online_stats_all_trained", "online_all_trained_per_img_stats", "per_img_metadata"]
        if mode in ["online_stats", "online_stats_all_trained"]:
            return {
                'depth_avg'       : stats["depth"]["avg"], 
                'depth_var'       : stats["depth"]["var"], 
                'depth_mid'       : stats["depth"]["mid"], 
                'depth_max'       : stats["depth"]["max"], 
                'aleatoric_avg'   : stats["aleatoric"]["avg"], 
                'aleatoric_var'   : stats["aleatoric"]["var"], 
                'aleatoric_mid'   : stats["aleatoric"]["mid"], 
                'aleatoric_max'   : stats["aleatoric"]["max"], 
                'epistemic_avg'   : stats["epistemic"]["avg"], 
                'epistemic_var'   : stats["epistemic"]["var"], 
                'epistemic_mid'   : stats["epistemic"]["mid"], 
                'epistemic_max'   : stats["epistemic"]["max"], 
                'landmark_cnt_avg': stats["landmarks"]["landmark_cnt_avg"], 
                'translation_avg' : stats["translation"]["avg"],
                'translation_max' : stats["translation"]["max"],
                'translation_min' : stats["translation"]["min"], 
                'rotation_avg'    : stats["rotation"]["avg"], 
                'rotation_max'    : stats["rotation"]["max"], 
                'rotation_min'    : stats["rotation"]["min"],
            }
        elif mode == "online_all_trained_per_img_stats":
            row_to_write = {}
            for i in range(self.all_trained_sample_count):
                if i not in stats:
                    row_to_write[f'query_{i}']            = 0
                    row_to_write[f'depth_avg_{i}']        = 0
                    row_to_write[f'depth_var_{i}']        = 0
                    row_to_write[f'depth_mid_{i}']        = 0
                    row_to_write[f'depth_max_{i}']        = 0
                    row_to_write[f'aleatoric_avg_{i}']    = 0
                    row_to_write[f'aleatoric_var_{i}']    = 0
                    row_to_write[f'aleatoric_mid_{i}']    = 0
                    row_to_write[f'aleatoric_max_{i}']    = 0
                    row_to_write[f'landmark_cnt_avg_{i}'] = 0
                    row_to_write[f'translation_avg_{i}']  = 0
                    row_to_write[f'translation_max_{i}']  = 0
                    row_to_write[f'translation_min_{i}']  = 0
                    row_to_write[f'rotation_avg_{i}']     = 0
                    row_to_write[f'rotation_max_{i}']     = 0
                    row_to_write[f'rotation_min_{i}']     = 0
                else:
                    row_to_write[f'query_{i}']            = stats[i]["query"]
                    row_to_write[f'depth_avg_{i}']        = stats[i]["depth"]["avg"]
                    row_to_write[f'depth_var_{i}']        = stats[i]["depth"]["var"]
                    row_to_write[f'depth_mid_{i}']        = stats[i]["depth"]["mid"]
                    row_to_write[f'depth_max_{i}']        = stats[i]["depth"]["max"]
                    row_to_write[f'aleatoric_avg_{i}']    = stats[i]["aleatoric"]["avg"]
                    row_to_write[f'aleatoric_var_{i}']    = stats[i]["aleatoric"]["var"]
                    row_to_write[f'aleatoric_mid_{i}']    = stats[i]["aleatoric"]["mid"]
                    row_to_write[f'aleatoric_max_{i}']    = stats[i]["aleatoric"]["max"]
                    row_to_write[f'landmark_cnt_avg_{i}'] = stats[i]["landmarks"]["landmark_cnt_avg"]
                    row_to_write[f'translation_avg_{i}']  = stats[i]["translation"]["avg"]
                    row_to_write[f'translation_max_{i}']  = stats[i]["translation"]["max"]
                    row_to_write[f'translation_min_{i}']  = stats[i]["translation"]["min"]
                    row_to_write[f'rotation_avg_{i}']     = stats[i]["rotation"]["avg"]
                    row_to_write[f'rotation_max_{i}']     = stats[i]["rotation"]["max"]
                    row_to_write[f'rotation_min_{i}']     = stats[i]["rotation"]["min"]
            return row_to_write
        elif mode == "per_img_metadata":
            return {
                'prev_loss_avg'  : stats["prev_loss"]["avg"],
                'prev_loss_var'  : stats["prev_loss"]["var"],
                'prev_loss_mid'  : stats["prev_loss"]["mid"],
                'prev_loss_max'  : stats["prev_loss"]["max"],
                'post_loss_avg'  : stats["post_loss"]["avg"],
                'post_loss_var'  : stats["post_loss"]["var"],
                'post_loss_mid'  : stats["post_loss"]["mid"],
                'post_loss_max'  : stats["post_loss"]["max"],
                'prev_loss_curr' : stats["prev_loss_curr"],
                'post_loss_curr' : stats["post_loss_curr"],
                'delta1_diff'    : stats["post_scores"]["delta1"]     - stats["prev_scores"]["delta1"],
                'delta1_lit_diff': stats["post_scores"]["delta1_lit"] - stats["prev_scores"]["delta1_lit"],
                'rmse_diff'      : stats["post_scores"]["rmse"]       - stats["prev_scores"]["rmse"],
                'absrel_diff'    : stats["post_scores"]["absrel"]     - stats["prev_scores"]["absrel"],
                'loss_diff'      : stats["post_loss"]["avg"]          - stats["prev_loss"]["avg"],
                'loss_diff_curr' : stats["post_loss_curr"]            - stats["prev_loss_curr"],
            }
        else:
            raise NotImplementedError


    def _write_online_stats_csv(self, csv_name):
        fieldnames = [
            'seq_idx', 
            'depth_avg', 'depth_var', 'depth_mid', 'depth_max', 
            'aleatoric_avg', 'aleatoric_var', 'aleatoric_mid', 'aleatoric_max', 
            'epistemic_avg', 'epistemic_var', 'epistemic_mid', 'epistemic_max', 
            'landmark_cnt_avg',
            'translation_avg', 'translation_max', 'translation_min', 
            'rotation_avg', 'rotation_max', 'rotation_min',
            'prev_loss_avg', 'prev_loss_var', 'prev_loss_mid', 'prev_loss_max', 'prev_loss_curr',
            'post_loss_avg', 'post_loss_var', 'post_loss_mid', 'post_loss_max', 'post_loss_curr',
            'delta1_diff', 'delta1_lit_diff', 'rmse_diff', 'absrel_diff', 'loss_diff', 'loss_diff_curr'
            ]
        with open(csv_name, 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for seq_idx in self.online_stats:
                _online_stats = self._get_online_stats_as_csv_dict(mode="online_stats", stats=self.online_stats[seq_idx])
                _per_img_stats = self._get_online_stats_as_csv_dict(mode="per_img_metadata", stats=self.per_img_metadata[seq_idx])
                row_to_write = {"seq_idx": seq_idx, **_online_stats, **_per_img_stats}
                writer.writerow(row_to_write)
        return
    
    def _write_online_stats_all_trained_csv(self, csv_name):
        fieldnames = [
            'seq_idx', 
            'depth_avg', 'depth_var', 'depth_mid', 'depth_max', 
            'aleatoric_avg', 'aleatoric_var', 'aleatoric_mid', 'aleatoric_max', 
            'epistemic_avg', 'epistemic_var', 'epistemic_mid', 'epistemic_max', 
            'landmark_cnt_avg',
            'translation_avg', 'translation_max', 'translation_min', 
            'rotation_avg', 'rotation_max', 'rotation_min',
            'prev_loss_avg', 'prev_loss_var', 'prev_loss_mid', 'prev_loss_max', 'prev_loss_curr',
            'post_loss_avg', 'post_loss_var', 'post_loss_mid', 'post_loss_max', 'post_loss_curr',
            'delta1_diff', 'delta1_lit_diff', 'rmse_diff', 'absrel_diff', 'loss_diff', 'loss_diff_curr'
            ]
        with open(csv_name, 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for seq_idx in self.online_stats_all_trained:
                _online_stats = self._get_online_stats_as_csv_dict(mode="online_stats_all_trained", stats=self.online_stats_all_trained[seq_idx])
                _per_img_stats = self._get_online_stats_as_csv_dict(mode="per_img_metadata", stats=self.per_img_metadata[seq_idx])
                row_to_write = {"seq_idx": seq_idx, **_online_stats, **_per_img_stats}
                writer.writerow(row_to_write)
        return
    
    def _write_online_all_trained_per_img_stats_csv(self, csv_name):
        fieldnames = ['seq_idx']
        for i in range(self.all_trained_sample_count):
            fieldnames += f'query_{i} depth_avg_{i} depth_var_{i} depth_mid_{i} depth_max_{i} aleatoric_avg_{i} aleatoric_var_{i} aleatoric_mid_{i} aleatoric_max_{i} landmark_cnt_avg_{i} translation_avg_{i} translation_max_{i} translation_min_{i} rotation_avg_{i} rotation_max_{i} rotation_min_{i}'.split()
        fieldnames += ['epistemic_avg', 'epistemic_var', 'epistemic_mid', 'epistemic_max']
        fieldnames += ['prev_loss_avg', 'prev_loss_var', 'prev_loss_mid', 'prev_loss_max', 'prev_loss_curr',
                       'post_loss_avg', 'post_loss_var', 'post_loss_mid', 'post_loss_max', 'post_loss_curr',
                       'delta1_diff', 'delta1_lit_diff', 'rmse_diff', 'absrel_diff', 'loss_diff', 'loss_diff_curr']
        with open(csv_name, 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for seq_idx in self.online_all_trained_per_img_stats:
                _online_stats = self._get_online_stats_as_csv_dict(mode="online_all_trained_per_img_stats", stats=self.online_all_trained_per_img_stats[seq_idx])
                _epistemic_stats = self._get_online_stats_as_csv_dict(mode="online_stats", stats=self.online_stats[seq_idx])
                _epistemic_stats = {k: _epistemic_stats[k] for k in ['epistemic_avg', 'epistemic_var', 'epistemic_mid', 'epistemic_max'] }
                _per_img_stats = self._get_online_stats_as_csv_dict(mode="per_img_metadata", stats=self.per_img_metadata[seq_idx])
                row_to_write = {"seq_idx": seq_idx, **_online_stats, **_epistemic_stats, **_per_img_stats}
                writer.writerow(row_to_write)
        return