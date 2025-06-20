import random 
random.seed(0) # set seed for reproducibility 
import torch
from copy import deepcopy
import math
import numpy as np
import faiss
from collections import Counter, OrderedDict


# please refer to https://github.com/robot-learning-freiburg/CoDEPS/tree/main for the original buffer implementation
# this buffer uses the target buffer replay strategy from CoDEPS, 
# with modifications for interfacing with out pipline, and add additional support for the DecTrain decision making.
class CoDEPS_Buffer:

    def __init__(self, run_parameters, debug=False):
        self.run_parameters = run_parameters
        self.recorded_buffer = {} # all previous metadata related to selected samples
        self.recorded_samples = [] # all queries for selected samples
        self.running_buffer = {}
        self.sample_max_size = run_parameters['buffer_max_size'] # max active learning sample (M)
        self.window_max_size = run_parameters['buffer_window_size'] # max window size for self-supervised learning (N)
        self.shuffled_keys = None
        self.current_query = None
        # for quickly determine if one image should be removed from the buffer
        self.img_counter = Counter()
        # for quickly looking up what is the nearest view to a query
        self.nearest_view = {} # t -> [t_2nd_near, t_1st_near, t]

        self.device = run_parameters['device'][0]
        self.source_buffer = {}
        self.source_samples = [] # all queries for selected samples from source sequence
        self.source_buffer_offset = None # separating source buffer index from target buffer
        self.similarity_threshold = 0.95 # default in CoDEPS
        self.faiss_index = None
        self.distance_matrix = None
        self.distance_matrix_indices = None

        # for CoDEPS min motion requirement
        self.motion_filering = run_parameters['buffer_motion_filtering'] if 'buffer_motion_filtering' in run_parameters else False
        self.motion_buffer = []
        self.motion_buffer_size = run_parameters['buffer_motion_filtering_window'] if 'buffer_motion_filtering_window' in run_parameters else 300
        if self.motion_filering:
            print(f"[Buffer] Warning: CoDEPS motion filtering is activated with window size {self.motion_buffer_size}")

        # for debugging
        self.debug = debug

        return
    
    def DBG(self, msg):
        if self.debug:
            print(msg)

    def push_metadata(self, query, metadata, intrinsics):
        # push data to running buffer (implicitly implied in CoDEPS)
        # pushed metadata includes original metadata and its intrinsics
        pushed_meta = {"intrinsics": intrinsics}
        pushed_meta.update(metadata)
        self.running_buffer[query] = pushed_meta
        self.current_query = query
        if (len(self.running_buffer) > self.window_max_size+1):
            idx_to_remove = min(self.running_buffer.keys())
            self.pop_metadata(idx_to_remove)
        return

    def push_sample(self, **kwargs):
        # null function, buffer update independent of training decision
        return

    def update_buffer(self, **kwargs):
        # update buffer after training is done
        query, features, online_stats = kwargs['query'], kwargs['features'], kwargs['online_stats']
        # return push, pop info
        pushed_to_buffer = False
        pop_index = None
        # in Codeps, just push every sample, and if exceeds max size, remove by CoDEPS policy
        # features: (B, C, H, W)
        assert features.shape[0] == 1, "CoDEPS buffer only support batch size 1 on features"
        flattened_features = features.mean(-1).mean(-1).cpu().numpy()
        if self.faiss_index is None:
            # Cosine similarity
            self.faiss_index = faiss.IndexIDMap(
                    faiss.index_factory(flattened_features.shape[1], "Flat",
                                        faiss.METRIC_INNER_PRODUCT))
        faiss.normalize_L2(flattened_features)
        # Only add if sufficiently different to existing samples
        if self.faiss_index.ntotal == 0:
            similarity = [[0]]
        else:
            similarity, _ = self.faiss_index.search(flattened_features, 1)
        if similarity[0][0] < self.similarity_threshold:
            self.recorded_samples.append(query)
            running_buffer_cpu = {q: self._load_metadata_to(d, 'cpu') for q, d in self.running_buffer.items()}
            running_buffer_cpu[query].update({"online_stats": online_stats})
            # record nearest view to the query
            self.nearest_view[query] = [q for q in running_buffer_cpu]
            for q in running_buffer_cpu:
                # increment the counter for each image in the buffer, including query itself
                self.img_counter[q] += 1
                if q not in self.recorded_buffer:
                    self.recorded_buffer[q] = running_buffer_cpu[q]
            self.faiss_index.add_with_ids(flattened_features, np.array([query]))
            print(f"Added sample {query} to the target buffer | similarity {similarity[0][0]}")
            pushed_to_buffer = True

            # Handler for exceeding max target buffer size
            if len(self.recorded_samples) > self.sample_max_size:
                remove_index = self.remove_less_diverse_sample(query, flattened_features)
                print(f"Removed sample {remove_index} from the target buffer")
                pop_index = remove_index

        # update motion buffer
        self._motion_buffer_update(query)

        return pushed_to_buffer, pop_index

    def pop_metadata(self, query):
        # handle deleting data in running buffer, same as ours
        self.running_buffer.pop(query, None)
        return

    def remove_less_diverse_sample(self, query, flattened_features):
        # if exceeds max size, remove by CoDEPS policy (maximize diversity)
        # always remove from target buffer only
        if self.distance_matrix is None:
            features = self.faiss_index.index.reconstruct_n(0, self.faiss_index.ntotal)
            dist_mat, matching = self.faiss_index.search(features,
                                                            self.faiss_index.ntotal)
            for i in range(self.faiss_index.ntotal):
                dist_mat[i, :] = dist_mat[i, matching[i].argsort()]
            self.distance_matrix = dist_mat
            self.distance_matrix_indices = faiss.vector_to_array(
                self.faiss_index.id_map)
        else:
            # Only update the elements that actually change
            fill_up_index = np.argwhere(self.distance_matrix_indices < 0)[0, 0]
            a, b = self.faiss_index.search(flattened_features, self.faiss_index.ntotal)
            self.distance_matrix_indices[fill_up_index] = query
            sorter = np.argsort(b[0])
            sorter_idx = sorter[
                np.searchsorted(b[0], self.distance_matrix_indices, sorter=sorter)]
            a = a[:, sorter_idx][0]
            self.distance_matrix[fill_up_index, :] = self.distance_matrix[:,
                                                        fill_up_index] = a
        
        # Subtract self-similarity
        remove_index_tmp = np.argmax(
            self.distance_matrix.sum(0) - self.distance_matrix.diagonal())
        self.distance_matrix[:, remove_index_tmp] = self.distance_matrix[
                                                    remove_index_tmp, :] = -1
        remove_index = self.distance_matrix_indices[remove_index_tmp]
        self.distance_matrix_indices[remove_index_tmp] = -1
        self.faiss_index.remove_ids(np.array([remove_index]))

        # remove from buffer
        self.pop_sample(remove_index)
        return remove_index

    def pop_sample(self, sample):
        # remove sample from recorded sample list
        self.recorded_samples.remove(sample)
        # remove recorded data that only related to the removed sample
        deleted_queries = self.nearest_view[sample] # [t_2nd_near, t_1st_near, t]
        for q in deleted_queries:
            self.img_counter[q] -= 1
            if self.img_counter[q] == 0:
                del self.img_counter[q]
                self.recorded_buffer.pop(q, None)
        # remove nearest view to the query
        self.nearest_view.pop(sample, None)
        
        return

    def reset_running_buffer(self):
        # clear running buffer
        print("[Buffer] Warning: Reset running buffer!")
        self.running_buffer = {}
        return

    def gen_random_metadata(self, batch_size):
        # only return one batch data, composed of 1. current window 2. B-1 data random sample from target replay buffer
        target_batch_size = min(batch_size - 1, len(self.recorded_samples))
        # no target replay buffer
        if len(self.recorded_samples) == 0:
            # return only current image window
            shuffled_keys_batch = {self.current_query: list(self.running_buffer.keys())}
            collected_metadata = self._concatenate_minibatch(self.running_buffer, shuffled_keys_batch)
            yield shuffled_keys_batch, collected_metadata, self.running_buffer
            return
        # collect only samples for randomized data
        # for all metadata, include running buffer (current image window)
        sample_buffer = deepcopy(self.running_buffer)
        # for target data sampling, ignore current image
        self.shuffled_keys = self._get_shuffled_keys({x:None for x in self.recorded_samples})
        random_selected_target_keys = self.shuffled_keys[:target_batch_size]
        all_sampled_buffer = [ self._get_sample_buffer(sample) for sample in random_selected_target_keys]
        _sample_buffer = {k:self._load_metadata_to(v, self.device) for d in all_sampled_buffer for k, v in d.items()}
        sample_buffer.update(_sample_buffer)
        # 1-batch generation
        # min motion requirement of CoDEPS: only train on current image when motion is above 10% of the avg of previous motions
        if self._motion_check(self.current_query) and len(self.running_buffer) >= self.window_max_size+1:
            shuffled_keys_batch = random_selected_target_keys + [self.current_query]
        else:
            shuffled_keys_batch = random_selected_target_keys
        collected_metadata = self._concatenate_minibatch(sample_buffer, shuffled_keys_batch)

        # include near view in shuffled_keys_batch
        shuffled_keys_batch_dict = OrderedDict()
        for k in shuffled_keys_batch:
            if k == self.current_query:
                shuffled_keys_batch_dict[k] = list(self.running_buffer.keys())
            else:
                shuffled_keys_batch_dict[k] = self.nearest_view[k]
        yield shuffled_keys_batch_dict, collected_metadata, sample_buffer
    
    def _motion_buffer_update(self, query):
        if not self.motion_filering:
            # if not activate CoDEPS motion filtering, always assume accept training
            return True
        # if running buffer has not enough view, i.e. < self.window_max_size+1, return False
        if len(self.running_buffer) < self.window_max_size+1:
            self.DBG("Warning: query is less than window size")
            return False
        # calculate motion of query: avg norm translation of t->t-1, t->t-2, ... t->t-self.window_max_size 
        trans_curr = self.running_buffer[query]['trans']
        motions = [ torch.linalg.norm(self.running_buffer[query-i-1]['trans'] - trans_curr) for i in range(self.window_max_size) ]
        avg_motion = sum(motions)/len(motions)
        # if buffer is not full, add to motion buffer, and return True
        if len(self.motion_buffer) < self.motion_buffer_size:
            self.DBG(f"Warning: motion buffer {len(self.motion_buffer)}/{self.motion_buffer_size}")
            self.motion_buffer.append(avg_motion)
            return True
        # check if the motion of query is above 10% of the avg of previous motions
        # if yes, add to motion buffer, and return True
        # if no, return False
        else:
            assert len(self.motion_buffer) == self.motion_buffer_size, "Motion buffer size is not correct"
            motion_thresh = (sum(self.motion_buffer)/len(self.motion_buffer))*0.1
            if avg_motion > motion_thresh:
                self.motion_buffer.pop(0)
                self.motion_buffer.append(avg_motion)
                return True
            else:
                self.DBG("[Buffer] Current image motion too small, not append to motion buffer")
                return False
            
    def _motion_check(self, query):
        if not self.motion_filering:
            # if not activate CoDEPS motion filtering, always push current image to self-supervised training
            return True
        # if running buffer has not enough view, i.e. < self.window_max_size+1, return False
        if len(self.running_buffer) < self.window_max_size+1:
            return False
        # calculate motion of query: avg norm translation of t->t-1, t->t-2, ... t->t-self.window_max_size 
        trans_curr = self.running_buffer[query]['trans']
        motions = [ torch.linalg.norm(self.running_buffer[query-i-1]['trans'] - trans_curr) for i in range(self.window_max_size) ]
        avg_motion = sum(motions)/len(motions)
        # if buffer is not full, return True
        if len(self.motion_buffer) < self.motion_buffer_size:
            return True
        # check if the motion of query is above 10% of the avg of previous motions
        # if yes, add to motion buffer, and return True
        # if no, return False
        else:
            assert len(self.motion_buffer) == self.motion_buffer_size, "Motion buffer size is not correct"
            motion_thresh = (sum(self.motion_buffer)/len(self.motion_buffer))*0.1
            if avg_motion > motion_thresh:
                return True
            else:
                self.DBG("[Buffer] Current image motion too small, not select for training")
                return False

    def _get_sample_buffer(self, sample, buffer_type='target'):
        # collect [sample-window, sample]
        # near_view_query = self.__search_near_view(query=sample)

        if buffer_type == 'target':
            return {k: self.recorded_buffer[k] for k in self.nearest_view[sample]}
            # return {k:v for k, v in self.recorded_buffer.items() if k <= sample and k >= sample-self.window_max_size}
        else:
            return {k: self.source_buffer[k] for k in self.nearest_view[sample]}
            # return {k:v for k, v in self.source_buffer.items() if k <= sample and k >= sample-self.window_max_size}
    
    def _get_shuffled_keys(self, buffer):
        keys = list(buffer.keys())
        assert len(keys) >= 1, "Trying to shuffle empty buffer"
        random.shuffle(keys) # in-place shuffle
        return keys
    
    def _concatenate_minibatch(self, buffer, minibatch_idx):
        if len(minibatch_idx) == 0:
            return {}
        return { k: torch.cat([buffer[idx][k] for idx in minibatch_idx], 0) for k in list(buffer.values())[0] if k not in ['intrinsics', 'online_stats'] }

    def _get_minibatch_idx(self, buffer, batch_idx, shuffled_keys, batch_size):
        # case: buffer size is less than batch size (return full buffer as minibatch)
        if len(buffer) <= batch_size:
            return shuffled_keys
        # case: buffer is divisible by batch size 
        elif len(buffer) % batch_size == 0: 
            return shuffled_keys[batch_idx*batch_size:batch_idx*batch_size+batch_size] 
        else: # case: buffer is divisible by batch size  
            if (batch_idx*batch_size+batch_size) > len(buffer): # last partial minibatch
                return shuffled_keys[batch_idx*batch_size:]
            else: 
                return shuffled_keys[batch_idx*batch_size:batch_idx*batch_size+batch_size] 

    def _load_metadata_to(self, metadata, device):
        metadata_cpu = {k: v.to(device) for k, v in metadata.items() if k not in ["online_stats"]}
        if "online_stats" in metadata:
            metadata_cpu.update({"online_stats": metadata["online_stats"]})
        return metadata_cpu
            
    def __str__(self):
        out  = f"Target Buffer:\n"
        out += f"    Recorded: {list(self.recorded_buffer.keys())}\n"
        out += f"    Samples : {self.recorded_samples}\n"
        out += f"    Running : {list(self.running_buffer.keys())}\n"
        return out

    def __getitem__(self, query):
        if query in self.recorded_buffer:
            return self.recorded_buffer[query]
        elif query in self.source_buffer:
            return self.source_buffer[query]
        else:
            return None
        
    def __len__(self):
        return len(self.recorded_buffer) + len(self.source_buffer)

    def items(self):
        all_buffer = {**self.recorded_buffer, **self.source_buffer}
        return all_buffer.items()