import torch
import torch.nn as nn
from copy import deepcopy
from nets.camera import Pose
from libs.CoDEPS import ReconstructionLoss, EdgeAwareSmoothnessLoss, SSIMLoss
from libs.CoDEPS import CameraModel

class CoDEPS_DepthLoss(nn.Module):

    def __init__(self, img_width, img_height, recon_loss_weight, smth_loss_weight, scales=5, device='cuda:0'):
        super(CoDEPS_DepthLoss, self).__init__()
        ssim_loss = SSIMLoss()
        self.reconstruction_loss = ReconstructionLoss(
            img_width,
            img_height,
            ssim_loss, 
            scales, #cfg.depth.num_recon_scales = 5, from cfg/default_config_adapt.py
            device
        )
        self.smoothness_loss = EdgeAwareSmoothnessLoss()
        self.device = device
        self.recon_loss_weight = recon_loss_weight
        self.smth_loss_weight = smth_loss_weight

    def __stack_poses(self, poses):
        # stack all poses
        stacked_poses = torch.cat([pose.transformation_matrix().view(1, 4, 4) for pose in poses], dim=0)
        return stacked_poses
    
    def __search_near_view(self, query, buffer, window_max_size):
        # select window_max_size nearest view to the query (smaller than query)
        near_view_query = []
        search_query = query-1
        while(len(near_view_query) < window_max_size):
            if search_query in buffer:
                near_view_query.append(search_query)
            search_query -= 1
            if search_query < 0 and len(near_view_query) < window_max_size:
                print(f"[CoDEPS Loss] Warning: query {query} has only {len(near_view_query)} near view")
                break
        return near_view_query

    def __collect_batch_data(self, buffer, batch_queries, device, window_max_size=2):
        # output
        # images: Tuple[Tensor, Tensor, Tensor], order as [t, t-1, t-2], each with shape (B, 3, H, W)
        # poses : (Tensor, Tensor), order as camera motion [t -> t-1] and [t -> t-2], each is transformation matrix [[R t], [0, 1]] with shape (B, 4, 4)

        collected_batch_data = []
        collected_intrinsics = []
        ignored_queries = []
        for query in batch_queries:
            # near_queries = sorted([ k for k in buffer if query - k <= window_max_size and query > k], reverse=True)
            # near_queries = self.__search_near_view(query, buffer, window_max_size)
            near_queries = [x for x in batch_queries[query] if x != query]
            if len(near_queries) != window_max_size:
                ignored_queries.append(query)
                continue
            batch_data = {}
            curr_pose = Pose(R=buffer[query]['rot'], t=buffer[query]['trans'], device=device)
            batch_data['rgb'] = buffer[query]['rgb']
            batch_data['rgb_nears'] = [ buffer[k]['rgb'] for k in near_queries ]
            batch_data['poses'] = [ Pose(R=buffer[k]['rot'], t=buffer[k]['trans'], device=device) - curr_pose for k in near_queries ]
            collected_batch_data.append(batch_data)
            collected_intrinsics.append(buffer[query]['intrinsics'])

        # combine batch data
        all_batch_data = {}
        if len(collected_batch_data) != 0:
            all_batch_data["images"] = tuple(
                [torch.cat([b_data['rgb'] for b_data in collected_batch_data], dim=0)] +
                [torch.cat([b_data['rgb_nears'][k] for b_data in collected_batch_data], dim=0) for k in range(window_max_size)] # each with shape (B, 3, H, W)
            )
            all_batch_data['poses'] = tuple([torch.stack([b_data['poses'][k].transformation_matrix() for b_data in collected_batch_data]) for k in range(window_max_size)])  # each with shape (B, 4, 4)
        
        # for k in range(window_max_size+1):
        #     print(all_batch_data['images'][k].shape)
        # for k in range(window_max_size):
        #     print(all_batch_data['poses'][k].shape)

        return all_batch_data, collected_intrinsics, ignored_queries

    def __remove_ignored_queries(self, criteria_data, ignored_queries):
        pred = criteria_data.pred
        batch_queries = deepcopy(list(criteria_data.batch_queries.keys()))
        for ignored_query in ignored_queries:
            print(f'[CoDEPS Loss] Warning: ignore idx = {ignored_query} from batch data.')
            bidx = batch_queries.index(ignored_query)
            pred = torch.cat((pred[:bidx], pred[bidx+1:]))
            batch_queries.remove(ignored_query)
        return pred, batch_queries
    
    def forward(self, criteria_data):
        assert criteria_data.device == self.device, "device mismatch"
        # pred, batch_queries = self.__remove_idx_zero_and_one(criteria_data)
        all_batch_data, collected_intrinsics, ignored_queries = self.__collect_batch_data(criteria_data.buffer, criteria_data.batch_queries, criteria_data.device)
        pred, batch_queries = self.__remove_ignored_queries(criteria_data, ignored_queries)
        if len(batch_queries) == 0:
            print('[CoDEPS Loss] Warning: no loss to be computed, ignored this batch')
            return torch.zeros(1, device=criteria_data.device)

        # depth reconstrution loss
        # camera_models: List[CameraModel], with length B
        # images: Tuple[Tensor, Tensor, Tensor], order as [t, t-1, t-2], each with shape (B, 3, H, W)
        # depth_map: Tensor, shape (B, 1, H, W)
        # poses : (Tensor, Tensor), order as camera motion [t -> t-1] and [t -> t-2], each is transformation matrix [[R t], [0, 1]] with shape (B, 4, 4)
        
        # note: pred = (batch, depth/unc/disparity, H, W)
        camera_models = camera_models = [intrinsics.to_CameraModel() for intrinsics in collected_intrinsics]
        depth_map = pred[:,0].unsqueeze(1)
        depth_recon_loss = self.reconstruction_loss(
            camera_models,
            all_batch_data["images"],
            depth_map,
            all_batch_data["poses"]
        )

        # depth smoothness loss
        # images: Tensor, current image, (B, 3, H, W)
        # disp_map: Tensor, disparity map, (B, 1, H, W)
        disp_map = pred[:,2].unsqueeze(1)
        depth_smth_loss = self.smoothness_loss(all_batch_data["images"][0], disp_map)

        loss = (self.recon_loss_weight*depth_recon_loss) + (self.smth_loss_weight*depth_smth_loss)
        return loss