import os
import sys
import yaml 
import pickle

import pandas as pd
import argparse


from dataclasses import dataclass

@dataclass
class FineDepthTrainingFlopsConfig:
    replay_num: int = 2

@dataclass
class FineEpistemicFlopsConfig:
    ufm_eps_method: str = "none" # mc-ufm, mc
    ufm_depth_decoder: str = "none"
    ufm_use_aleatoric: bool = False
    ufm_use_extra_depth: bool = False
    ufm_mc_samples: int = 0

@dataclass
class FinePolicyTrainingFlopsConfig:
    policy_train_method: str = "none" # none, reinference
    policy_train_period: int = 1
    policy_train_epoch: int = 0
    policy_train_input_cnt: int = 0

@dataclass
class FineFlopsEstimationConfigs:
    # flow setup
    use_depth_inference: bool = False
    use_depth_training: bool = False
    use_policy_inference: bool = False
    use_policy_training: bool = False
    # detailed setup
    dnn_encoder: str = "none"
    dnn_depth_decoder: str = "none"
    self_supervised_loss: str = "none"
    landmark_extraction: str = "none"
    aleatoric_extration: str = "none"
    dnn_policy_model: str = "none"
    # policy setup
    policy_use_landmark: bool = False
    policy_use_aleatoric: bool = False
    policy_use_epistemic: bool = False
    policy_use_loss: bool = False
    # depth training setup
    depth_training_configs: FineDepthTrainingFlopsConfig = FineDepthTrainingFlopsConfig()
    # epistemic unc setup
    eps_unc_configs: FineEpistemicFlopsConfig = FineEpistemicFlopsConfig()
    # policy training setup
    policy_training_configs: FinePolicyTrainingFlopsConfig = FinePolicyTrainingFlopsConfig()

class FlopsEstimator:
    def __init__(self):
        self.fine_flops_dict = self._get_fine_flops_dict()
    
    def _get_fine_flops_dict(self):
        # Phase: depth inference
        dnn_encoder_flops_dict = {
            "resnet101": 15752228352, # 15.75 GFLOPs
            "resnet50": 8278312448, # 8.28 GFLOPs
            "resnet18": 3380306944, # 3.38 GFLOPs
            "dinov2-small": 10918939008, # 10.92 GFLOPs
            "dinov2-base": 43666084608, # 43.67 GFLOPs
            "dinov2-large": 155248329728, # 155.25 GFLOPs
            "dinov2-giant": 582191171072, # 582.19 GFLOPs
            "resnet101-kitti360": 169738297344, # 169.74 GFLOPs
            "resnet101-kitti": 150449854464, # 150.45 GFLOPs
            "dinov2-small-kitti360": 120193301376, # 120.19 GFLOPs
        }
        dnn_depth_decoder_inference_flops_dict = {
            "monodepthv2": 5348426048, # 5.35 GFLOPs
            "resnet50-monodepthv2": 5348426048, # 5.35 GFLOPs
            "resnet18-monodepthv2": 2920710464, # 2.92 GFLOPs
            "dinov2small-linear4": 6446740480, # 6.45 GFLOPs
            "dinov2small-dpt": 64653541376, # 64.65 GFLOPs
            "dinov2base-dpt": 68588847104,  # 68.59 GFLOPs
            "dinov2large-dpt": 72040759296, # 72.04 GFLOPs
            "dinov2giant-dpt": 80932683776, # 80.93 GFLOPs
            "monodepthv2-kitti360": 57632019456, # 57.63 GFLOPs
            "monodepthv2-kitti": 51082926336, # 51.08 GFLOPs
            "dinov2small-dpt-kitti360": 722931224064, # 722.93 GFLOPs
        }
        # Phase: depth decoder training
        self_supervised_loss_flops_dict = {
            "codeps_depth_loss": 137916468, # 137.92 MFLOPs, per image
            "codeps_depth_loss-kitti360": 1469174928, # 1.47 GFLOPs, per image
            "codeps_depth_loss-kitti": 1302306168, # 1.30 GFLOPs, per image
        }
        dnn_depth_decoder_training_flops_dict = {
            "monodepthv2": 18594572486, # 18.59 GFLOPs, per image
            "resnet50-monodepthv2": 18594572486, # 18.59 GFLOPs, per image
            "resnet18-monodepthv2": 10527877318, # 10.53 GFLOPs, per image
            "dinov2small-linear4": 73435989, # 0.07 GFLOPs, per image
            "dinov2small-dpt": 128221157582, # 128.22 GFLOPs, per image
            "monodepthv2-kitti360": 210327572242, # 210.33 GFLOPs, per image
            "monodepthv2-kitti": 186426753762, # 186.43 GFLOPs, per image
        }
        # Phase: policy inference
        landmark_extration_flops_dict = {
            "sift": 566222177, # 566 MFLOPs, per image, others
            "sift-kitti360": 974087834, # 974MFLOPs, per image, kitti-360
            "sift-kitti": 858745098, # 858 MFLOPs, per image, kitti
        }
        aleatoric_uncertainty_flops_dict = {
            "monodepthv2": 5346892544, # 5.35 GFLOPs
            "resnet50-monodepthv2": 5346892544, # 5.35 GFLOPs
            "resnet18-monodepthv2": 2919176960, # 2.92 GFLOPs
            "dinov2small-adabin": 0,
            "dinov2small-linear4": 6446740480,
            "dinov2small-resnet101-monodepth2": 15752228352+5346892544,
            "monodepthv2-kitti360": 57615495168, # 57.62 GFLOPs
            "monodepthv2-kitti": 51068279808, # 51.07 GFLOPs
        }
        ufm_depth_decoder_flops_dict = {
            "monodepthv2-p.05": 5348426048,
            "resnet50-monodepthv2-p.05": 5348426048,
            "resnet18-monodepthv2-p.05": 2920710464,
            # "dinov2-linear4-p.05": 6446740480,
            # "dinov2-dpt-p.05": 64653541376,
            "dinov2small-linear4-p.05": 6446740480,
            "dinov2small-dpt-p.05": 64653541376,
            "monodepthv2-p.05-kitti360": 57632019456,
            "monodepthv2-p.05-kitti": 51082926336,
        }
        ufm_epistemic_uncertainty_flops_dict = {
            "ufm": 26007848, # 26 MFLOPs
            "ufm-kitti360": 404095894, # 404 MFLOPs
            "ufm-kitti": 470278165, # 470 MFLOPs
        }
        dnn_policy_inference_flops_dict = {
            "dnn-mlp-32": 4864,
            "dnn-mlp-256": (42*256+256*256+256)*2,
        }
        # Phase: policy training
        dnn_policy_training_flops_dict = {
            "dnn-mlp-32": 5436, # per input, per epoch
            "dnn-mlp-256": (42*256+256*256+256)*2,
        }

        fine_flops_dict = {
            "dnn_encoder": dnn_encoder_flops_dict,
            "dnn_depth_decoder_inference": dnn_depth_decoder_inference_flops_dict,
            "self_supervised_loss": self_supervised_loss_flops_dict,
            "dnn_depth_decoder_training": dnn_depth_decoder_training_flops_dict,
            "landmark_extraction": landmark_extration_flops_dict,
            "dnn_aleatoric_uncertainty": aleatoric_uncertainty_flops_dict,
            "ufm_depth_decoder": ufm_depth_decoder_flops_dict,
            "ufm_epistemic_uncertainty": ufm_epistemic_uncertainty_flops_dict,
            "dnn_policy_inference": dnn_policy_inference_flops_dict,
            "dnn_policy_training": dnn_policy_training_flops_dict,
        }

        return fine_flops_dict
    
    def _get_fine_epistemic_unc_flops(self, configs: FineEpistemicFlopsConfig):
        eps_flops = 0
        if configs.ufm_eps_method == "mc-ufm":
            if configs.ufm_use_extra_depth:
                eps_flops += self.fine_flops_dict["ufm_depth_decoder"][configs.ufm_depth_decoder]
            eps_flops += self.fine_flops_dict["ufm_epistemic_uncertainty"]["ufm"]
        elif configs.ufm_eps_method == "mc-ufm-kitti360":
            if configs.ufm_use_extra_depth:
                eps_flops += self.fine_flops_dict["ufm_depth_decoder"][configs.ufm_depth_decoder]
            eps_flops += self.fine_flops_dict["ufm_epistemic_uncertainty"]["ufm-kitti360"]
        elif configs.ufm_eps_method == "mc":
            eps_flops += configs.ufm_mc_samples * self.fine_flops_dict["ufm_depth_decoder"][configs.ufm_depth_decoder]
        else:
            raise NotImplementedError(f"Unknown epistemic uncertainty method: {configs.ufm_eps_method}")
        
        return eps_flops

    def _get_fine_policy_train_flops(self, configs: FineEpistemicFlopsConfig):
        policy_train_flops = 0
        # compute cost for reinference
        policy_train_flops += self.fine_flops_dict["dnn_depth_decoder_inference"][configs.dnn_depth_decoder] + self.fine_flops_dict["self_supervised_loss"][configs.self_supervised_loss]
        
        policy_train_flops += configs.policy_training_configs.policy_train_epoch * configs.policy_training_configs.policy_train_input_cnt * self.fine_flops_dict["dnn_policy_training"][configs.dnn_policy_model]
        # multipy by 1/train_period
        policy_train_flops /= configs.policy_training_configs.policy_train_period
        return policy_train_flops
    
    def fine_flops_estimation(self, configs: FineFlopsEstimationConfigs, f_train):
        assert f_train >= 0 and f_train <= 1
        depth_inf_flops, depth_train_flops, policy_inf_flops, policy_train_flops = 0, 0, 0, 0
        policy_inf_breakdown = {"landmark": 0, "aleatoric": 0, "epistemic": 0, "loss": 0, "dnn_policy_inference": 0}
        if configs.use_depth_inference:
            # Depth inference phase: dnn_encoder_inference + dnn_depth_decoder_inference
            depth_inf_flops = self.fine_flops_dict["dnn_encoder"][configs.dnn_encoder] + self.fine_flops_dict["dnn_depth_decoder_inference"][configs.dnn_depth_decoder]
        if configs.use_depth_training:
            # Depth training phase: repaly_num*(dnn_depth_dec_inference + loss_cal) + (repaly_num+1)*dnn_depth_decoder_training
            depth_train_flops = configs.depth_training_configs.replay_num*(self.fine_flops_dict["dnn_depth_decoder_inference"][configs.dnn_depth_decoder] + self.fine_flops_dict["self_supervised_loss"][configs.self_supervised_loss]) + (configs.depth_training_configs.replay_num+1)*self.fine_flops_dict["dnn_depth_decoder_training"][configs.dnn_depth_decoder]
            if not configs.use_policy_inference:
                depth_train_flops += self.fine_flops_dict["self_supervised_loss"][configs.self_supervised_loss] # current frame loss
        if configs.use_policy_inference:
            # Policy inference phase: landmark_extraction + aleatoric_uncertainty + f(epistemic_uncertainty) + loss_cal + dnn_policy_inference
            if configs.policy_use_landmark:
                policy_inf_flops += self.fine_flops_dict["landmark_extraction"][configs.landmark_extraction]
                policy_inf_breakdown["landmark"] = self.fine_flops_dict["landmark_extraction"][configs.landmark_extraction]
            if configs.policy_use_aleatoric or configs.eps_unc_configs.ufm_use_aleatoric:
                policy_inf_flops += self.fine_flops_dict["dnn_aleatoric_uncertainty"][configs.aleatoric_extration]
                policy_inf_breakdown["aleatoric"] = self.fine_flops_dict["dnn_aleatoric_uncertainty"][configs.aleatoric_extration]
            if configs.policy_use_epistemic:
                policy_inf_flops += self._get_fine_epistemic_unc_flops(configs.eps_unc_configs)
                policy_inf_breakdown["epistemic"] = self._get_fine_epistemic_unc_flops(configs.eps_unc_configs)
            if configs.policy_use_loss:
                policy_inf_flops += self.fine_flops_dict["self_supervised_loss"][configs.self_supervised_loss]
                policy_inf_breakdown["loss"] = self.fine_flops_dict["self_supervised_loss"][configs.self_supervised_loss]
            # Note: policy inference is dependent on input size, not included here
            policy_inf_flops += self.fine_flops_dict["dnn_policy_inference"][configs.dnn_policy_model]
            policy_inf_breakdown["dnn_policy_inference"] = self.fine_flops_dict["dnn_policy_inference"][configs.dnn_policy_model]
        if configs.use_policy_training:
            # Policy training phase: dnn_depth_decoder_inference + loss_cal + dnn_policy_training
            policy_train_flops = self._get_fine_policy_train_flops(configs)
        flops_per_frame = (depth_inf_flops + policy_inf_flops) + f_train*(depth_train_flops + policy_train_flops)

        flops_meta = {
            # per-step cost
            "depth_inf_flops": depth_inf_flops,
            "depth_train_flops": depth_train_flops,
            "policy_inf_flops": policy_inf_flops,
            "policy_train_flops": policy_train_flops,
            "depth_train_flops_weighted": depth_train_flops*f_train,
            "policy_train_flops_weighted": policy_train_flops*f_train,
            # depth-inf breakdown
            # policy-inf breakdown
            "policy_inf_breakdown": policy_inf_breakdown,
            # train freq
            "f_train": f_train,
        }

        return flops_per_frame, flops_meta
    
    def _read_from_model(self, policy_model_path, models_root):
        
        if policy_model_path is None:
            return "none"
        else:
            # decision model from repo
            policy_model_path = os.path.join(models_root, "decision", policy_model_path)
        if not os.path.exists(policy_model_path):
            raise FileNotFoundError(f"[Error] Policy model path not found: {policy_model_path}")
        print(f"[Compute Estimation] Decision DNN model path: {policy_model_path}")
        with open(policy_model_path, 'rb') as f:
            load_obj = pickle.load(f)
        policy_arch_cofigs = load_obj["config"]["model_cfgs"]
        policy_arch = policy_arch_cofigs["model"]
        policy_hdim = policy_arch_cofigs["hidden_dims"][0]
        return f"{policy_arch}-{policy_hdim}"
    
    def read_fine_setup_from_configs(self, config_file, models_root):
        with open(config_file) as f: 
            run_parameters = yaml.load(f, Loader=yaml.FullLoader)
        run_parameters = self.backward_compatible_config_modification(run_parameters)
        fine_configs = FineFlopsEstimationConfigs()
        # flow setup
        fine_configs.use_depth_inference = True
        fine_configs.use_depth_training = run_parameters["acquisition_type"] not in ["none"]
        fine_configs.use_policy_inference = run_parameters["acquisition_type"] in ["balance_compute_accuracy_gain_policy"]
        fine_configs.use_policy_training = run_parameters["acquisition_type"] in ["balance_compute_accuracy_gain_policy"] and run_parameters["online_policy_train_method"] in ["dnn-adapt"]
        # detailed setup
        if run_parameters["arch"] == "resnet101_monodepth2":
            fine_configs.dnn_encoder = "resnet101"
            fine_configs.dnn_depth_decoder = "monodepthv2"
            fine_configs.aleatoric_extration = "monodepthv2"
        elif run_parameters["arch"] == "resnet18_monodepth2":
            fine_configs.dnn_encoder = "resnet18"
            fine_configs.dnn_depth_decoder = "resnet18-monodepthv2"
            fine_configs.aleatoric_extration = "resnet18-monodepthv2"
        elif run_parameters["arch"] == "resnet50_monodepth2":
            fine_configs.dnn_encoder = "resnet50"
            fine_configs.dnn_depth_decoder = "resnet50-monodepthv2"
            fine_configs.aleatoric_extration = "resnet50-monodepthv2"
        elif run_parameters["arch"] == "dinov2":
            dinov2_size = run_parameters["dinov2_backbone_size"]
            fine_configs.dnn_encoder = f"dinov2-{dinov2_size}"
            fine_configs.dnn_depth_decoder = f"dinov2{dinov2_size}-{run_parameters['dinov2_decoder_type']}"
            fine_configs.aleatoric_extration = f"dinov2{dinov2_size}-adabin" # not used in experiments
        fine_configs.self_supervised_loss = run_parameters["loss"]
        fine_configs.landmark_extraction = "sift"
        fine_configs.dnn_policy_model = self._read_from_model(run_parameters["policy_model_path"], models_root)
        # policy setup
        if fine_configs.use_policy_inference:
            fine_configs.policy_use_landmark = "landmark" not in run_parameters["ablation_target"]
            fine_configs.policy_use_aleatoric = "aleatoric_unc" not in run_parameters["ablation_target"]
            fine_configs.policy_use_epistemic = "epistemic_unc" not in run_parameters["ablation_target"] and run_parameters["unc_method"] != "none"
            fine_configs.policy_use_loss = "loss" not in run_parameters["ablation_target"]
        else:
            fine_configs.policy_use_landmark = False
            fine_configs.policy_use_aleatoric = False
            fine_configs.policy_use_epistemic = False
            fine_configs.policy_use_loss = False
        # depth training setup
        fine_configs.depth_training_configs.replay_num = run_parameters["buffer_window_size"]
        # epistemic unc setup
        fine_configs.eps_unc_configs.ufm_eps_method = run_parameters["unc_method"]
        fine_configs.eps_unc_configs.ufm_depth_decoder = f"{fine_configs.dnn_depth_decoder}-p.05"
        fine_configs.eps_unc_configs.ufm_use_aleatoric = False # FIXME: find the corresponding setup
        fine_configs.eps_unc_configs.ufm_use_extra_depth = (run_parameters["unc_method"] == 'mc-ufm') # FIXME: find the corresponding setup
        fine_configs.eps_unc_configs.ufm_mc_samples = run_parameters["num_inferences"]
        # policy training setup
        if fine_configs.use_policy_training:
            fine_configs.policy_training_configs.policy_train_method = run_parameters["policy_update_method"]
            fine_configs.policy_training_configs.policy_train_period = run_parameters["policy_update_period"]
            fine_configs.policy_training_configs.policy_train_epoch = run_parameters["online_training_configs"]["epochs"]
            fine_configs.policy_training_configs.policy_train_input_cnt = run_parameters["online_training_configs"]["train_target_replay"]+run_parameters["online_training_configs"]["train_source_replay"]

        # kitti-360 profiling
        if run_parameters['height'] == 384 and run_parameters['width'] == 1408:
            fine_configs.dnn_encoder += "-kitti360"
            fine_configs.dnn_depth_decoder += "-kitti360"
            fine_configs.aleatoric_extration += "-kitti360"
            fine_configs.self_supervised_loss += "-kitti360"
            fine_configs.eps_unc_configs.ufm_depth_decoder += "-kitti360"
            fine_configs.eps_unc_configs.ufm_eps_method += "-kitti360"
            fine_configs.landmark_extraction += "-kitti360"
        # kitti profiling
        elif run_parameters['height'] == 384 and run_parameters['width'] == 1248:
            fine_configs.dnn_encoder += "-kitti"
            fine_configs.dnn_depth_decoder += "-kitti"
            fine_configs.aleatoric_extration += "-kitti"
            fine_configs.self_supervised_loss += "-kitti"
            fine_configs.eps_unc_configs.ufm_depth_decoder += "-kitti"
            fine_configs.eps_unc_configs.ufm_eps_method += "-kitti"
            fine_configs.landmark_extraction += "-kitti"
        return fine_configs

    def backward_compatible_config_modification(self, run_parameters):
        # FIXME: remove this before the release
        # backward compatible modification
        if run_parameters["acquisition_type"].startswith("balance_compute_accuracy_gain") and run_parameters["acquisition_type"] != "balance_compute_accuracy_gain_zero_regret":
            run_parameters["acquisition_type"] = "balance_compute_accuracy_gain_policy"
        if "policy_model_path" not in run_parameters:
            run_parameters["policy_model_path"] = run_parameters["linear_regression_model_path"]
        if "policy_update_method" not in run_parameters:
            run_parameters["policy_update_method"] = "none"
        if "ablation_target" not in run_parameters:
            run_parameters["ablation_target"] = []
        if "policy_update_period" not in run_parameters:
            run_parameters["policy_update_period"] = 1
        return run_parameters
    
    def read_train_count_from_results(self, result_file):
        train_df = pd.read_csv(os.path.join(result_file))
        n_notrain = train_df['seq_idx'].iloc[-1] - train_df['num_decisions_train'].iloc[-1] + 1
        n_train = train_df['num_decisions_train'].iloc[-1]
        train_counts = {
            "n_notrain": n_notrain,
            "n_train": n_train
        }
        return train_counts

def run_compute_estimation(args, verbose=False):
    flops_estimator = FlopsEstimator()
    print(f"[Compute Estimation] Exp. config file: {args.configs}")
    print(f"[Compute Estimation] Exp. result dir: {args.output_dir}")
    fine_configs = flops_estimator.read_fine_setup_from_configs(args.configs, args.model_root)
    if verbose:
        print("[Compute Estimation] Configs for compute estimation:")
        print(fine_configs)
    train_counts = flops_estimator.read_train_count_from_results(os.path.join(args.output_dir,"train_dnn_0.csv"))
    f_train = train_counts["n_train"]/(train_counts["n_notrain"]+train_counts["n_train"])
    flops_per_frame, flops_meta = flops_estimator.fine_flops_estimation(
        fine_configs, f_train
    )
    avg_gflops_per_frame = flops_per_frame/1e9
    avg_percent_trained = f_train*100
    print(f"[Compute Estimation] Results: % trained = {avg_percent_trained:.2f}% | GFLOPs/img = {avg_gflops_per_frame:.1f}")
    print()

def parse_args():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="Online DNN Training Compute Estimation Script.")
    parser.add_argument("-c", "--configs", type=str, help="Path to the YAML configuration file for running the online depth training experiment.")
    parser.add_argument("-m", "--model-root", type=str, help="Path to DNN models (depth, decision).")
    parser.add_argument("-o", "--output-dir", type=str, help="Directory to the saved results of the experiment.")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    if not args.configs or not args.model_root or not args.output_dir:
        print("Usage: python compute_estimation.py -c <configs> -r <models_root> -o <output_dir>")
        sys.exit(1)
    run_compute_estimation(args)