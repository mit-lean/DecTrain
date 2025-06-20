import os
import sys
import pandas as pd
import yaml 
from tqdm import tqdm
from policy.PolicyTrainer import PolicyTrainer
import argparse

def parse_args():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="DecTrain Decision DNN Offline Training Script")
    parser.add_argument("-c", "--configs", type=str, help="Path to the YAML configuration file.")
    parser.add_argument("-d", "--dataset-root", type=str, help="Path to all datasets.")
    parser.add_argument("-o", "--output-dir", type=str, help="Directory to save the results.")
    return parser.parse_args()

def setup_dataset(args, run_parameters):
    """
    Setup dataset
    """
    run_parameters['dataset_dir'] = os.path.join(args.dataset_root, 'decision', run_parameters['dataset_dir'])
    return run_parameters

def offline_policy_training(run_parameters, output_dir):
    policyTrainer = PolicyTrainer(run_parameters, output_dir)
    policyTrainer.train_offline_model()
    policyTrainer.test_offline_model()
    return

def offline_policy_testing(run_parameters, output_dir):
    policyTrainer = PolicyTrainer(run_parameters, output_dir)
    policyTrainer.test_offline_model(model_name="policy_model_best_val.pkl", tag="post")
    return

def main():
    args = parse_args()
    if not args.configs or not args.dataset_root or not args.output_dir:
        print("Usage: python policy_training.py -c <configs> -d <dataset> -o <output_dir>")
        sys.exit(1)
    with open(args.configs) as f: 
        run_parameters = yaml.load(f, Loader=yaml.FullLoader)
    run_parameters = setup_dataset(args, run_parameters)
    offline_policy_training(run_parameters, args.output_dir)
    offline_policy_testing(run_parameters, args.output_dir)

if __name__ == '__main__':
    main()