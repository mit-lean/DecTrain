import sys
import os

import src.train as trainer
import yaml
import argparse

def parse_args():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="DecTrain Depth DNN Online Training Script")
    parser.add_argument("-c", "--configs", type=str, help="Path to the YAML configuration file.")
    parser.add_argument("-d", "--dataset-root", type=str, help="Path to all datasets.")
    parser.add_argument("-m", "--model-root", type=str, help="Path to DNN models (depth, decision).")
    parser.add_argument("-o", "--output-dir", type=str, help="Directory to save the results.")
    return parser.parse_args()

def setup_model_dataset(args, run_parameters):
    """
    Setup model, dataset
    """
    run_parameters['resume'] = [ os.path.join(args.model_root, 'depth', run_parameters['arch'], p) for p in run_parameters['resume'] ]
    run_parameters['train_dir'] = [ os.path.join(args.dataset_root, 'depth', p) for p in run_parameters['train_dir'] ]
    if 'depth_dir' in run_parameters:
        run_parameters['depth_dir'] = [ os.path.join(args.dataset_root, 'depth', p) for p in run_parameters['depth_dir'] ]
    run_parameters['val_target_dir'] = [ os.path.join(args.dataset_root, 'depth', p) for p in run_parameters['val_target_dir'] ]
    if run_parameters['policy_model_path'] is not None:
        run_parameters['policy_model_path'] = os.path.join(args.model_root, 'decision', run_parameters['policy_model_path'])
    if run_parameters['online_buffer_configs'] is not None:
        if run_parameters['online_buffer_configs']['activate_source_buffer']:
            assert run_parameters['online_buffer_configs']['source_buffer_load_path'] is not None, "Source buffer load path must be specified when source buffer is activated."
            run_parameters['online_buffer_configs']['source_buffer_load_path'] = os.path.join(args.dataset_root, 'decision', run_parameters['online_buffer_configs']['source_buffer_load_path'])
    return run_parameters

# read in run_parameters as directory
def run():
    args = parse_args()
    if not args.configs or not args.dataset_root or not args.model_root or not args.output_dir:
        print("Usage: python train_dnn.py -c <configs> -d <dataset> -m <model> -o <output_dir>")
        sys.exit(1)
    # run online training
    with open(args.configs) as f: 
        run_parameters = yaml.load(f, Loader=yaml.FullLoader)
        run_parameters = setup_model_dataset(args, run_parameters)
    # create results folder, if not already exists
    os.makedirs(args.output_dir, exist_ok=True)
    # train DNN online, return updated run_parameters
    run_parameters = trainer.train(run_parameters, args.output_dir)
    # write experimented configs
    with open(os.path.join(args.output_dir, "run_parameters.yaml"), 'w') as f:
        data = yaml.dump(run_parameters, f)

if __name__ == "__main__":
    run()