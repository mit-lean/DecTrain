#!/bin/bash
# This script is used to run online training with specified configurations.
CONFIG_FILE=configs/table2/exp1_dectrain/config_1.yaml
DATASETS_PATH=datasets/
MODELS_PATH=models/
OUTPUT_PATH=outputs/online_depth/

# run online depth DNN training
python3 examples/online_train_depth_dnn.py \
        -c $CONFIG_FILE \
        -d $DATASETS_PATH \
        -m $MODELS_PATH \
        -o $OUTPUT_PATH

# compute estimation for the online training process
python3 examples/compute_estimation.py \
        -c $CONFIG_FILE \
        -m $MODELS_PATH \
        -o $OUTPUT_PATH