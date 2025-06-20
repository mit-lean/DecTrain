#!/bin/bash
# This script is used to run offline training for DecTrain decision DNN with spcified configurations.

python3 examples/offline_train_decision_dnn.py \
    -c configs/decision-dnn/resnet101-monodepth2-decision-dnn.yaml \
    -d datasets/ \
    -o outputs/resnet101-monodepth2-decision-dnn/