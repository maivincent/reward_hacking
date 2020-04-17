#!/bin/bash
cd ../../../src
source activate dense_duckie

python -u train_model.py -c mila -l Reward -e duckietown -g random -n DT_rn18_reg_e-3 -m resnet18 -r 0.001
python -u test_model.py -c mila -l Reward -e duckietown -g random -n DT_rn18_reg_e-3 -m resnet18