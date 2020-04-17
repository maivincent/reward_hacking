#!/bin/bash
cd ../../../src
source activate dense_duckie

python -u train_model.py -c mila -l Reward -e duckietown -g random -n DT_rn18_head_e-4 -m dk_resnet18_DT -r 0.0001
python -u test_model.py -c mila -l Reward -e duckietown -g random -n DT_rn18_reg_e-4 -m dk_resnet18_DT