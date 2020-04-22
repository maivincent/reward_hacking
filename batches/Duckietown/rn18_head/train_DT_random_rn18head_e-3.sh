#!/bin/bash
cd ../../../src
source activate dense_duckie

python -u train_model.py -c mila -l Reward -e duckietown -g random -n DT_rn18_head_e-3 -m dk_resnet18_DT -r 0.001
python -u test_model.py -c mila -l Reward -e duckietown -g random -n DT_rn18_head_e-3 -m dk_resnet18_DT