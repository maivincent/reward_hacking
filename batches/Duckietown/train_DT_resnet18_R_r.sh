#!/bin/bash
cd ../../src
source activate dense_duckie

python -u train_model.py -c mila -l Reward -e duckietown -g random -n dt_r_rn18_r -m resnet18
python -u test_model.py -c mila -l Reward -e duckietown -g random -n dt_r_rn18_r -m resnet18