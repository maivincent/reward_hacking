#!/bin/bash
cd ../../src
source activate dense_duckie

python -u train_model.py -c mila -l State -e duckietown -g random_0_straight -n dt_s_rn18_r_0_s_slow -m resnet18 -r 0.001
python -u test_model.py -c mila -l State -e duckietown -g random_0_straight -n dt_s_rn18_r_0_s_slow -m resnet18