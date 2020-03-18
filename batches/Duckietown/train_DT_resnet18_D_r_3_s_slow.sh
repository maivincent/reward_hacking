#!/bin/bash
cd ../../src
source activate dense_duckie

python -u train_model.py -c mila -l Distance -e duckietown -g random_3_straight -n dt_d_rn18_r_3_s_slow -m resnet18 -r 0.001
python -u test_model.py -c mila -l Distance -e duckietown -g random_3_straight -n dt_d_rn18_r_3_s_slow -m resnet18