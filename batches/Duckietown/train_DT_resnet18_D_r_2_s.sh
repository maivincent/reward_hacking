#!/bin/bash
cd ../../src
source activate dense_duckie

python -u train_model.py -c mila -l Distance -e duckietown -g random_2_straight -n dt_d_rn18_r_2_s -m resnet18
python -u test_model.py -c mila -l Distance -e duckietown -g random_2_straight -n dt_d_rn18_r_2_s -m resnet18