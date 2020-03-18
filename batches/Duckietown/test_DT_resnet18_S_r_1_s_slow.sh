#!/bin/bash
cd ../../src
source activate dense_duckie

python -u test_model.py -c mila -l State -e duckietown -g random_1_straight -n dt_s_rn18_r_1_s_slow -m resnet18