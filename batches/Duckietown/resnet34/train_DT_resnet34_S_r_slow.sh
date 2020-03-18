#!/bin/bash
cd ../../../src
source activate dense_duckie

python -u train_model.py -c mila -l State -e duckietown -g random -n dt_s_rn34_r_slow -m resnet34 -r 0.001
python -u test_model.py -c mila -l State -e duckietown -g random -n dt_s_rn34_r_slow -m resnet34