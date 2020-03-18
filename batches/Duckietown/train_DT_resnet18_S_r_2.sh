#!/bin/bash
cd ../../src
source activate dense_duckie

python -u train_model.py -c mila -l State -e duckietown -g random_2 -n dt_s_rn18_r_2 -m resnet18
python -u test_model.py -c mila -l State -e duckietown -g random_2 -n dt_s_rn18_r_2 -m resnet18