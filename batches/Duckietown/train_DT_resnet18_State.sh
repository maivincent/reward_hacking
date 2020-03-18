#!/bin/bash
cd ../../src
source activate dense_duckie

python -u train_model.py -c mila -l State -e duckietown -g random -n dt_s_rn18_rand -m resnet18
python -u test_model.py -c mila -l State -e duckietown -g random -n dt_s_rn18_rand -m resnet18