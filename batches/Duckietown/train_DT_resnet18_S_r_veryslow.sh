#!/bin/bash
cd ../../src
source activate dense_duckie

python -u train_model.py -c mila -l State -e duckietown -g random -n dt_s_rn18_r_veryslow -m resnet18 -r 0.0001
python -u test_model.py -c mila -l State -e duckietown -g random -n dt_s_rn18_r_veryslow -m resnet18