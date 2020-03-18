#!/bin/bash
cd ../../../src
source activate dense_duckie

python -u train_model.py -c mila -l Reward -e duckietown -g random -n dt_r_rn34_r -m resnet34 -r 0.01
python -u test_model.py -c mila -l Reward -e duckietown -g random -n dt_r_rn34_r -m resnet34