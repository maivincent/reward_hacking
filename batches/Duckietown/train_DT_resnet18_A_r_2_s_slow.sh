#!/bin/bash
cd ../../src
source activate dense_duckie

python -u train_model.py -c mila -l Angle -e duckietown -g random_2_straight -n dt_a_rn18_r_2_s_slow -m resnet18 -r 0.001
python -u test_model.py -c mila -l Angle -e duckietown -g random_2_straight -n dt_a_rn18_r_2_s_slow -m resnet18