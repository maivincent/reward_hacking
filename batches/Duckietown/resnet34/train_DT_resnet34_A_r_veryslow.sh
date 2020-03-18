#!/bin/bash
cd ../../../src
source activate dense_duckie

python -u train_model.py -c mila -l Angle -e duckietown -g random -n dt_a_rn34_r_veryslow -m resnet34 -r 0.0001
python -u test_model.py -c mila -l Angle -e duckietown -g random -n dt_a_rn34_r_veryslow -m resnet34