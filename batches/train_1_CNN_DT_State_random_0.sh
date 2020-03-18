#!/bin/bash
cd ../src
source activate dense_duckie

python -u train_model.py -c mila -l State -e duckietown -g random_0 -n dt_s_rand0_1
python -u test_model.py -c mila -l State -e duckietown -g random_0 -n dt_s_rand0_1