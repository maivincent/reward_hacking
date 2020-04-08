#!/bin/bash
cd ../../src
source activate dense_cartpole

python -u train_model.py -c mila -l Reward -e cartpole -g random_weird -n CPWeird_DK_e-3 -m resnet18 -r 0.001
python -u test_model.py -c mila -l Reward -e cartpole -g random_weird -n CPWeird_DK_e -m resnet18