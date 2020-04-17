#!/bin/bash
cd ../../src
source activate dense_cartpole

python -u train_model.py -c mila -l Reward -e cartpole -g random_weird -n CPWeird_DK_e-2 -m dk_resnet18_CP_weird -r 0.01 -i
python -u test_model.py -c mila -l Reward -e cartpole -g random_weird -n CPWeird_DK_e-2 -m dk_resnet18_CP_weird -i