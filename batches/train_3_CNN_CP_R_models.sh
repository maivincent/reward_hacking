#!/bin/bash
cd ../src
source activate dense_cartpole

python -u train_model.py -c mila -l Reward -e cartpole -g random -n cp_r_rand_1
python -u test_model.py -c mila -l Reward -e cartpole -g random -n cp_r_rand_1

python -u train_model.py -c mila -l Reward -e cartpole -g random -n cp_r_rand_2
python -u test_model.py -c mila -l Reward -e cartpole -g random -n cp_r_rand_2

python -u train_model.py -c mila -l Reward -e cartpole -g random -n cp_r_rand_3
python -u test_model.py -c mila -l Reward -e cartpole -g random -n cp_r_rand_3