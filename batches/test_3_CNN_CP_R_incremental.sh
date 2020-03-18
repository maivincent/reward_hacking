#!/bin/bash
cd ../src
source activate dense_cartpole

python -u test_model.py -c mila -l Reward -e cartpole -g random -n cp_r_rand_1 -i
python -u test_model.py -c mila -l Reward -e cartpole -g random -n cp_r_rand_2 -i
python -u test_model.py -c mila -l Reward -e cartpole -g random -n cp_r_rand_3 -i
