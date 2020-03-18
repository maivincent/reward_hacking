#!/bin/bash
cd ../src
source activate dense_cartpole

python -u test_model.py -c mila -l State -e cartpole -g random -n cp_s_rand_1 -i
python -u test_model.py -c mila -l State -e cartpole -g random -n cp_s_rand_2 -i
python -u test_model.py -c mila -l State -e cartpole -g random -n cp_s_rand_3 -i
