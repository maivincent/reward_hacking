#!/bin/bash
cd ../src
source activate dense_duckie

for i in {1..20}
do
   python -u rl_solver_sac.py -c mila -t DT_Noisy_Reward_10 -e duckietown -g random 
done


