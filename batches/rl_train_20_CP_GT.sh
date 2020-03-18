#!/bin/bash
cd ../src
source activate dense_cartpole

for i in {1..20}
do
   python -u rl_solver_sac.py -c mila -t CP_GT_Reward -e cartpole 
done


