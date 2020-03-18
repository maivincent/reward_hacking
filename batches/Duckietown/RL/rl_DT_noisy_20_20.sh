#!/bin/bash
cd ../../../src
source activate dense_duckie

for i in {1..20}
do
   xvfb-run -a -s "-screen 0 1400x900x24" python -u rl_solver_sac.py -c mila -t DT_Noisy_Reward_20 -e duckietown
done
