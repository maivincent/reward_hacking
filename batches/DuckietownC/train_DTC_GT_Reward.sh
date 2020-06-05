#!/bin/bash
cd ../../src
source activate dense_duckie

xvfb-run -a -s "-screen 0 1400x900x24" python -u rl_solver_sac.py -c local -e duckietown_cam -t DTC_GT_Reward