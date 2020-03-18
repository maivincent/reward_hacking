#!/bin/bash
cd ../src
source activate dense_cartpole

if [ $# -eq 0 ]
then
	echo "Please give as an argument to the script the name of the CNN you want to use!"
else
	for i in {1..20}
	do
   		xvfb-run -a -s "-screen 0 1400x900x24" python -u rl_solver_sac.py -c mila -t CP_CNN_Reward -e cartpole -g random -n $1
	done
fi
