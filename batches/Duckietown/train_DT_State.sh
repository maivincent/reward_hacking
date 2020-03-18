#!/bin/bash
cd ../../src
source activate dense_duckie

if [ $# -ne 3 ]
then
	echo "Arg 1: model type -m. Arg 2: name of the data set -g. Arg 3: name of the CNN -n"
else
	python -u train_model.py -c mila -l State -e duckietown -g $2 -n $3 -m $1
	python -u test_model.py -c mila -l State -e duckietown -g $2 -n $3 -m $1
fi