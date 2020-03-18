#!/bin/bash

cd ../../src
source activate dense_duckie

xvfb-run -a -s "-screen 0 1400x900x24" python -u generate_images.py -c mila -e duckietown -g random