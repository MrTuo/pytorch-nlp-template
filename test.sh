#! /bin/bash
set -x
set -e
python ./src/test.py -r ./data/saved/Answer_selection/1104_235026/model_best.pth -d 0,1
