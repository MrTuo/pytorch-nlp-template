#! /bin/bash
set -x
set -e
python ./src/train.py -c ./src/config.json -d 0,1
