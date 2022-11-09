#!/usr/bin/env bash

cd /home/ubuntu/Desktop/STMTrack-main/videoanalyst/evaluation/vot_benchmark/pysot/utils/
python3 setup.py clean
python3 setup.py build_ext --inplace
cd ../../
