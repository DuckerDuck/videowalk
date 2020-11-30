#!/usr/bin/bash
python -W ignore train.py --data-path ../kinetics \
--frame-aug grid --dropout 0.1 --clip-len 4 --temp 0.05 \
--model-type scratch --workers 16 --batch-size 20  \
--cache-dataset --data-parallel --visualize --lr 0.0001
