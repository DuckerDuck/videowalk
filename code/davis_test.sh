#!/usr/bin/bash

python test.py \
	--filelist ./eval/davis_vallist.txt \
	--model-type scratch \
	--resume pretrained.pth \
	--save-path ./davis_output/ \
	--topk 10 \
	--videoLen 20 \
	--radius 12  \
	--temperature 0.05  \
	--cropSize -1
