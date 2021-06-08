#!/bin/bash

OPT = ""
OPT+="--task_id try4 "
OPT+="--epoch 10 "
OPT+="--batch_size 6 "

CUDA_VISIBLE_DEVICES=1 python3  ../main.py $OPT
