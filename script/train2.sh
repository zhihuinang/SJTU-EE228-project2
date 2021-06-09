#!/bin/bash

OPT = ""
OPT+="--task_id try4 "
OPT+="--epoch 10 "
OPT+="--batch_size 1 "
OPT+="--down_rate 0.125 "

CUDA_VISIBLE_DEVICES=1 python3  ../main.py $OPT
