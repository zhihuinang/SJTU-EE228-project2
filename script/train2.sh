#!/bin/bash

OPT = ""
OPT+="--task_id try3 "
OPT+="--epoch 10 "
OPT+="--batch_size 1 "
OPT+="--down_rate 0.25 "

CUDA_VISIBLE_DEVICES=1 python3  ../main.py $OPT
