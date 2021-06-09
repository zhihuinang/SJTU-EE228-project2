#!/bin/bash

OPT = ""
OPT+="--task_id new1 "
OPT+="--epoch 2 "
OPT+="--batch_size 1 "

CUDA_VISIBLE_DEVICES=1 python3  ../main.py $OPT
