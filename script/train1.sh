#!/bin/bash

OPT = ""
OPT+="--task_id try2 "
OPT+="--epoch 10 "
OPT+="--batch_size 6 "

CUDA_VISIBLE_DEVICES=0 python3  ../main.py $OPT
