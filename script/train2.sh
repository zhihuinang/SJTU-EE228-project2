#!/bin/bash

OPT = ""
OPT+="--task_id new_10sample "
OPT+="--epoch 50 "
OPT+="--batch_size 1 "
OPT+="--gpu 0,1 "

python3  ../main.py $OPT
