#!/bin/bash

OPT = ""
OPT+="--task_id new_all "
OPT+="--epoch 50 "
OPT+="--batch_size 1 "
OPT+="--gpu 0 "

python3  ../main.py $OPT
