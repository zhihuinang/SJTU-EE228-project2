#!/bin/bash

OPT = ""
OPT+="--task_id new2 "
OPT+="--epoch 2 "
OPT+="--batch_size 2 "
OPT+="--gpu 0,1"

python3  ../main.py $OPT
