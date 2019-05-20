#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1 python classify_main.py --save_prefix lr2e-2_save --lr 0.02 |tee 2>&1 lr2e-2-train.log

CUDA_VISIBLE_DEVICES=0,1 python classify_main.py --save_prefix lr7e-2_save --lr 0.07 |tee 2>&1 lr7e-2-train.log

CUDA_VISIBLE_DEVICES=0,1 python classify_main.py --save_prefix lr1e-1_save --lr 0.1 |tee 2>&1 lr1e-1-train.log

CUDA_VISIBLE_DEVICES=0,1 python classify_main.py --save_prefix lr3e-1_save --lr 0.3 |tee 2>&1 lr3e-1-train.log



