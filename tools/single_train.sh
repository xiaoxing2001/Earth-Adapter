#!/bin/bash

export CUDA_VISIBLE_DEVICES=3
python tools/train.py\
    --config configs/DA/dinov2/earth_adapter/u2r.py\
    --no_debug