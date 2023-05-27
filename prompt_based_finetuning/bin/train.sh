#!/usr/bin/env bash

set -exu

config_file=$1
echo "$config_file"
CUDA_VISIBLE_DEVICES=0,3 python -m src.train -c $config_file
