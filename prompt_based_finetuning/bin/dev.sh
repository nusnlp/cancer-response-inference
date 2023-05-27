#!/usr/bin/env bash

set -exu

exp_dir=$1

CUDA_VISIBLE_DEVICES=0,3 python -m src.dev -e $exp_dir

