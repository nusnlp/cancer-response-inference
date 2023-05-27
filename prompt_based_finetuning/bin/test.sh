#!/usr/bin/env bash

set -exu

exp_dir=$1

CUDA_VISIBLE_DEVICES=1 python -m src.test -e $exp_dir
