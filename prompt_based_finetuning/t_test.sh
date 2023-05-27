#!/usr/bin/bash 

echo "p-value for subset 100, 500, 1k, 5k, and full"

# subset 100
python t_test.py -base /home/rlin/bio/ADAPET-copy/gatortron-finetuning/saved/nemo_gatortron_baseline_100_sd802/LOCAL_test_0.jsonl \
-pet /home/rlin/bio/ADAPET-copy/gatortronog-new/2023-05-08-09-16-42/test.json \
-y /home/rlin/bio/ADAPET-copy/data/medical_updated/train_100/test.jsonl


# subset 500
python t_test.py -base /home/rlin/bio/ADAPET-copy/gatortron-finetuning/saved/nemo_gatortron_baseline_500_sd802/LOCAL_test_0.jsonl \
-pet /home/rlin/bio/ADAPET-copy/gatortronog-new/2023-05-08-09-17-17/test.json \
-y /home/rlin/bio/ADAPET-copy/data/medical_updated/train_100/test.jsonl


# subset 1000
python t_test.py -base /home/rlin/bio/ADAPET-copy/gatortron-finetuning/saved/nemo_gatortron_baseline_1000_sd802/LOCAL_test_0.jsonl \
-pet /home/rlin/bio/ADAPET-copy/gatortronog-new/2023-05-08-09-18-25/test.json \
-y /home/rlin/bio/ADAPET-copy/data/medical_updated/train_100/test.jsonl


# subset 5000
python t_test.py -base /home/rlin/bio/ADAPET-copy/gatortron-finetuning/saved/nemo_gatortron_baseline_5000_sd802/LOCAL_test_0.jsonl \
-pet /home/rlin/bio/ADAPET-copy/gatortronog-new/2023-05-08-09-20-42/test.json \
-y /home/rlin/bio/ADAPET-copy/data/medical_updated/train_100/test.jsonl


# full train
python t_test.py -base /home/qlin/workspace/2medical/SSPC_BioMegatron/saved/nemo_gatortron_v1_b8_ref/test_pred.jsonl \
-pet /home/rlin/bio/ADAPET-copy/gatortronog-new/2023-05-15-00-37-52/test.json \
-y /home/rlin/bio/ADAPET-copy/data/medical_updated/train_100/test.jsonl
