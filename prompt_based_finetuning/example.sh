source env1/bin/activate
export ADAPET_ROOT=`pwd`
export PYTHONPATH=$ADAPET_ROOT:$PYTHONPATH
export PYTHON_EXEC=python


####### Example: run prompt-based fine-tuning experiments on GatorTron #######
# Please specify your data path to train on full train set or subsets, e.g., data/medical/train_{} (full, 100, 500, 1000, 5000)
# Place your GatorTron model files in gatortronModel
CUDA_VISIBLE_DEVICES=1 nohup python cli.py -d data/medical \
              -p '[TEXT1] [SEP] In summary, this is a [LBL]' \
              -v '{"0": "no evidence of disease", "1": "partial response", "2": "stable disease", "3": "progressive disease"}' \
              -w gatortronModel \
              -bs 1 \
              --grad_accumulation_factor 16 \
              --num_batches 2000 \
              --eval_every 100 \
              --max_text_length 256 --lr 5e-5 \
              --weight_decay 1e-2 \
              --warmup_ratio 0.06 \
              --pattern_idx 1 \
              --max_num_lbl_tok 2 > run_exp.log 2>&1 &


# [EXP_FOLDER] is the trained model folder name, e.g., 2023-01-26-13-26-43, then get test preds
sh bin/test.sh gatortronModel/[EXP_FOLDER]
# Evaluate on test preds
python get_test_scores_medical.py -y data/medical -pred gatortronModel/[EXP_FOLDER]

