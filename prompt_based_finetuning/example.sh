source env1/bin/activate
export ADAPET_ROOT=`pwd`
export PYTHONPATH=$ADAPET_ROOT:$PYTHONPATH
export PYTHON_EXEC=python


####### Example: run prompt-based fine-tuning experiments on Biomegatron cased #######
# Train on full train set, change to data/medical/train_{} (100, 500, 1000, 5000) for subset experiemnts 
CUDA_VISIBLE_DEVICES=1 nohup python cli.py -d data/medical \
              -p '[TEXT1] [SEP] In summary, this is a [LBL]' \
              -v '{"0": "complete response", "1": "stable disease", "2": "progressive disease", "3": "partial response"}' \
              -w biomegatronModel \
              -bs 1 \
              --grad_accumulation_factor 16 \
              --num_batches 2000 \
              --eval_every 100 \
              --max_text_length 256 --lr 5e-5 \
              --weight_decay 1e-2 \
              --warmup_ratio 0.06 \
              --pattern_idx 1 \
              --max_num_lbl_tok 2 > run_exp.log 2>&1 &


# Get test preds, put the model folder name in [EXP_FOLDER], e.g., 2023-01-26-13-26-43
sh bin/test.sh biomegatron345mcased/[EXP_FOLDER]
# Eval test
python get_test_scores_medical.py -y data/medical -pred biomegatron345mcased/[EXP_FOLDER]

