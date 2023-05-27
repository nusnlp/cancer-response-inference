## Experiment of Sentence Permutation + Consistency Loss ##

### Prerequisite

Install required packages from `requirements.txt`.

Download GatorTron weight file (*.nemo) from [Nvidia](https://catalog.ngc.nvidia.com/models) and place it into `ckpt/GatorTron-OG_nemo/`.

### Generation of Permuted Data
```bash
python permutation_fullset.py
```

### Training and Evaluation
Specify `os.environ["CUDA_VISIBLE_DEVICES"]` and run the training script:
```bash
python train.py --config config.json --cuda 1
```

Evaluation results are auto-generated and can be found in `saved/exp_name/train.log`.
