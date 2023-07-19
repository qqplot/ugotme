#!/bin/bash

#SBATCH --job-name=run_train
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=16000MB
#SBATCH --cpus-per-task=16
#SBATCH --output=./slurm_log/femnist/S-%x.%j.out
#SBATCH --time=0-12:00:00

eval "$(conda shell.bash hook)"
conda activate maicon

# MNIST
SEEDS="0 1 2"
SHARED_ARGS="\
    --data_dir ./data/\
    --dataset mnist \
    --train 1 \
    --n_context_channels 12 \
    --eval_on val test \
    --dropout_rate 0.2 \
    --auto 0 \
    --epochs_per_eval 10 \
    --smaller_model 1 \
    --num_epochs 200 \
    --T 3 \
    --optimizer adam \
    --meta_batch_size 6 \
    --support_size 50 \
    "

srun python run.py --algorithm ERM --norm_type batch --learning_rate 1e-4 --model convnet --exp_name erm $SHARED_ARGS
srun python run.py --algorithm ARM-CML --norm_type batch --learning_rate 1e-4 --scheduler none --model convnet --exp_name cml $SHARED_ARGS
srun python run.py --algorithm ARM-CUSUM --norm_type layer --learning_rate 1e-3 --scheduler cosine --model convnet --exp_name cusum $SHARED_ARGS
srun python run.py --algorithm ARM-UNC --norm_type layer --learning_rate 1e-4 --scheduler cosine --model convnet_unc --exp_name unc $SHARED_ARGS


# FEMNIST
SEEDS="0 1 2"
SHARED_ARGS="\
    --data_dir ./data/\
    --dataset femnist \
    --num_epochs 200 \
    --eval_on val test \
    --seeds ${SEEDS} \
    --train 1 \
    --sampler group \
    --uniform_over_groups 1 \
    --n_context_channels 1 \
    --smaller_model 1 \
    --epochs_per_eval 10 \
    --dropout_rate 0.2 \
    --meta_batch_size 2 \
    --support_size 50 \
    --T 3 \
    "

srun python run.py --algorithm ERM --norm_type batch --optimizer sgd --learning_rate 1e-4 --weight_decay 1e-4 --exp_name erm $SHARED_ARGS
srun python run.py --algorithm ARM-CML --norm_type batch --optimizer sgd --learning_rate 1e-4 --weight_decay 1e-4 --exp_name cml $SHARED_ARGS
srun python run.py --algorithm ARM-CUSUM --norm_type layer --optimizer adam --scheduler cosine --learning_rate 1e-4 --exp_name cusum $SHARED_ARGS
srun python run.py --algorithm ARM-UNC --norm_type layer --optimizer adam --learning_rate 1e-4 --scheduler cosine --model convnet_unc --exp_name unc $SHARED_ARGS



