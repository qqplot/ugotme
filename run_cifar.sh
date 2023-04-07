#!/bin/bash

#SBATCH --job-name=cml_large
#SBATCH --gres=gpu:1
#SBATCH --time=0-12:00:00
#SBATCH --mem=16000MB
#SBATCH --cpus-per-task=8
#SBATCH --output=./slurm_log/cifar10/S-%x.%j.out

# eval "$(conda shell.bash hook)"
# conda activate maicon

# CIFAR-C
SEEDS="0"
SHARED_ARGS="\
    --data_dir ./data/\
    --dataset cifar-c \
    --num_epochs 100 \
    --n_samples_per_group 2000 \
    --test_n_samples_per_group 3000 \
    --eval_on val test \
    --seeds ${SEEDS} \
    --meta_batch_size 3 \
    --support_size 100 \
    --optimizer sgd \
    --learning_rate 1e-2 \
    --weight_decay 1e-4 \
    --auto 1
    --dropout_rate 0.3 \
    --adapt_bn 0 \
    --model convnet \
    --T 3 \
    --auto 1 \
    --beta 1.0 \
    --epochs_per_eval 10 \
    --smaller_model 0    
    "

N_CONTEXT_CHANNELS=3 # For CML

# python run.py --exp_name erm ${SHARED_ARGS}

srun python run.py --exp_name cml_large --algorithm ARM-CML $SHARED_ARGS
