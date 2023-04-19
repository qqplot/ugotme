#!/bin/bash

#SBATCH --job-name=cml_large_test
#SBATCH --gres=gpu:1
#SBATCH --time=0-12:00:00
#SBATCH --mem=16000MB
#SBATCH --cpus-per-task=8
#SBATCH --output=./slurm_log/cifar10/S-%x.%j.out

# eval "$(conda shell.bash hook)"
# conda activate maicon

# CIFAR-C
SEEDS="0 1 2"
SHARED_ARGS="\
    --data_dir ./data/\
    --dataset cifar-c \
    --num_epochs 100 \
    --n_samples_per_group 2000 \
    --test_n_samples_per_group 3000 \
    --n_context_channels 3 \
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
    --epochs_per_eval 50 \
    --smaller_model 0
    "

# N_CONTEXT_CHANNELS=3 # For CML

# python run.py --exp_name erm ${SHARED_ARGS}

# srun python run.py --exp_name cusum_notnorm_context_large --algorithm ARM-CUSUM $SHARED_ARGS

srun python run_test.py --data_dir ./data/ --dataset cifar-c --eval_on test --auto 1 --test 1 --train 0 --ckpt_folders cifar-c_cusum_notnorm_context_large_0_20230411-145847 cifar-c_cusum_notnorm_context_large_1_20230411-151807 cifar-c_cusum_notnorm_context_large_2_20230411-153721



# cifar-c_cml_large_0_20230410-191552 cifar-c_cml_large_1_20230410-193350 cifar-c_cml_large_2_20230410-195144