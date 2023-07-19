#!/bin/bash

#SBATCH --job-name=run_test
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=16000MB
#SBATCH --cpus-per-task=16
#SBATCH --output=./slurm_log/femnist/S-%x.%j.out
#SBATCH --time=0-12:00:00

eval "$(conda shell.bash hook)"
conda activate maicon


YOUR_MODELS = "model1 model2 model3"
DATASET = "mnist" # femnist
DATA_DIR = "./data/"

# Online (Clean)
srun python run_test.py --eval_on test --norm_type layer \
--data_dir ${DATA_DIR} --dataset ${DATASET} --ckpt_folders ${YOUR_MODELS} \
--auto 0 --test 1 --train 0 --online 1 --T 3 \
--noisy 0 --support_size 50 --seeds 0 1 2 \


# Online (Cycle)
srun python run_test.py --eval_on test --norm_type layer \
--data_dir ${DATA_DIR} --dataset ${DATASET} --ckpt_folders ${YOUR_MODELS} \
--auto 0 --test 1 --train 0 --online 1 --T 3 \
--noisy 1 --support_size 50 --seeds 0 1 2 \
--noise_level 0.10 --noise_type sp --normal_iter 10 


# Online (Always)
srun python run_test.py --eval_on test --norm_type layer \
--data_dir ${DATA_DIR} --dataset ${DATASET} --ckpt_folders ${YOUR_MODELS} \
--auto 0 --test 1 --train 0 --online 1 --T 3 \
--noisy 1 --support_size 50 --seeds 0 1 2 \
--noise_level 0.10 --noise_type sp --normal_iter 0 


# Offline (but 'Ours' is still online)
python run.py --eval_on test --norm_type layer \
--data_dir ${DATA_DIR} --dataset ${DATASET} --ckpt_folders ${YOUR_MODELS} \
--auto 0 --test 1 --train 0 --online 0 --normalize 0 --T 3 \
--noisy 0 --support_size 50 --seeds 0 1 2 \


