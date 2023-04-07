#!/bin/bash

#SBATCH --job-name=arm_unc_large
#SBATCH --gres=gpu:1
#SBATCH --time=0-12:00:00
#SBATCH --mem=16000MB
#SBATCH --cpus-per-task=8
#SBATCH --output=./slurm_log/S-%x.%j.out

eval "$(conda shell.bash hook)"
conda activate maicon


SEEDS="0"
SHARED_ARGS="\
    --data_dir ./data/\
    --dataset femnist \
    --train 1 \
    --num_epochs 200 \
    --n_context_channels 12 \
    --eval_on val test \
    --seeds ${SEEDS} \
    --dropout_rate 0.5 \
    --model convnet_unc \
    --T 3 \
    --auto 1 \
    --beta 1.0 \
    --epochs_per_eval 20 \
    --smaller_model 0 \
    "
    

# Origin: Train (Batch 6 * 50)
# srun python run.py --algorithm ERM --auto 1 --exp_name arm_unc_femnist $SHARED_ARGS



srun python run.py --algorithm ARM-UNC --meta_batch_size 2 --support_size 50 --exp_name arm_unc_large $SHARED_ARGS



# python run.py --eval_on test --test 1 --train 0 --ckpt_folders mnist_unc_coef2_drop30_6_50_0_20230320-103020 --meta_batch_size 1 --support_size 1  
# python run.py --eval_on test --test 1 --train 0 --ckpt_folders mnist_unc_coef05_drop30_6_50_0_20230318-101158 --meta_batch_size 1 --support_size 1  
# python run.py --eval_on test --test 1 --train 0 --ckpt_folders mnist_unc_drop30_6_50_0_20230315-104713 --meta_batch_size 1 --support_size 1  
# python run_bk.py --eval_on test --test 1 --train 0 --ckpt_folders mnist_ori50_0_20230203-002701 --meta_batch_size 6 --support_size 50 --noisy 1
# python run_bk.py --eval_on test --test 1 --train 0 --ckpt_folders mnist_unc_drop30_6_50_0_20230315-104713 --meta_batch_size 6 --support_size 50 --noisy 1


# Origin: Train (Batch 6 * 50) -> default
# srun python run.py --algorithm ARM-CML --meta_batch_size 6 --support_size 50 --exp_name ori6_50 $SHARED_ARGS


# Origin: Train (Batch 1 * 1)
# srun python run.py --algorithm ARM-CML --meta_batch_size 1 --support_size 1 --exp_name ori1_1 $SHARED_ARGS
# python run.py --eval_on test --test 1 --train 0 --ckpt_folders mnist_arm_cml1_0_20230203-000058 --meta_batch_size 1 --support_size 50  

# Origin: Train (Batch 1 * 50)
# srun python run.py --algorithm ARM-CML --meta_batch_size 1 --support_size 50 --exp_name ori1_50 $SHARED_ARGS
# python run.py --eval_on test --test 1 --train 0 --ckpt_folders mnist_ori50_0_20230203-002701 --meta_batch_size 1 --support_size 50  

# Origin: Train (Batch 1 * 5)
# srun python run.py --algorithm ARM-CML --meta_batch_size 1 --support_size 5 --exp_name ori1_5 $SHARED_ARGS


# Origin: Train (Batch 6 * 1)
# srun python run.py --algorithm ARM-CML --meta_batch_size 6 --support_size 1 --exp_name ori6_1 $SHARED_ARGS
# python run.py --eval_on test --test 1 --train 0 --ckpt_folders mnist_ori6_1_0_20230203-013559 --meta_batch_size 1 --support_size 50  

# Origin: Train (Batch 6 * 5)
# srun python run.py --algorithm ARM-CML --meta_batch_size 6 --support_size 5 --exp_name ori6_5 $SHARED_ARGS
# python run.py --eval_on test --test 1 --train 0 --ckpt_folders mnist_ori6_5_0_20230203-020218 --meta_batch_size 1 --support_size 50



#########


# Cumsum-Meta: Train (Batch 6 * 50)
# srun python run.py --algorithm MY-ARM-CML --meta_batch_size 6 --support_size 50 --exp_name cum6_50 $SHARED_ARGS
# python run.py --eval_on test --test 1 --train 0 --ckpt_folders mnist_cum6_50_0_20230203-011443 --meta_batch_size 1 --support_size 50

# Cumsum-Meta: Train (Batch 6 * 5)
# srun python run.py --algorithm MY-ARM-CML --meta_batch_size 6 --support_size 5 --exp_name cum6_5 $SHARED_ARGS
# python run.py --eval_on test --test 1 --train 0 --ckpt_folders mnist_cum6_5_0_20230203-020043 --meta_batch_size 1 --support_size 50

# Cumsum-Meta: Train (Batch 6 * 1)
# srun python run.py --algorithm MY-ARM-CML --meta_batch_size 6 --support_size 1 --exp_name cum6_1 $SHARED_ARGS


# Cumsum-Meta: Train (Batch 1 * 1)
# srun python run.py --algorithm MY-ARM-CML --meta_batch_size 1 --support_size 1 --exp_name cum1_1 $SHARED_ARGS

# Cumsum-Meta: Train (Batch 1 * 5)
# srun python run.py --algorithm MY-ARM-CML --meta_batch_size 1 --support_size 5 --exp_name cum1_5 $SHARED_ARGS


# Cumsum-Meta: Train (Batch 1 * 50)
# srun python run.py --algorithm MY-ARM-CML --meta_batch_size 1 --support_size 50 --exp_name cum1_50 $SHARED_ARGS
# python run.py --eval_on test --test 1 --train 0 --ckpt_folders mnist_cum1_50_0_20230203-012510 --meta_batch_size 1 --support_size 50

# sbatch --dependency=afterok:<jobID> job.sh

# sbatch tmux launch-shell


