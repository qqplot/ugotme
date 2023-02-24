#!/bin/bash

#SBATCH --job-name=test_ori2_50
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=0-12:00:00
#SBATCH --mem=16000MB
#SBATCH --cpus-per-task=8
#SBATCH --output=./slurm_log/S-%x.%j.out

eval "$(conda shell.bash hook)"
conda activate maicon

# FEMNIST
SEEDS="0 1 2"
N_CONTEXT_CHANNELS=1 # For CML
SHARED_ARGS="\
    --data_dir ./data/\
    --dataset femnist \
    --num_epochs 200 \
    --eval_on val test \
    --seeds ${SEEDS} \
    --epochs_per_eval 1 \
    --optimizer sgd \
    --learning_rate 1e-4 \
    --weight_decay 1e-4 \
    --log_wandb 0 \
    --train 0 \
    --sampler group --uniform_over_groups 1 \
    --n_context_channels ${N_CONTEXT_CHANNELS} \
    "



# python run.py --exp_name erm ${SHARED_ARGS}
# srun python run.py --exp_name ori2_50 --meta_batch_size 2 --support_size 50 --algorithm ARM-CML $SHARED_ARGS

# srun python run.py --algorithm ARM-CML --meta_batch_size 6 --support_size 50 --exp_name ori6_50 $SHARED_ARGS
srun python run.py --eval_on test --test 1 --ckpt_folders femnist_ori2_50_0_20230213-092945 --meta_batch_size 1 --support_size 10 $SHARED_ARGS

# srun python run.py --exp_name cum2_50 --meta_batch_size 2 --support_size 50 --algorithm MY-ARM-CML $SHARED_ARGS
# srun python run.py --eval_on test --test 1 --ckpt_folders femnist_cum2_50_0_20230213-093334 --meta_batch_size 1 --support_size 10 $SHARED_ARGS

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

