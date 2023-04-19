#!/bin/bash

#SBATCH --job-name=new_cusum_small_layer
#SBATCH --gres=gpu:1
#SBATCH --time=0-12:00:00
#SBATCH --mem=16000MB
#SBATCH --cpus-per-task=8
#SBATCH --output=./slurm_log/mnist/S-%x.%j.out

eval "$(conda shell.bash hook)"
conda activate maicon


SEEDS="0 1 2"
SHARED_ARGS="\
    --data_dir ./data/\
    --dataset mnist \
    --train 1 \
    --num_epochs 200 \
    --n_context_channels 12 \
    --eval_on val test \
    --seeds 0 1 2 \
    --dropout_rate 0.3 \
    --model convnet \
    --T 3 \
    --auto 1 \
    --beta 1.0 \
    --epochs_per_eval 10 \
    --smaller_model 1 \
    --mask 0\
    --mask_p 0.4\
    --norm_type layer \
    --normalize 0
    "
    

# Origin: Train (Batch 6 * 50)
srun python run.py --algorithm ARM-CUSUM --meta_batch_size 6 --support_size 50 --exp_name new_cusum_small_layer $SHARED_ARGS






# srun python run.py --algorithm ARM-CML --meta_batch_size 6 --support_size 50 --exp_name cml_not_context_small $SHARED_ARGS
# srun python run_test.py --eval_on test --auto 1 --test 1 --train 0 --online 1 --normalize 1 --ckpt_folders mnist_new_cml_small_norm_m10_0_20230419-214657 mnist_new_cml_small_norm_m10_1_20230419-215758 mnist_new_cml_small_norm_m10_2_20230419-220856




# mnist_new_cml_small_norm_m50_0_20230419-122910 mnist_new_cml_small_norm_m50_1_20230419-124512 mnist_new_cml_small_norm_m50_2_20230419-130036



# mnist_new_cml_small_norm_m30_0_20230419-215643 mnist_new_cml_small_norm_m30_1_20230419-220834 mnist_new_cml_small_norm_m30_2_20230419-222025


# mnist_cusum_small_m50_normalize_0_20230414-190354 mnist_cusum_small_m50_normalize_1_20230414-191452 mnist_cusum_small_m50_normalize_2_20230414-192550


# mnist_cusum_small_normalize_0_20230414-172641 mnist_cusum_small_normalize_1_20230414-173717 mnist_cusum_small_normalize_2_20230414-174752



#  mnist_cusum_small_0_20230414-111749 mnist_cusum_small_1_20230414-112822 mnist_cusum_small_2_20230414-113855


# mnist_arm-cml_0_20230412-232027 mnist_arm-cml_1_20230412-233133 mnist_arm-cml_2_20230412-234236








# srun python run_test.py --eval_on test --auto 1 --test 1 --train 0 --seeds ${SEEDS} --ckpt_folders mnist_cusum_small_0_20230414-111749 mnist_cusum_small_1_20230414-112822 mnist_cusum_small_2_20230414-113855
# python run.py --eval_on test --meta_batch_size 1 --support_size 199 --test 1 --train 0 --seeds 0 1 2 --ckpt_folders mnist_cusum_small_0_20230414-111749 mnist_cusum_small_1_20230414-112822 mnist_cusum_small_2_20230414-113855
# python run.py --eval_on test --meta_batch_size 1 --support_size 1 --test 1 --train 0 --seeds 0 1 2 --ckpt_folders mnist_cusum_small_0_20230414-111749 mnist_cusum_small_1_20230414-112822 mnist_cusum_small_2_20230414-113855




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


