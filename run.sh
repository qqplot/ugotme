#!/bin/bash

#SBATCH --job-name=test-cml-batch-cycle
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
    --n_context_channels 12 \
    --eval_on val test \
    --seeds 0 1 2 \
    --dropout_rate 0.2 \
    --auto 0 \
    --epochs_per_eval 10 \
    --smaller_model 1 \
    --normalize 0\
    --affine_on 0 \
    --num_epochs 200 \
    --beta 1.0 \
    --T 3 \
    --optimizer adam \
    --cxt_self_include 0 \
    --zero_init 1 \
    --meta_batch_size 6 \
    --support_size 50 \
    --bald 0 \
    "
    
# Origin: Train (Batch 6 * 50)
# srun python run.py --algorithm ERM --norm_type batch --learning_rate 1e-4 --model convnet --exp_name erm $SHARED_ARGS
# srun python run.py --algorithm ARM-UNC --sampler standard --uniform_over_groups 0 --norm_type layer --learning_rate 1e-4 --scheduler cosine --model convnet_unc --exp_name unc_softplus_randinit $SHARED_ARGS
# srun python run.py --algorithm ARM-UNC --norm_type layer --learning_rate 1e-4 --scheduler cosine --model convnet_unc --exp_name unc $SHARED_ARGS
# srun python run.py --algorithm ARM-CML --norm_type batch --learning_rate 1e-4 --scheduler none --model convnet --exp_name cml $SHARED_ARGS
# srun python run.py --algorithm ARM-CUSUM --norm_type layer --learning_rate 1e-3 --scheduler cosine --model convnet --exp_name cusum $SHARED_ARGS


# srun python run.py --algorithm ARM-CML --meta_batch_size 6 --support_size 50 --exp_name cml_not_context_small $SHARED_ARGS


# srun python run_test.py --eval_on test --norm_type layer \
# --auto 0 --test 1 --train 0 --online 1 --normalize 0 --T 3 \
# --noisy 1 --bald 0 --save_img 0 \
# --noise_level 0.10 --noise_type sp --cxt_self_include 0 --normal_iter 10 \
# --support_size 50 --zero_context 0 \
# --ckpt_folders mnist_cml_0_20230428-003931 mnist_cml_1_20230428-005003 mnist_cml_2_20230428-010033

# mnist_unc_0_20230517-152619 mnist_unc_1_20230517-163143 mnist_unc_2_20230517-173715




# mnist_cml_0_20230428-003931 mnist_cml_1_20230428-005003 mnist_cml_2_20230428-010033




# mnist_cml_0_20230428-003931 mnist_cml_1_20230428-005003 mnist_cml_2_20230428-010033


python run_test.py --eval_on test --norm_type layer \
--auto 0 --test 1 --train 0 --online 1 --normalize 0 --T 3 \
--noisy 1 --bald 0 --save_img 0 \
--noise_level 0.10 --noise_type sp --cxt_self_include 0 --normal_iter 0 \
--support_size 50 --seeds 0 1 2 \
--ckpt_folders mnist_erm_0_20230428-152600 mnist_erm_1_20230428-153122 mnist_erm_2_20230428-153641

# mnist_cml_0_20230428-003931 mnist_cml_1_20230428-005003 mnist_cml_2_20230428-010033









# mnist_unc_entropy_cosine_inc_0_20230511-010349 mnist_unc_entropy_cosine_inc_1_20230511-020804 mnist_unc_entropy_cosine_inc_2_20230511-031213


# mnist_unc_entropy_cosine_not_inc_0_20230511-001432 mnist_unc_entropy_cosine_not_inc_1_20230511-011820 mnist_unc_entropy_cosine_not_inc_2_20230511-022205

# mnist_unc_entropy_c0_0_20230509-160906 mnist_unc_entropy_c0_1_20230509-171541 mnist_unc_entropy_c0_2_20230509-182206


# mnist_unc_entropy_beta0_0_20230509-023822 mnist_unc_entropy_beta0_1_20230509-034447 mnist_unc_entropy_beta0_2_20230509-045102

# mnist_cusum_layer_fb_lre-3_0_20230428-152725 mnist_cusum_layer_fb_lre-3_1_20230428-154129 mnist_cusum_layer_fb_lre-3_2_20230428-155531
# mnist_unc_layer_fb_learn_cosine_retry_0_20230428-002316 mnist_unc_layer_fb_learn_cosine_retry_1_20230428-013010 mnist_unc_layer_fb_learn_cosine_retry_2_20230428-023702
# mnist_cml_0_20230428-003931 mnist_cml_1_20230428-005003 mnist_cml_2_20230428-010033


# python run_test.py --eval_on test --norm_type layer \
# --auto 1 --test 1 --train 0 --online 1 --normalize 0 --T 3 \
# --ckpt_folders mnist_unc_layer_fb_learn_0_20230427-141534 mnist_unc_layer_fb_learn_1_20230427-152245 mnist_unc_layer_fb_learn_2_20230427-163000




# mnist_cusum_layer_fb_epoch300_0_20230426-051834 mnist_cusum_layer_fb_epoch300_1_20230426-054337 mnist_cusum_layer_fb_epoch300_2_20230426-060842


# mnist_unc_layer_fb_epoch200_cosine_learn_0_20230426-180736 mnist_unc_layer_fb_epoch200_cosine_learn_1_20230426-200559 mnist_unc_layer_fb_epoch200_cosine_learn_2_20230426-220018





# mnist_new_cusum_small_instance_0_20230419-232943 mnist_new_cusum_small_instance_1_20230419-234026 mnist_new_cusum_small_instance_2_20230419-235108




# mnist_new_cml_small_norm_m10_0_20230419-214657 mnist_new_cml_small_norm_m10_1_20230419-215758 mnist_new_cml_small_norm_m10_2_20230419-220856




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


