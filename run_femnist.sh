#!/bin/bash

#SBATCH --job-name=test-cml-always-lv1
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=16000MB
#SBATCH --cpus-per-task=16
#SBATCH --output=./slurm_log/femnist/S-%x.%j.out
#SBATCH --time=0-12:00:00

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
    --train 1 \
    --sampler group \
    --uniform_over_groups 1 \
    --n_context_channels ${N_CONTEXT_CHANNELS} \
    --smaller_model 1 \
    --epochs_per_eval 10 \
    --dropout_rate 0.2 \
    --meta_batch_size 2 \
    --support_size 50 \
    --T 3 \
    "

# srun python run.py --algorithm ERM --norm_type batch --optimizer sgd --learning_rate 1e-4 --weight_decay 1e-4 --exp_name erm $SHARED_ARGS
# srun python run.py --algorithm ARM-UNC --norm_type layer --optimizer adam --learning_rate 1e-4 --scheduler cosine --model convnet_unc --exp_name unc $SHARED_ARGS
# srun python run.py --algorithm ARM-CML --norm_type batch --optimizer sgd --learning_rate 1e-4 --weight_decay 1e-4 --exp_name cml $SHARED_ARGS
# srun python run.py --algorithm ARM-CUSUM --norm_type layer --optimizer sgd --learning_rate 1e-4 --scheduler cosine --exp_name cusum $SHARED_ARGS
# srun python run.py --algorithm ARM-CUSUM --norm_type layer --optimizer adam --scheduler cosine --learning_rate 1e-4 --exp_name cusum $SHARED_ARGS


    
# srun python run_test.py --eval_on test --dataset femnist --data_dir ./data/ \
# --norm_type layer --auto 1 --test 1 --train 0 --online 1 --normalize 0 \
# --ckpt_folders femnist_unc_0_20230514-212136 femnist_unc_1_20230515-015733 femnist_unc_2_20230515-063419



srun python run_test.py --eval_on test --dataset femnist --data_dir ./data/ \
--norm_type layer \
--auto 0 --test 1 --train 0 --online 1 --normalize 0 --T 3 \
--noisy 1 --bald 0 --save_img 1 \
--noise_level 0.05 --noise_type sp --cxt_self_include 0 --normal_iter 10 \
--support_size 50 --seeds 0 1 2 --zero_context 0 \
--ckpt_folders femnist_cml_0_20230428-012013 femnist_cml_1_20230428-014337 femnist_cml_2_20230428-020715

# femnist_cml_0_20230428-012013 femnist_cml_1_20230428-014337 femnist_cml_2_20230428-020715

# femnist_unc_0_20230517-154033 femnist_unc_1_20230517-200644 femnist_unc_2_20230518-003301






# femnist_unc_0_20230517-154033 femnist_unc_1_20230517-200644 femnist_unc_2_20230518-003301






python run.py --eval_on test --dataset femnist --data_dir ./data/ \
--norm_type layer \
--auto 0 --test 1 --train 0 --online 0 --normalize 0 --T 3 \
--noisy 1 --bald 0 --save_img 0 \
--noise_level 0.05 --noise_type sp --cxt_self_include 0 --normal_iter 10 \
--support_size 50 --seeds 0 1 2 \
--ckpt_folders femnist_cml_0_20230428-012013 femnist_cml_1_20230428-014337 femnist_cml_2_20230428-020715











# python run_test.py --eval_on test --dataset femnist --data_dir ./data/ \
# --norm_type batch --auto 1 --test 1 --train 0 --online 1 --normalize 0 \
# --ckpt_folders femnist_new_cusum_small_0_20230420-100725 femnist_new_cusum_small_1_20230420-102928 femnist_new_cusum_small_2_20230420-105146







# srun python run_test.py --data_dir ./data/ --dataset femnist --eval_on test --auto 1 --test 1 --train 0 --ckpt_folders femnist_cusum_small_norm_m20_0_20230415-103552 femnist_cusum_small_norm_m20_1_20230415-105925 femnist_cusum_small_norm_m20_2_20230415-112320




# femnist_cusum_small_norm_m30_0_20230415-013346 femnist_cusum_small_norm_m30_1_20230415-015656 femnist_cusum_small_norm_m30_2_20230415-022025


# femnist_cusum_small_norm_m50_0_20230415-000557 femnist_cusum_small_norm_m50_1_20230415-002912 femnist_cusum_small_norm_m50_2_20230415-005246




# femnist_cusum_small_m50_0_20230415-000535 femnist_cusum_small_m50_1_20230415-002755 femnist_cusum_small_m50_2_20230415-005029



# femnist_cusum_small_0_20230415-000453 femnist_cusum_small_1_20230415-002638 femnist_cusum_small_2_20230415-004836

# femnist_cml_small_0_20230415-000453 femnist_cml_small_1_20230415-002748 femnist_cml_small_2_20230415-005044


# femnist_cml_large_0_20230409-000146 femnist_cml_large_1_20230409-003535 femnist_cml_large_2_20230409-010937

# python run.py --exp_name erm ${SHARED_ARGS}
# srun python run.py --exp_name ori2_50 --meta_batch_size 2 --support_size 50 --algorithm ARM-CML $SHARED_ARGS

# srun python run.py --algorithm ARM-CML --meta_batch_size 6 --support_size 50 --exp_name ori6_50 $SHARED_ARGS
# srun python run.py --eval_on test --test 1 --ckpt_folders femnist_ori2_50_0_20230213-092945 --meta_batch_size 1 --support_size 10 $SHARED_ARGS

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

