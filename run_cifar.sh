#!/bin/bash

#SBATCH --job-name=cusum
#SBATCH --gres=gpu:1
#SBATCH --mem=16000MB
#SBATCH --cpus-per-task=8
#SBATCH --output=./slurm_log/cifar10/S-%x.%j.out
#SBATCH --time=0-12:00:00


eval "$(conda shell.bash hook)"
conda activate maicon

# CIFAR-C
SEEDS="0 1 2"
SHARED_ARGS="\
    --data_dir ./data/\
    --dataset cifar \
    --n_samples_per_group 2000 \
    --test_n_samples_per_group 3000 \
    --n_context_channels 3 \
    --eval_on val test \
    --seeds ${SEEDS} \
    --meta_batch_size 1 \
    --support_size 128 \
    --dropout_rate 0.2 \
    --T 3 \
    --epochs_per_eval 5 \
    --num_epochs 200 \
    --sampler standard \
    --uniform_over_groups 0 \
    --drop_last 1 \
    --learning_rate 0.1 \
    --weight_decay 0.0005 \
    "

# srun python run.py --algorithm ERM --smaller_model 0 --norm_type batch --optimizer sgd --model resnet18 --exp_name erm $SHARED_ARGS
# srun python run.py --algorithm ARM-CML --norm_type batch  --optimizer sgd --model resnet18 --exp_name cml $SHARED_ARGS
# srun python run.py --algorithm ARM-UNC --norm_type layer --scheduler cosine_warm --optimizer sgd --model resnet18_unc --exp_name unc $SHARED_ARGS
# srun python run.py --algorithm ARM-CUSUM --norm_type layer --scheduler cosine_warm --optimizer sgd --learning_rate 0.03 --weight_decay 1e-3 --exp_name cusum $SHARED_ARGS


# srun python run.py --exp_name cusum_small_layer_aff --algorithm ARM-CUSUM $SHARED_ARGS


# srun python run_test.py --eval_on test --dataset cifar-c --data_dir ./data/ \
# --norm_type layer \
# --auto 0 --test 1 --train 0 --online 1 --normalize 0 --T 3 \
# --noisy 1 --bald 0 --save_img 1 \
# --noise_level 0.10 --noise_type gaussian --cxt_self_include 0 \
# --ckpt_folders cifar-c_cml_0_20230428-012431 cifar-c_cml_1_20230428-013652 cifar-c_cml_2_20230428-014909




# cifar-c_cusum_small_instance_0_20230423-111648 cifar-c_cusum_small_instance_1_20230423-112635 cifar-c_cusum_small_instance_2_20230423-113621

# python run.py --eval_on test --dataset cifar-c --data_dir ./data/ --seeds 0 1 2 \
# --norm_type layer --auto 1 --test 1 --train 0 --online 0 --normalize 0 --adapt_bn 1 \
# --ckpt_folders cifar-c_cml_0_20230428-012431 cifar-c_cml_1_20230428-013652 cifar-c_cml_2_20230428-014909




# cifar-c_cusum_small_layer_0_20230423-110543 cifar-c_cusum_small_layer_1_20230423-111542 cifar-c_cusum_small_layer_2_20230423-112540




# cifar-c_cusum_small_div_0_20230423-110513 cifar-c_cusum_small_div_1_20230423-111524 cifar-c_cusum_small_div_2_20230423-112532




# cifar-c_cusum_small_0_20230423-110444 cifar-c_cusum_small_1_20230423-111450 cifar-c_cusum_small_2_20230423-112452


# cifar-c_cml_small_0_20230423-110415 cifar-c_cml_small_1_20230423-111411 cifar-c_cml_small_2_20230423-112406



# cifar-c_cml_large_0_20230410-191552 cifar-c_cml_large_1_20230410-193350 cifar-c_cml_large_2_20230410-195144