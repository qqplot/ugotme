
# Uncertainty-Guided Online Test-time Adaptation via Meta-Learning

***ICML Workshop 2023 on Spurious Correlations, Invariance and Stability***
* Poster: [poster](https://icml.cc/media/PosterPDFs/ICML%202023/26325.png?t=1689562620.9608736)



The structure of this repo and the way certain details around the training loop and evaluation loop is set up is inspired by and adapted from the [DomainBed repo](https://github.com/facebookresearch/DomainBed/tree/main/domainbed), the [Wilds repo](https://github.com/p-lambda/wilds), and the [ARM](https://github.com/henrikmarklund/arm).

* Environment
* Logging Results
* Experiments Setup
    * Train
    * Evaluate

## Environment

python version: 3.8

Using pip
 - `pip install -r requirements.txt` or `pip3 install -r requirements.txt`

## Logging results.
Weights and Biases, which is an alternative to Tensorboard, is used to log results in the cloud. This is used for both training and evaluating the test set.
To get it running quickly without WandB, we have set --log_wandb 0 below. Much of the results will be printed in the console. We recommend using WandB which is free for researchers.

## Data

Femnist
The train/val/test data split used in the paper can be found here: https://drive.google.com/file/d/1xvT13Sl3vJIsC2I7l7Mp8alHkqKQIXaa/view?usp=sharing

## Experiments Setup

Showing example args for MNIST here. See `run_train.sh` and `run_test.sh` (See all_commands.sh for ARM).

### 1. Train

##### Shared args
```
SEEDS="0"
SHARED_ARGS="--dataset mnist --num_epochs 200 --n_samples_per_group 300 --epochs_per_eval 10 --seeds ${SEEDS} --meta_batch_size 6 --epochs_per_eval 10 --log_wandb 0 --train 1"
```

##### ERM
```
python run.py --exp_name erm $SHARED_ARGS
```

##### ARM-CML (Adaptive Risk Minimization - Contextual Meta-learner)
```
python run.py --algorithm ARM-CML --sampler group --uniform_over_groups 1 --n_context_channels 12 --exp_name arm_cml $SHARED_ARGS
```

##### ARM-CUSUM (Adaptive Risk Minimization - Contextual Meta-learner with Cumulative sum)
```
python run.py --algorithm ARM-CML --sampler group --uniform_over_groups 1 --n_context_channels 12 --exp_name arm_cml $SHARED_ARGS
```

##### ARM-UNC (Ours)
```
python run.py --algorithm ARM-CML --sampler group --uniform_over_groups 1 --n_context_channels 12 --norm_type layer --model convnet_unc --exp_name unc $SHARED_ARGS
```



### 2. Evaluate

Your trained models are saved in `output/checkpoints/{dataset}_{exp_name}_{seed}_{datetime}/`

An example of checkpoint could be:
- `output/checkpoints/mnist_erm_0_20230529-140707/best_weights.pkl`

To evaluate a set of checkpoints, you run:
```
python run.py --eval_on test --test 1 --train 0 --ckpt_folders CKPT_FOLDER1 CKPT_FOLDER2 CKPT_FOLDER3 --log_wandb 0`
```

E.g., you could run
```
python run.py --eval_on test --test 1 -- train 0 --ckpt_folders mnist_erm_0_1231414 mnist_erm_1_1231434 mnist_erm_2_2_1231414 --log_wandb 0`
```

`--ckpt_folders` is a list of the folders

You can vary support size with `--support_size`.


For the online settings, you run:
```
python run_test.py --eval_on test --test 1 --train 0 --online 1 --ckpt_folders CKPT_FOLDER1 CKPT_FOLDER2 CKPT_FOLDER3 --log_wandb 0`
```


## Citation

If you find this codebase useful in your research, consider citing:

```
coming soon
```
