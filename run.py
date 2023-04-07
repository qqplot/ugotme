import sys, os
from datetime import datetime
from pathlib import Path
import random

import numpy as np
import wandb
import torch

import utils
import train as train
import data as data


# For reproducibility.
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def set_seed(seed, cuda):

    print('setting seed', seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


class ScoreKeeper:

    def __init__(self, splits, n_seeds):

        self.splits = splits
        self.n_seeds = n_seeds

        self.results = {}
        for split in splits:
            self.results[split] = {}

    def log(self, stats):
        for split in stats:
            split_stats = stats[split]
            for key in split_stats:
                value = split_stats[key]
                metric_name = key.split('/')[1]

                if metric_name not in self.results[split]:
                    self.results[split][metric_name] = []

                self.results[split][metric_name].append(value)

    def print_stats(self, metric_names=['worst_case_acc', 'average_acc', 'empirical_acc']):

        for split in self.splits:
            print("Split: ", split)

            for metric_name in metric_names:

                values = np.array(self.results[split][metric_name])
                avg = np.mean(values)
                standard_error = 0
                if self.n_seeds > 1:
                    standard_error =  np.std(values) / np.sqrt(self.n_seeds - 1)

                print(f"{metric_name}: {avg}, standard error: {standard_error}")


def test(args, algorithm, seed, eval_on):

    # Get data
    _, train_eval_loader, val_loader, test_loader = data.get_loaders(args)

    stats = {}
    loaders = {'train': train_eval_loader,
                'val': val_loader,
                'test': test_loader}
    for split in eval_on:
        set_seed(seed + 10, args.cuda)
        loader = loaders[split]
        split_stats = train.eval_groupwise(args, algorithm, loader, split=split, n_samples_per_group=args.test_n_samples_per_group)
        stats[split] = split_stats

    return stats


def main():
    parser = utils.make_arm_train_parser()
    args = parser.parse_args()

    if args.auto:
        utils.update_arm_parser(args)

    args.cuda, args.device = utils.get_device_from_arg(args.device_id)
    print('Using device:', args.device)

    algorithm = utils.init_algorithm(args) 
    print('Args', '-'*50, '\n', args, '\n', '-'*50)

    start_time = datetime.now()

    if args.train:

        score_keeper = ScoreKeeper(args.eval_on, len(args.seeds))
        print("args seeds: ", args.seeds)
        ckpt_dirs = []

        for ind, seed in enumerate(args.seeds):
            print("seeeed: ", seed)
            set_seed(seed, args.cuda)
            tags = ['supervised', args.dataset, args.algorithm]

            # Save folder
            datetime_now = datetime.now().strftime("%Y%m%d-%H%M%S")
            name = args.dataset + '_' + args.exp_name + '_' + str(seed)
            args.ckpt_dir = Path('output') / 'checkpoints' / f'{name}_{datetime_now}'
            ckpt_dirs.append(args.ckpt_dir)
            print("CHECKPOINT DIR: ", args.ckpt_dir)

            if args.debug: 
                tags.append('debug')

            if args.log_wandb:
                if ind != 0:
                    wandb.join()
                run = wandb.init(name=name,
                                 project=f"arm_{args.dataset}",
                                 tags=tags,
                                 allow_val_change=True,
                                 reinit=True)
                wandb.config.update(args, allow_val_change=True)

            train.train(args, algorithm)

            # Test the model just trained on
            if args.test:
                args.ckpt_path = args.ckpt_dir / f'best.pkl'
                algorithm = torch.load(args.ckpt_path).to(args.device)
                stats = test(args, algorithm, seed, eval_on=args.eval_on)
                score_keeper.log(stats)

        print("Ckpt dirs: \n ", ckpt_dirs)
        score_keeper.print_stats()

    elif args.test and args.ckpt_folders: # test a set of already trained models

        # Check if checkpoints exist
        for ckpt_folder in args.ckpt_folders:
            ckpt_path = Path('output') / 'checkpoints' / ckpt_folder / f'best.pkl'
            algorithm = torch.load(ckpt_path)
            print("Found: ", ckpt_path)

        score_keeper = ScoreKeeper(args.eval_on, len(args.ckpt_folders))
        for i, ckpt_folder in enumerate(args.ckpt_folders):

            # test algorithm
            seed = args.seeds[i]
            args.ckpt_path = Path('output') / 'checkpoints' / ckpt_folder / f'best.pkl' # final_weights.pkl
            algorithm = torch.load(args.ckpt_path).to(args.device)
            stats = test(args, algorithm, seed, eval_on=args.eval_on)
            score_keeper.log(stats)

        score_keeper.print_stats()


    end_time = datetime.now()
    runtime = (end_time - start_time).total_seconds() / 60.0
    print("\nTotal runtime: ", runtime)

    return





if __name__ == '__main__':

    main()




