import sys, os
from datetime import datetime
from pathlib import Path
import random

import numpy as np
import wandb
import torch

import utils
from utils import ScoreKeeper
import train as train
import data as data





def set_seed(seed, cuda):

    print('setting seed', seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


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

def test_online():
    parser = utils.make_arm_train_parser()
    args = parser.parse_args()

    if args.auto:
        utils.update_arm_parser(args)

    if not (args.test and args.ckpt_folders): # test a set of already trained models
        print("Check args.test and args.ckpt_folders!!!")
        return

    args.cuda, args.device = utils.get_device_from_arg(args.device_id)
    print('Using device:', args.device)
    start_time = datetime.now()

    # Online test
    args.meta_batch_size = 1
    args.support_size = 100
    args.seeds = [0, 1, 2]

    algorithm = utils.init_algorithm(args) 
    print('Args', '-'*50, '\n', args, '\n', '-'*50)

    # Check if checkpoints exist
    for ckpt_folder in args.ckpt_folders:
        ckpt_path = Path('output') / 'checkpoints' / ckpt_folder / f'best.pkl'
        algorithm = torch.load(ckpt_path)
        print("Found: ", ckpt_path)

    score_keeper = ScoreKeeper(args.eval_on, len(args.ckpt_folders))

    avg_online_acc = []
    for i, ckpt_folder in enumerate(args.ckpt_folders):

        # test algorithm
        seed = args.seeds[i]
        args.ckpt_path = Path('output') / 'checkpoints' / ckpt_folder / f'best.pkl' # final_weights.pkl
        algorithm = torch.load(args.ckpt_path).to(args.device)
        algorithm.support_size = args.support_size
        algorithm.normalize = args.normalize
        algorithm.context_norm = None
        
        stats = test(args, algorithm, seed, eval_on=args.eval_on)
        
        # print('online_acc:', stats['test']['test/online_acc'])
        # print(stats['test']['test/online_acc'].shape)
        online_sum = [0 for _ in range(args.support_size)]
        online_len = [0 for _ in range(args.support_size)]
        for _, acc in enumerate(stats['test']['test/online_acc']):

            for i, a in enumerate(acc):
                online_sum[i] += a 
                online_len[i] += 1
            
        online_acc = [online_sum[i]/online_len[i] for i in range(args.support_size)]
        avg_online_acc.append(online_acc)
        print("length:", len(stats['test']['test/online_acc']),"online_acc:", online_acc[:10])        
        score_keeper.log(stats)

    avg_online_acc = np.array(avg_online_acc).mean(axis=0)
    print("\nsupport size is", args.support_size)        
    print(avg_online_acc.tolist())

    score_keeper.print_stats()

    end_time = datetime.now()
    runtime = (end_time - start_time).total_seconds() / 60.0
    print("\nTotal runtime: ", runtime)
    




if __name__ == '__main__':
    # For reproducibility.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    test_online()




