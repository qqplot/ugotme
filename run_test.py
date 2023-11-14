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
    # torch.use_deterministic_algorithms(True)
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

    return stats, loaders

def test_with_context(args, algorithm, seed, eval_on):

    # Get data
    _, train_eval_loader, val_loader, test_loader = data.get_loaders(args)

    stats = {}
    loaders = {'train': train_eval_loader,
                'val': val_loader,
                'test': test_loader}

    for split in eval_on:
        set_seed(seed + 10, args.cuda)
        loader = loaders[split]
        split_stats = train.eval_groupwise_with_context(args, algorithm, loader, split=split, n_samples_per_group=args.test_n_samples_per_group)
        stats[split] = split_stats

    return stats, loaders


def test_zero(args, algorithm, seed, eval_on, loaders):

    stats = {}

    for split in eval_on:
        set_seed(seed + 10, args.cuda)
        loader = loaders[split]
        split_stats = train.eval_groupwise(args, algorithm, loader, split=split, n_samples_per_group=args.test_n_samples_per_group)
        stats[split] = split_stats

    return stats


def test_online_noise(args):

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
    # args.support_size = 100
    args.seeds = [0, 1, 2]

    # Check if checkpoints exist
    for ckpt_folder in args.ckpt_folders:
        ckpt_path = Path('output') / 'checkpoints' / ckpt_folder / f'best.pkl'
        algorithm = torch.load(ckpt_path)
        print("Found: ", ckpt_path)

    score_keeper = ScoreKeeper(args.eval_on, len(args.ckpt_folders))
    
    avg_online_acc = []
    avg_weights = []
    avg_stds = []
    for i, ckpt_folder in enumerate(args.ckpt_folders):

        # test algorithm
        seed = args.seeds[i]
        args.ckpt_path = Path('output') / 'checkpoints' / ckpt_folder / f'best.pkl' # final_weights.pkl

        algorithm = utils.init_algorithm(args) 
        print('Args', '-'*50, '\n', args, '\n', '-'*50)
        algorithm = torch.load(args.ckpt_path).to(args.device)
        algorithm.support_size = args.support_size
        algorithm.normalize = args.normalize
        algorithm.online = args.online
        algorithm.T = args.T
        algorithm.adapt_bn = args.adapt_bn
        algorithm.cxt_self_include = args.cxt_self_include
        algorithm.zero_init = args.zero_init
        algorithm.bald = args.bald
        algorithm.zero_context = args.zero_context
        algorithm.is_offline = args.is_offline

        if args.norm_type == 'batch':
            algorithm.context_norm = None
        
        stats, loaders = test(args, algorithm, seed, eval_on=args.eval_on)

        online_sum = [0 for _ in range(args.support_size)]
        online_len = [0 for _ in range(args.support_size)]
        weights = [0 for _ in range(args.support_size)]
        stds = [0 for _ in range(args.support_size)]

        for idx, acc in enumerate(stats['test']['test/online_acc']):
            whts = stats['test']['test/weights'][idx]
            standard_errors = stats['test']['test/standard_errors'][idx]
            for i, a in enumerate(acc):
                online_sum[i] += a 
                online_len[i] += 1
                if algorithm.__class__.__name__[-3:] == 'UNC':
                    if type(whts) != list:
                        weights[i] += whts
                        stds[i] += standard_errors
                    else:
                        weights[i] += whts[i]
                        stds[i] += standard_errors[i]

        online_acc = [online_sum[i]/online_len[i] for i in range(args.support_size)]

        if algorithm.__class__.__name__[-3:] == 'UNC':
            weights = [weights[i]/online_len[i] for i in range(args.support_size)]
            stds = [stds[i]/online_len[i] for i in range(args.support_size)]

        avg_online_acc.append(online_acc)
        avg_weights.append(weights)
        avg_stds.append(stds)
    
        print("length:", len(stats['test']['test/online_acc']),"online_acc:", online_acc[:10])        
        if algorithm.__class__.__name__[-3:] == 'UNC':
            print("length:", len(stats['test']['test/weights']), "weights:", weights[:10]) 
            print("length:", len(stats['test']['test/standard_errors']), "standard_errors:", stds[:10])        

        score_keeper.log(stats)


    avg_online_acc = np.array(avg_online_acc).mean(axis=0)


    avg_weights = np.array(avg_weights).mean(axis=0)
    avg_stds = np.array(avg_stds).mean(axis=0)

    print("\nsupport size is", args.support_size)        
    print(avg_online_acc.tolist())

    print("online weights..")
    print(avg_weights.tolist())

    print("online standard errors..")
    print(avg_stds.tolist())

    score_keeper.print_stats()

    end_time = datetime.now()
    runtime = (end_time - start_time).total_seconds() / 60.0
    print("\nTotal runtime: ", runtime)

def test_context(args):

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
    # args.support_size = 100
    args.seeds = [0, 1, 2]

    # Check if checkpoints exist
    for ckpt_folder in args.ckpt_folders:
        ckpt_path = Path('output') / 'checkpoints' / ckpt_folder / f'best.pkl'
        algorithm = torch.load(ckpt_path)
        print("Found: ", ckpt_path)

    score_keeper = ScoreKeeper(args.eval_on, len(args.ckpt_folders))
    
    avg_online_acc = []
    avg_weights = []
    avg_stds = []
    for i, ckpt_folder in enumerate(args.ckpt_folders):
    # for i, seed in enumerate(args.seeds):
        # test algorithm
        seed = args.seeds[i]
        # ckpt_folder = args.ckpt_folders[0]
        args.ckpt_path = Path('output') / 'checkpoints' / ckpt_folder / f'best.pkl' # final_weights.pkl

        algorithm = utils.init_algorithm(args) 
        print('Args', '-'*50, '\n', args, '\n', '-'*50)
        algorithm = torch.load(args.ckpt_path).to(args.device)
        algorithm.support_size = args.support_size
        algorithm.normalize = args.normalize
        algorithm.online = args.online
        algorithm.T = args.T
        algorithm.adapt_bn = args.adapt_bn
        algorithm.cxt_self_include = args.cxt_self_include
        algorithm.zero_init = args.zero_init
        algorithm.bald = args.bald
        algorithm.zero_context = args.zero_context
        algorithm.is_offline = args.is_offline

        if args.norm_type == 'batch':
            algorithm.context_norm = None
        
        stats, loaders = test_with_context(args, algorithm, seed, eval_on=args.eval_on)
        acc_group_ep_ctx = np.array(stats['test']['test/average_acc']) # [group, ep, 50]
        np.save(f'unseen_acc_{ckpt_folder}.npy', acc_group_ep_ctx)

        acc_group_ep_ctx_seen = np.array(stats['test']['test/average_acc_seen']) 
        np.save(f'seen_acc_{ckpt_folder}.npy', acc_group_ep_ctx_seen)

    print("\nsupport size is", args.support_size)        

    end_time = datetime.now()
    runtime = (end_time - start_time).total_seconds() / 60.0
    print("\nTotal runtime: ", runtime)

if __name__ == '__main__':
    # For reproducibility.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    parser = utils.make_arm_train_parser()
    args = parser.parse_args()
    if args.ctx_test:
        test_context(args)
    else:
        test_online_noise(args)



