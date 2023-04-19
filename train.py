import os
import random

import numpy as np
import torch
import torch.nn as nn
from tqdm import trange, tqdm
import wandb

import data
import utils


os.environ['KMP_DUPLICATE_LIB_OK'] = "TRUE"


####################
###### TRAIN #######
####################
def run_epoch(algorithm, loader, train, progress_bar=True, mask=None, mask_p=1.0):

    epoch_labels = []
    epoch_logits = []
    epoch_group_ids = []

    if progress_bar:
        loader = tqdm(loader, desc=f'{"train" if train else "eval"} loop')

    for images, labels, group_ids in loader:

        # Put on GPU
        images = images.to(algorithm.device)
        labels = labels.to(algorithm.device)

        # Forward
        if train:
            if mask:
                logits, batch_stats = algorithm.learn(images, labels, group_ids, mask, mask_p)
            else:
                logits, batch_stats = algorithm.learn(images, labels, group_ids)
            if logits is None: # DANN
                continue
        else:
            logits = algorithm.predict(images)

        epoch_labels.append(labels.to('cpu').clone().detach())
        epoch_logits.append(logits.to('cpu').clone().detach())
        epoch_group_ids.append(group_ids.to('cpu').clone().detach())

    return torch.cat(epoch_logits), torch.cat(epoch_labels), torch.cat(epoch_group_ids), epoch_logits, epoch_labels, epoch_group_ids

def train(args, algorithm):

    # Get data
    train_loader, _, val_loader, _ = data.get_loaders(args)
    args.n_groups = train_loader.dataset.n_groups

    # algorithm = init_algorithm(args, train_loader.dataset)
    saver = utils.Saver(algorithm, args.device, args.ckpt_dir)

    # Train loop
    best_worst_case_acc = 0
    best_average_acc = 0

    for epoch in trange(args.num_epochs):
        _, _, _, epoch_logits, epoch_labels, epoch_group_ids = run_epoch(algorithm, train_loader, train=True, progress_bar=args.progress_bar, mask=args.mask, mask_p=args.mask_p)

        if epoch % args.epochs_per_eval == 0:
            stats = eval_groupwise(args, algorithm, val_loader, epoch, split='val', n_samples_per_group=args.n_samples_per_group)


            if args.worst_case:
                # Track early stopping values with respect to worst case.
                if stats['val/worst_case_acc'] > best_worst_case_acc:
                    best_worst_case_acc = stats['val/worst_case_acc']
                    saver.save(epoch, is_best=True)
            else:
                # Track early stopping values with respect to worst case.
                if stats['val/average_acc'] > best_average_acc:
                    best_average_acc = stats['val/average_acc']
                    saver.save(epoch, is_best=True)

            # Log early stopping values
            if args.log_wandb:
                wandb.log({"val/best_worst_case_acc": best_worst_case_acc})

            print(f"\nEpoch: ", epoch, "\nWorst Case Acc: ", stats['val/worst_case_acc'], "Average Acc: ", stats['val/average_acc'])

##############################
###### Evaluate / Test #######
##############################

def get_group_iterator(loader, group, support_size, n_samples_per_group=None):
    example_ids = np.nonzero(loader.dataset.group_ids == group)[0]
    # print("example_ids", len(example_ids)) # 3333
    example_ids = example_ids[np.random.permutation(len(example_ids))] # Shuffle example ids

    # Create batches
    batches = []
    X, Y, G = [], [], []
    counter = 0
    for i, idx in enumerate(example_ids):
        x, y, g = loader.dataset[idx]
        X.append(x); Y.append(y); G.append(g)
        if (i + 1) % support_size == 0:
            X, Y, G = torch.stack(X), torch.tensor(Y, dtype=torch.long), torch.tensor(G, dtype=torch.long)
            batches.append((X, Y, G))
            X, Y, G = [], [], []

        if n_samples_per_group is not None and i == (n_samples_per_group - 1):
            break
    if X:
        X, Y, G = torch.stack(X), torch.tensor(Y, dtype=torch.long), torch.tensor(G, dtype=torch.long)
        batches.append((X, Y, G))

    return batches


def get_group_iterator_noisy(args, loader, group, support_size, n_samples_per_group=None):
    example_ids = np.nonzero(loader.dataset.group_ids == group)[0]
    
    example_ids_not_group = np.nonzero(loader.dataset.group_ids != group)[0]
    example_ids_not_group = np.random.choice(example_ids_not_group, len(example_ids), replace=False)
    
    example_ids = example_ids[np.random.permutation(len(example_ids))] # Shuffle example ids

    # Create batches
    batches = []
    X, Y, G = [], [], []
    X_noise, Y_noise, G_noise = [], [], []
    num_noise = args.num_noise
    if args.num_noise >= support_size:
        print(f"num_noise is {args.num_noise} and support_size is {support_size}!!!")
        num_noise = support_size

    for i, (idx, idx_not_group) in enumerate(zip(example_ids, example_ids_not_group)):
        x, y, g = loader.dataset[idx]
        x_noise, y_noise, g_noise = loader.dataset[idx_not_group]     

        X.append(x); Y.append(y); G.append(g)
        X_noise.append(x_noise); Y_noise.append(y_noise); G_noise.append(g_noise)
        
        if (i + 1) % support_size == 0:
            X[len(X)-num_noise:] = X_noise[:num_noise]
            Y[len(Y)-num_noise:] = Y_noise[:num_noise]
            G[len(G)-num_noise:] = G_noise[:num_noise]

            if args.noise_type == 'front':               
                X = X[len(X)-num_noise:].copy() + X_noise[:len(X)-num_noise]
                Y = Y[len(Y)-num_noise:].copy() + Y_noise[:len(Y)-num_noise]
                G = G[len(G)-num_noise:].copy() + G_noise[:len(G)-num_noise]

            elif args.noise_type == 'random':
                randIdx = np.arange(len(X))
                np.random.shuffle(randIdx)

                X = list(np.array(X)[randIdx])
                Y = list(np.array(Y)[randIdx])
                G = list(np.array(G)[randIdx])

            X, Y, G = torch.stack(X), torch.tensor(Y, dtype=torch.long), torch.tensor(G, dtype=torch.long)
            batches.append((X, Y, G))
            X, Y, G = [], [], []
            X_noise, Y_noise, G_noise = [], [], []

        if n_samples_per_group is not None and i == (n_samples_per_group - 1):
            break
    if X:
        X, Y, G = torch.stack(X), torch.tensor(Y, dtype=torch.long), torch.tensor(G, dtype=torch.long)
        batches.append((X, Y, G))

    return batches

def eval_groupwise(args, algorithm, loader, epoch=None, split='val', n_samples_per_group=None):
    """ Test model on groups and log to wandb

        Separate script for femnist for speed."""

    groups = []
    accuracies = np.zeros(len(loader.dataset.groups))
    num_examples = np.zeros(len(loader.dataset.groups))
    online_accuracies = []

    if args.adapt_bn:
        algorithm.train()
    else:
        algorithm.eval()

    # Loop over each group
    for i, group in tqdm(enumerate(loader.dataset.groups), desc='Evaluating', total=len(loader.dataset.groups)):
        
        if args.noisy:
            num_noise = 1 if args.num_noise else args.num_noise
            group_iterator = get_group_iterator_noisy(args, loader, group, args.support_size, n_samples_per_group)
        else:
            group_iterator = get_group_iterator(loader, group, args.support_size, n_samples_per_group)

        probs, labels, group_ids, epoch_logits, epoch_labels, epoch_group_ids = run_epoch(algorithm, group_iterator, train=False, progress_bar=False)
        preds = np.argmax(probs, axis=1)

        # Evaluate
        if args.test and args.online:

            for i, logits in enumerate(epoch_logits):
                preds_online = np.argmax(logits, axis=1)
                label = epoch_labels[i]
                online_accuracies.append((preds_online == label).numpy().astype(int)) 
                       
        accuracy = np.mean((preds == labels).numpy())
        num_examples[group] = len(labels)
        accuracies[group] = accuracy

        if args.log_wandb:
            if epoch is None:
                wandb.log({f"{split}/acc": accuracy, # Gives us Acc vs Group Id
                           f"{split}/group_id": group})
            else:
                wandb.log({f"{split}/acc_e{epoch}": accuracy, # Gives us Acc vs Group Id
                           f"{split}/group_id": group})

    # Log worst, average and empirical accuracy
    worst_case_acc = np.amin(accuracies)
    worst_case_group_size = num_examples[np.argmin(accuracies)]

    num_examples = np.array(num_examples)
    props = num_examples / num_examples.sum()
    empirical_case_acc = accuracies.dot(props)
    average_case_acc = np.mean(accuracies)

    total_size = num_examples.sum()

    stats = {
                f'{split}/worst_case_acc': worst_case_acc,
                f'{split}/worst_case_group_size': worst_case_group_size,
                f'{split}/average_acc': average_case_acc,
                f'{split}/total_size': total_size,
                f'{split}/empirical_acc': empirical_case_acc,
                f'{split}/online_acc': online_accuracies
            }

    if epoch is not None:
        stats['epoch'] = epoch

    if args.log_wandb:
        wandb.log(stats)

    return stats

