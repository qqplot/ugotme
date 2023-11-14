import os
import random

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, CosineAnnealingWarmRestarts
import torch.nn.functional as F
from tqdm import trange, tqdm
import utils
import wandb

import data
import utils

from skimage.util import random_noise
from torchvision.utils import save_image

os.environ['KMP_DUPLICATE_LIB_OK'] = "TRUE"


####################
###### TRAIN #######
####################
def run_epoch(algorithm, loader, train, progress_bar=True, mask=None, mask_p=1.0):

    epoch_labels = []
    epoch_logits = []
    epoch_group_ids = []
    epoch_results = {
        'epoch_u_list' : [],
        'epoch_std_list' : [],
    }

    if progress_bar:
        loader = tqdm(loader, desc=f'{"train" if train else "eval"} loop')

    for idx, (images, labels, group_ids) in enumerate(loader):

        # Put on GPU
        images = images.to(algorithm.device)
        labels = labels.to(algorithm.device)

        # Forward
        if train:
            logits, batch_stats = algorithm.learn(images, labels, group_ids)
            if logits is None: # DANN
                continue
        else:
            if algorithm.__class__.__name__[-3:] == 'UNC':           
                logits, u_list, std_list = algorithm.predict(images)
            else:
                logits = algorithm.predict(images)
                u_list, std_list = None, None
            epoch_results['epoch_u_list'].append(u_list)
            epoch_results['epoch_std_list'].append(std_list)

            
        epoch_labels.append(labels.to('cpu').clone().detach())
        epoch_logits.append(logits.to('cpu').clone().detach())
        epoch_group_ids.append(group_ids.to('cpu').clone().detach())
        epoch_results['epoch_logits'] = epoch_logits
        epoch_results['epoch_labels'] = epoch_labels
        epoch_results['epoch_group_ids'] = epoch_group_ids

    return torch.cat(epoch_logits), torch.cat(epoch_labels), torch.cat(epoch_group_ids), epoch_results

def run_epoch_with_context(algorithm, loader, context):

    epoch_labels, epoch_logits, epoch_group_ids = [], [], []
    epoch_u, epoch_std = [], []
    first_batch_size = None

    for idx, (images, labels, group_ids) in enumerate(loader): # episode

        # Put on GPU
        images = images.to(algorithm.device)
        labels = labels.to(algorithm.device)
        
        # (batch_size, channels, H, W)

        # Forward
        batch_size, c, h, w = images.shape
        # first_batch_size = batch_size if first_batch_size is None else first_batch_size
        # if batch_size != first_batch_size: continue
        # print(f"first_context:{first_context.size()}, images:{images.size()}")

        repeated_context = context.unsqueeze(0).repeat_interleave(repeats=batch_size, dim=0).to(algorithm.device) # (batch_size * batch_size, channels, H, W)
        logits, u_list, std_list = algorithm.predict_with_context(images, context=repeated_context)

        epoch_labels.append(labels.to('cpu').clone().detach())
        epoch_logits.append(logits.to('cpu').clone().detach())
        epoch_u.append(u_list)
        epoch_std.append(std_list)

    return {
        'logits': torch.cat(epoch_logits, dim=0),
        'labels': torch.cat(epoch_labels, dim=0),
        'epoch_u': epoch_u,
        'epoch_std': epoch_std
    }
 


def train(args):

    # Get data
    train_loader, _, val_loader, _ = data.get_loaders(args)
    args.n_groups = train_loader.dataset.n_groups

    algorithm = utils.init_algorithm(args) 

    print('Args', '-'*50, '\n', args, '\n', '-'*50)

    # algorithm = init_algorithm(args, train_loader.dataset)
    saver = utils.Saver(algorithm, args.device, args.ckpt_dir)

    # Train loop
    best_worst_case_acc, best_average_acc = 0, 0
    
    if args.scheduler == 'cosine':
        scheduler = CosineAnnealingLR(optimizer=algorithm.optimizer, T_max=args.num_epochs)
    elif args.scheduler == 'step':
        scheduler = StepLR(optimizer=algorithm.optimizer, step_size=10, gamma=0.5)
    elif args.scheduler == 'cosine_warm':
        scheduler = CosineAnnealingWarmRestarts(optimizer=algorithm.optimizer, T_0=10, T_mult=1, eta_min=0.00001)

    for epoch in trange(1, args.num_epochs+1):
        _, _, _, epoch_results = run_epoch(algorithm, train_loader, train=True, progress_bar=args.progress_bar, mask=args.mask, mask_p=args.mask_p)
        
        if args.scheduler != 'none':
            scheduler.step()

        print(f"Epoch {epoch} - lr: {algorithm.optimizer.param_groups[0]['lr'] : .6f}", end='  ')
        if args.algorithm in ['ARM-UNC', 'ARM-CONF']:
            print(f"beta: {algorithm.beta.item():.4f}, tau: {algorithm.tau.item():4f}", end='  ')
            print(f"context_init: {algorithm.context_init[0][0][0][0].item():.4f}, {algorithm.context_init[0][0][0][1].item():.4f}")
        print()
        if epoch == 1 or epoch % args.epochs_per_eval == 0:
            stats = eval_groupwise(args, algorithm, val_loader, epoch, split='val', n_samples_per_group=args.n_samples_per_group)

            if args.worst_case:
                # Track early stopping values with respect to worst case.
                if stats['val/worst_case_acc'] > best_worst_case_acc:
                    print(f"\nBest updated at Epoch {epoch} !! - Worst: {stats['val/worst_case_acc']:.4f}, Avg: {stats['val/average_acc']:.4f}")
                    best_worst_case_acc = stats['val/worst_case_acc']
                    saver.save(epoch, is_best=True)
            else:
                # Track early stopping values with respect to worst case.
                if stats['val/average_acc'] > best_average_acc:
                    print(f"Best updated at Epoch {epoch} !! - \nWorst Case Acc: {stats['val/worst_case_acc']:.4f}, Average Acc: {stats['val/average_acc']:.4f}")
                    best_average_acc = stats['val/average_acc']
                    saver.save(epoch, is_best=True)

            # Log early stopping values
            if args.log_wandb:
                wandb.log({"val/best_worst_case_acc": best_worst_case_acc})

            print(f"\nEpoch: {epoch}\nWorst Case Acc: {stats['val/worst_case_acc']:.4f}, Average Acc: {stats['val/average_acc']:.4f}")

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
    num_noise = args.num_noise
    if args.num_noise >= support_size:
        print(f"num_noise is {args.num_noise} and support_size is {support_size}!!!")
        num_noise = support_size

    normal_iter = args.normal_iter
    noise_iter = args.noise_iter
    normal_count, noise_count = 0, 0
    for i, (idx, idx_not_group) in enumerate(zip(example_ids, example_ids_not_group)):
        x, y, g = loader.dataset[idx]
        
        if normal_count < normal_iter:
            normal_count += 1            
            X.append(x); Y.append(y); G.append(g)
        elif normal_iter == 0:
            x_noise = x.clone()
            x_noise = torch.tensor(make_noise(args, x_noise))
            X.append(x_noise); Y.append(y); G.append(g)
        else:
            if noise_count < noise_iter:
                noise_count += 1
                if args.noise_type != 'group':
                    x_noise = x.clone()
                    x_noise = torch.tensor(make_noise(args, x_noise))
                    if args.save_img:                        
                        # dir_img = f'assets/{args.dataset}/{str(g)}/'
                        dir_img = f'assets/{args.dataset}/{str(y)}/'
                        print("SAVE", dir_img)
                        createFolder(dir_img)
                        name = dir_img + f'{str(y.item())}_{str(g)}_{i}.png'
                        name_ori = dir_img + f'ori_{str(y.item())}_{str(g)}_{i}.png'
                        save_noisy_image(x, name_ori)
                        save_noisy_image(x_noise, name)
                    X.append(x_noise); Y.append(y); G.append(g)
                else:
                    x_noise, y_noise, g_noise = loader.dataset[idx_not_group]     
                    X.append(x_noise); Y.append(y_noise); G.append(g_noise)
            else:
                normal_count, noise_count = 1, 0
                X.append(x); Y.append(y); G.append(g)
       
        if (i + 1) % support_size == 0:
            normal_count, noise_count = 0, 0
            X, Y, G = torch.stack(X), torch.tensor(Y, dtype=torch.long), torch.tensor(G, dtype=torch.long)
            batches.append((X, Y, G))
            X, Y, G = [], [], []

        if n_samples_per_group is not None and i == (n_samples_per_group - 1):
            break
    if X:
        X, Y, G = torch.stack(X), torch.tensor(Y, dtype=torch.long), torch.tensor(G, dtype=torch.long)
        batches.append((X, Y, G))

    return batches

def make_context(args, algorithm, images):
        first_batch_size, c, h, w = images.shape
        algorithm.context_net.eval()
        with torch.no_grad():
            first_context = algorithm.context_net(images.to(algorithm.device)) # (batch_size, channels, H, W)

        if algorithm.__class__.__name__[-3:] == 'UNC':
            # compute sample-wise context (batch_size, channels, H, W)
            ctx = first_context.to(algorithm.device)

            # reshape x and ctx
            x = images.reshape(1, first_batch_size, c, h, w).to(algorithm.device)
            ctx = ctx.reshape(1, first_batch_size, algorithm.n_context_channels, h, w)

            # accumulate ctx by uncertainty weights
            u_list, ent_list = [], []
            ctx_list = [ctx.transpose(0, 1)[0]]
            
            # for each input data
            for idx, (x_t, ctx_t) in enumerate(zip(x.transpose(0, 1), ctx.transpose(0, 1))):
                x_ctx_t = torch.cat([x_t, algorithm.context_norm(ctx_list[-1])], dim=1)

                # compute uncertainty
                with torch.no_grad():
                    out_prob = []
                    for _ in range(args.T):
                        out_prob.append(algorithm.model(x_ctx_t) * torch.exp(algorithm.tau))

                out_prob = F.softmax(torch.stack(out_prob, dim=0), dim=-1)
                out_prob = torch.mean(out_prob, dim=0)
                entropy = torch.sum(-out_prob * torch.log2(out_prob + algorithm.eps), dim=-1)

                u = torch.exp(-torch.exp(algorithm.beta) * entropy)

                u_list.append(u); ent_list.append(entropy)
                ctx_list.append(u.reshape(-1, 1, 1, 1) * ctx_t + ctx_list[-1])

            # stack results
            u_list = torch.stack(u_list, dim=0).squeeze().tolist()
            ent_list = torch.stack(ent_list, dim=0).squeeze().tolist()

            # reshape context list (meta_batch_size, support_size, self.n_context_channels, h, w)
            ctx_list = torch.stack(ctx_list[:-1], dim=0).transpose(0, 1)

            # reshape input / context (meta_batch_size * support_size, self.n_context_channels, h, w)
            ctx_list = ctx_list.reshape(-1, algorithm.n_context_channels, h, w)
            x = x.reshape(-1, c, h, w)

            return ctx_list
        elif algorithm.__class__.__name__ == 'ARM_CML':
            # print("first_context:", first_context.size())
            first_context = first_context.cumsum(dim=0)
            length_tensor = torch.arange(1, first_batch_size+1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).to(first_context.device)
            first_context = torch.div(first_context, length_tensor.detach())
            return first_context
        else:
            print("Nooo...")
            return None



def eval_groupwise_with_context(args, algorithm, loader, epoch=None, split='val', n_samples_per_group=None):
    """ Test model on groups and log to wandb

        Separate script for femnist for speed."""

    groups = []
    accuracies = np.zeros(len(loader.dataset.groups))
    num_examples = np.zeros(len(loader.dataset.groups))

    acc_by_group_unseen, acc_by_group_seen = [], []
    acc_std = []
    print("loader.dataset.groups", loader.dataset.groups)

    # Loop over each group
    for i, group in tqdm(enumerate(loader.dataset.groups), desc='Evaluating', total=len(loader.dataset.groups)):
                
        group_iterator = get_group_iterator(loader, group, args.support_size, n_samples_per_group)[:-1]        
        group_iterator = random.sample(group_iterator, 2) # sample

        unseen_episodes = [[] for _ in range(len(group_iterator))]
        seen_episodes = [[] for _ in range(len(group_iterator))]

        # Loop over each episode
        for ep in range(len(group_iterator)):

            # First context vector
            images, labels, group_ids = group_iterator[ep] # ep: seen
            first_context = make_context(args, algorithm, images)

            # Loop over each context 1 ~ 50
            acc_with_context, entropy_with_context = [], []
            seen_acc_with_context, unseen_acc_with_context = [], []
            for idx_context in range(args.support_size):
                new_ctx = first_context[idx_context]
                results_with_context = run_epoch_with_context(algorithm, group_iterator, new_ctx)
                
                probs = results_with_context['logits'] # torch.Size([65, 2500, 10])
                labels = results_with_context['labels']
                epoch_u = results_with_context['epoch_u']
                epoch_std = results_with_context['epoch_std']

                preds = np.argmax(probs, axis=1)  # torch.Size([550])
                seen_preds = preds[ep*args.support_size:(ep+1)*args.support_size]; seen_labels = labels[ep*args.support_size:(ep+1)*args.support_size]
                unseen_preds = np.delete(preds, np.s_[ep*args.support_size:(ep+1)*args.support_size]); unseen_labels = np.delete(labels, np.s_[ep*args.support_size:(ep+1)*args.support_size])
                seen_accuracy = np.mean((seen_preds == seen_labels).numpy())
                unseen_accuracy = np.mean((unseen_preds == unseen_labels).numpy())

                # accuracy = np.mean((preds == labels).numpy()) # 1개 값. 컨텍스트벡터0에 대한 에피소드 하나의 정확도

                if epoch_std[0] is not None:
                    entropy = np.mean(np.array(epoch_std))
                    entropy_with_context.append(entropy)
                
                # acc_with_context.append(accuracy)
                seen_acc_with_context.append(seen_accuracy)
                unseen_acc_with_context.append(unseen_accuracy)
            seen_episodes[ep].extend(seen_acc_with_context)
            unseen_episodes[ep].extend(unseen_acc_with_context)
                        
            print(f"[seen ep {ep}] {seen_episodes[ep]}")
            print(f"[useen ep {ep}] {unseen_episodes[ep]}")

        acc_by_group_unseen.append(unseen_episodes)
        acc_by_group_seen.append(seen_episodes)
        # ent_by_group.append(entropy_with_context)
        # print(f"[group {group}] {acc_by_group[group]}")
        # raise Exception

    stats = {
                f'{split}/average_acc': acc_by_group_unseen,
                f'{split}/average_acc_seen': acc_by_group_seen,
                # f'{split}/average_ent': ent_by_group,
            }

    return stats



def eval_groupwise(args, algorithm, loader, epoch=None, split='val', n_samples_per_group=None):
    """ Test model on groups and log to wandb

        Separate script for femnist for speed."""

    groups = []
    accuracies = np.zeros(len(loader.dataset.groups))
    num_examples = np.zeros(len(loader.dataset.groups))
    online_accuracies = []
    online_weights = []
    online_stds = []

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

        probs, labels, group_ids, epoch_results = run_epoch(algorithm, group_iterator, train=False, progress_bar=False)

        epoch_logits = epoch_results['epoch_logits']
        epoch_labels = epoch_results['epoch_labels']
        epoch_u_list, epoch_std_list = epoch_results['epoch_u_list'], epoch_results['epoch_std_list']
        # if args.train and algorithm.model.__class__.__name__[-3:] == 'UNC':
            # print('epoch_u_list:', epoch_u_list[0][:5])
            # print('epoch_std_list:', epoch_std_list[0][:5])

        preds = np.argmax(probs, axis=1)

        # Evaluate
        if args.test and args.online:

            for i, logits in enumerate(epoch_logits):
                preds_online = np.argmax(logits, axis=1)
                label = epoch_labels[i]
                online_accuracies.append((preds_online == label).numpy().astype(int)) 
                online_weights.append(epoch_u_list[i])
                online_stds.append(epoch_std_list[i])
                       
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
                f'{split}/online_acc': online_accuracies,
                f'{split}/weights': online_weights,
                f'{split}/standard_errors': online_stds,
            }

    if epoch is not None:
        stats['epoch'] = epoch

    if args.log_wandb:
        wandb.log(stats)

    return stats


def save_noisy_image(img, name):
    save_image(img, name)
    # if img.size(0) == 3:
    #     img = img.view(img.size(0), 3, 32, 32)
    #     save_image(img, name)
    # else:
    #     img = img.view(img.size(0), 1, 28, 28)
    #     save_image(img, name)


def make_noise(args, img):

    if args.noise_type == 'sp':
        return random_noise(img, mode='s&p', amount=args.noise_level)
    else:
        return random_noise(img, mode='gaussian', var=args.noise_level).astype(np.float32)

import os
def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)    