import argparse
from algorithm.models import ContextNet, ConvNet, ResNet, ConvNetUNC, MLP, ContextNetEx, ResNetContext, ResNet_UNC
from algorithm.ResNet import ResNet18, ResNet18UNC
from algorithm.algorithm import ERM, DRNN, MMD, ARM_LL, DANN, ARM_BN, ARM_CML, ARM_CUSUM, ARM_UNC, ARM_CONF
import torch
from torch import nn
from pathlib import Path
import numpy as np


CPU_DEVICE = torch.device('cpu')
def get_device_from_arg(device_id):
    if (device_id is not None and
            torch.cuda.is_available() and
            0 <= device_id < torch.cuda.device_count()):
        return True, torch.device(f'cuda:{device_id}')
    else:
        return False, CPU_DEVICE

def make_arm_train_parser():
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    add_model_args(parser)

    # Data args
    parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist', 'femnist', 'cifar-c', 'tinyimg', 'cifar'])
    parser.add_argument('--data_dir', type=str, default='../data/')
    parser.add_argument('--auto', type=int, default=0, help='auto default option')

    # Main
    parser.add_argument('--meta_batch_size', type=int, default=6, help='Number of classes')
    parser.add_argument('--support_size', type=int, default=50, help='Support size: same as what we call batch size in the appendix')


    parser.add_argument('--smaller_model', type=int, default=1, help='use smaller model ') 
    parser.add_argument('--noisy', type=int, default=0, help='add noisy if 1') 
    parser.add_argument('--num_noise', type=int, default=1, help='Number of noises')  
    # parser.add_argument('--noise_type', type=str, default='random', choices=['random', 'front', 'back'])  
    parser.add_argument('--beta', type=float, default=1.0, help='coef of exponential distribution') 
    parser.add_argument('--T', type=int, default=3, help='num of iter') 
    parser.add_argument('--mask', type=int, default=None, help='masking loss if 1') 
    parser.add_argument('--mask_p', type=float, default=0.2, help='proportion of masking logits') 
    parser.add_argument('--normalize', type=int, default=0, help='normalize or not') 
    parser.add_argument('--worst_case', type=int, default=1, help='validation with worst_case or not') 
    parser.add_argument('--norm_type', type=str, default='batch', choices=['batch', 'layer', 'instance']) 
    parser.add_argument('--online', type=int, default=0, help='online test yn') 
    parser.add_argument('--affine_on', type=int, default=0, help='elementwise_affine on yn') 
    parser.add_argument('--scheduler', type=str, default='none', choices=['none', 'cosine', 'cosine_warm']) 
    parser.add_argument('--debug_unc', type=int, default=0)
    parser.add_argument('--zero_context', type=int, default=0)
    parser.add_argument('--noise_level', type=float, default=0.1)
    parser.add_argument('--cxt_self_include', type=int, default=0)
    parser.add_argument('--zero_init', type=int, default=1)
    parser.add_argument('--noise_type', type=str, default='sp', choices=['group', 'sp', 'gaussian'])
    parser.add_argument('--bald', type=int, default=0)

    parser.add_argument('--save_img', type=int, default=0)

    parser.add_argument('--normal_iter', type=int, default=10)
    parser.add_argument('--noise_iter', type=int, default=3)


    return parser


def update_arm_parser(args):
    
    if args.dataset in ['mnist']:
        args.meta_batch_size = 6
        args.support_size = 50
        args.n_context_channels = 12
    elif args.dataset == 'femnist':
        args.meta_batch_size = 2
        args.support_size = 50
        args.optimizer = 'sgd'
        args.weight_decay = 1e-4
        args.n_context_channels = 1
    elif args.dataset == 'cifar-c':
        args.meta_batch_size = 3
        args.support_size = 100
        args.n_context_channels = 3
        # args.num_epochs = 100
        args.n_samples_per_group = 2000
        args.test_n_samples_per_group = 3000
        args.optimizer = 'sgd'
        args.weight_decay = 1e-4
        args.learning_rate = 1e-2
        # args.adapt_bn = 1 # Need to check
    elif args.dataset == 'tinyimg':
        args.meta_batch_size = 3
        args.support_size = 100
        args.n_context_channels = 3
        args.optimizer = 'sgd'
        # args.num_epochs = 50        
        args.n_samples_per_group = 2000
        args.test_n_samples_per_group = 3000
        args.weight_decay = 1e-4
        args.learning_rate = 1e-2 
        # args.adapt_bn = 1        # Need to check  
        # args.model = 'resnet50'  # Need to check


def add_common_args(parser):

    # Data sampling
    parser.add_argument('--sampler', type=str, default='group', choices=['standard', 'group', 'online'], help='Standard or group sampler')
    parser.add_argument('--uniform_over_groups', type=int, default=1, help='Sample across groups uniformly')

    # Train / test
    parser.add_argument('--train', type=int, default=1, help="Train models")
    parser.add_argument('--test', type=int, default=1, help="Test models")
    parser.add_argument('--ckpt_folders', type=str, nargs='+') # only applicable when train is 0 and test is 1

    parser.add_argument('--progress_bar', type=int, default=0, help="Test models")

    # Training / Optimization args
    parser.add_argument('--num_epochs', type=int, default=200, help='Number of epochs')
    parser.add_argument('--optimizer', type=str, default='adam', choices=['sgd', 'adam'])
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0)

    parser.add_argument('--shuffle_train', type=int, default=1,
                        help='Only relevant when no group sampling = 0 \
                        and --uniform_over_groups 0')
    parser.add_argument('--drop_last', type=int, default=0)
    parser.add_argument('--loading_type', type=str, choices=['PIL', 'jpeg'], default='jpeg',
                        help='Whether to use PIL or jpeg4py when loading images. Jpeg is faster. See README for deatiles')

    # Torch
    parser.add_argument('--num_workers', type=int, default=8, help='Num workers for pytorch data loader')
    parser.add_argument('--pin_memory', type=int, default=1, help='Pytorch loader pin memory. Best practice is to use this')
    parser.add_argument('--device_id', type=int, default=0, help='device_id')

    # Evalaution
    parser.add_argument('--n_samples_per_group', type=int, default=300, help='Number of examples to evaluate on per test distribution For EVAL')
    parser.add_argument('--test_n_samples_per_group', type=int, default=None, help='Number of examples to evaluate on per test distribution For TEST')
    parser.add_argument('--epochs_per_eval', type=int, default=10)

    # Test
    parser.add_argument('--eval_on', type=str, nargs="*", default=['val', 'test'])

    # Logging
    parser.add_argument('--seeds', type=int, nargs="*", default=[0], help='Seeds')
    parser.add_argument('--plot', type=int, default=0, help='Plot or not')
    parser.add_argument('--exp_name', type=str, default='arm_cml')
    parser.add_argument('--debug', type=int, default=0)
    parser.add_argument('--log_wandb', type=int, default=0)




def add_model_args(parser):
    # Model args
    parser.add_argument('--model', type=str, default='convnet', choices=['resnet50', 'convnet', 'convnet_unc', 'resnet50_unc', 'resnet18_unc', 'resnet18'])
    parser.add_argument('--pretrained', type=int, default=1, help='Pretrained resnet')

    # Method
    parser.add_argument('--algorithm', type=str, default='ERM', 
                        choices=['ERM', 'DRNN', 'ARM-BN', 'ARM-LL', 'DANN', 'MMD', 'ARM-CML', 'ARM-CUSUM', 'ARM-UNC', 'ARM-CONF'])

    # ARM-CML
    parser.add_argument('--n_context_channels', type=int, default=12, help='Used when using a convnet/resnet')
    parser.add_argument('--context_net', type=str, default='ContextNet', choices=['ContextNet', 'ContextNetEx']) # ContextNet
    parser.add_argument('--pret_add_channels', type=int, default=1)
    parser.add_argument('--adapt_bn', type=int, default=0)
    parser.add_argument('--dropout_rate', type=float, default=0.2)

    # DANN
    parser.add_argument('--lambd', type=float, default=0.01)
    parser.add_argument('--d_steps_per_g_step', type=int, default=1)


def init_algorithm(args):

    if args.dataset in ['mnist']:
        num_classes = 10
        num_train_domains = 14
        n_img_channels = 1
        input_shape = (n_img_channels, 28, 28)
    elif args.dataset == 'femnist':
        num_classes = 62
        num_train_domains = 262
        n_img_channels = 1
        input_shape = (n_img_channels, 28, 28)
    elif args.dataset in 'cifar-c':
        num_classes = 10
        num_train_domains = 56
        n_img_channels = 3
        input_shape = (n_img_channels, 32, 32)        
    elif args.dataset in 'tinyimg':
        num_classes = 200
        num_train_domains = 51
        n_img_channels = 3
        input_shape = (n_img_channels, 64, 64)


    # Channels of main model
    if args.algorithm in ['ARM-CML', 'ARM-CUSUM', 'ARM-UNC', 'ARM-CONF']:
        n_channels = n_img_channels + args.n_context_channels
        hidden_dim = 64
        if args.context_net == 'ContextNetEx':
            context_net = ResNetContext(input_shape=(args.n_context_channels, 64, 64),
                                        in_channels=n_img_channels, 
                                        out_channels=args.n_context_channels * 64 * input_shape[2],
                                        model_name='resnet50',
                                        pretrained=args.pretrained).to(args.device)
        else:
            context_net = ContextNet(in_channels=n_img_channels, 
                                     out_channels=args.n_context_channels,
                                     hidden_dim=hidden_dim, 
                                     kernel_size=5).to(args.device)
    else:
        n_channels = n_img_channels

    if args.algorithm in ['DANN', 'MMD']:
        return_features = True
    else:
        return_features = False

    # Main model
    if args.model == 'convnet':
        model = ConvNet(num_channels=n_channels, num_classes=num_classes, 
                        # smaller_model=(args.algorithm == 'ARM-CML'), 
                        smaller_model=bool(args.smaller_model), 
                        return_features=return_features)
    elif args.model == 'resnet50':
        model = ResNet(num_channels=n_channels, num_classes=num_classes, model_name=args.model,
                                     pretrained=args.pretrained, return_features=return_features)
    elif args.model == 'convnet_unc':
        model = ConvNetUNC(num_classes=num_classes, 
                           num_channels=n_channels, 
                           smaller_model=bool(args.smaller_model), 
                           return_features=return_features, 
                           dropout_rate=args.dropout_rate)
    elif args.model == 'resnet50_unc':
        model = ResNet_UNC(num_channels=n_channels, num_classes=num_classes, model_name=args.model,
                                     pretrained=args.pretrained, return_features=return_features,
                                     dropout_rate=args.dropout_rate
                                     )
    elif args.model == 'resnet18':        
        model = ResNet18(n_channels)
        num_feats = model.fc.in_features
        model.fc = nn.Linear(num_feats, num_classes) # match class number

    elif args.model == 'resnet18_unc':
        model = ResNet18UNC(n_channels, args.dropout_rate)
        num_feats = model.fc.in_features
        model.fc = nn.Linear(num_feats, num_classes) # match class number


    model = model.to(args.device)


    # Loss fn
    if args.algorithm in ['DRNN']:
        loss_fn = nn.CrossEntropyLoss(reduction='none')
    else:
        if args.mask:
            loss_fn = nn.CrossEntropyLoss(reduction='none')
        else:
            loss_fn = nn.CrossEntropyLoss() # reduction='none'

    # Algorithm
    hparams = {'optimizer': args.optimizer,
               'learning_rate': args.learning_rate,
               'weight_decay': args.weight_decay}

    if args.algorithm == 'ERM':
        algorithm = ERM(model, loss_fn, args.device, hparams)
    elif args.algorithm == 'DRNN':
        hparams['robust_step_size'] = 0.01
        algorithm = DRNN(model, loss_fn, args.device, num_train_domains, hparams)

    elif args.algorithm == 'DANN':
        hparams['d_steps_per_g_step'] = args.d_steps_per_g_step
        hparams['lambd'] = args.lambd
        hparams['support_size'] = args.support_size
        algorithm = DANN(model, loss_fn, args.device, hparams, num_train_domains, num_classes)

    elif args.algorithm == 'MMD':
        hparams['support_size'] = args.support_size
        hparams['gamma'] = 1
        algorithm = MMD(model, loss_fn, args.device, hparams, num_classes)

    elif args.algorithm == 'ARM-CML':
        hparams['support_size'] = args.support_size
        hparams['n_context_channels'] = args.n_context_channels
        hparams['adapt_bn'] = args.adapt_bn
        hparams['online'] = args.online
        hparams['normalize'] = args.normalize
        hparams['T'] = args.T
        hparams['zero_context'] = args.zero_context

        print("Algorithm is ARM_CML.")
        algorithm = ARM_CML(model, loss_fn, args.device, context_net, hparams)

    elif args.algorithm == 'ARM-CUSUM':
        hparams['support_size'] = args.support_size
        hparams['n_context_channels'] = args.n_context_channels
        hparams['adapt_bn'] = args.adapt_bn
        hparams['normalize'] = args.normalize
        hparams['norm_type'] = args.norm_type
        hparams['input_shape'] = input_shape
        hparams['affine_on'] = args.affine_on
        hparams['T'] = args.T
        hparams['beta'] = args.beta
        hparams['zero_context'] = args.zero_context
        hparams['cxt_self_include'] = args.cxt_self_include
        
        print("Algorithm is ARM_CUSUM.")
        algorithm = ARM_CUSUM(model, loss_fn, args.device, context_net, hparams)

    elif args.algorithm == 'ARM-UNC':
        hparams['support_size'] = args.support_size
        hparams['n_context_channels'] = args.n_context_channels
        hparams['adapt_bn'] = args.adapt_bn
        hparams['normalize'] = args.normalize
        hparams['norm_type'] = args.norm_type
        hparams['input_shape'] = input_shape
        hparams['affine_on'] = args.affine_on
        hparams['beta'] = args.beta
        hparams['T'] = args.T
        hparams['debug'] = args.debug_unc
        hparams['zero_context'] = args.zero_context
        hparams['cxt_self_include'] = args.cxt_self_include
        hparams['zero_init'] = args.zero_init
        hparams['bald'] = args.bald
        

        print("Algorithm is ARM_UNC.")
        algorithm = ARM_UNC(model, loss_fn, args.device, context_net, hparams)

    elif args.algorithm == 'ARM-CONF':
        hparams['support_size'] = args.support_size
        hparams['n_context_channels'] = args.n_context_channels
        hparams['adapt_bn'] = args.adapt_bn
        hparams['normalize'] = args.normalize
        hparams['norm_type'] = args.norm_type
        hparams['input_shape'] = input_shape
        hparams['affine_on'] = args.affine_on
        hparams['beta'] = args.beta
        hparams['T'] = args.T
        hparams['debug'] = args.debug_unc
        hparams['zero_context'] = args.zero_context
        hparams['cxt_init_include'] = args.cxt_init_include
        

        print("Algorithm is ARM_CONF.")
        algorithm = ARM_CONF(model, loss_fn, args.device, context_net, hparams)

    elif args.algorithm == 'ARM-LL':
        learned_loss_net = MLP(in_size=num_classes, norm_reduce=True).to(args.device)
        hparams['support_size'] = args.support_size
        algorithm = ARM_LL(model, loss_fn, args.device, learned_loss_net, hparams)
    elif args.algorithm == 'ARM-BN':
        hparams['support_size'] = args.support_size
        algorithm = ARM_BN(model, loss_fn, args.device, hparams)

    return algorithm


class Saver:

    def __init__(self, algorithm, device, ckpt_dir):

        self.algorithm = algorithm
        self.device = device
        self.ckpt_dir = Path(ckpt_dir)

    def save(self, epoch, is_best):
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = self.ckpt_dir / f'{epoch}.pkl'
        torch.save(self.algorithm.to('cpu'), ckpt_path)

        if is_best:
            ckpt_path = self.ckpt_dir / f'best.pkl'
            torch.save(self.algorithm, ckpt_path)

        self.algorithm.to(self.device)



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
