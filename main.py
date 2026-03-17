import os
import sys
import random
import math
import copy
import warnings
from datetime import datetime

# Third-party libraries
import torch
import numpy as np
import argparse

# Custom modules
from train import nlcer
from Noise_early_stopping import nes

# ===================== Global Configuration and Warning Handling =====================
# Set OMP thread count to avoid multi-threading conflicts
os.environ['OMP_NUM_THREADS'] = '1'

# Ignore specific warnings
warnings.filterwarnings(
    "ignore",
    message="y_pred contains classes not in y_true",
    module="sklearn.metrics._classification"
)


# ===================== Utility Functions =====================
def setup_random_seed(seed_value):
    """Set random seed to ensure experiment reproducibility"""
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)


def get_device():
    """Get available computing device"""
    return 'cuda:0' if torch.cuda.is_available() else 'cpu'


# ===================== Parameter Configuration =====================
def create_arg_parser():
    """Create command line argument parser"""
    parser = argparse.ArgumentParser(description='RoCL Algorithm - Open Set Learning with Noisy Labels')

    # 1. Core Algorithm Related Parameters
    parser.add_argument('--method', type=str, default='NLCer',
                        help='Name of the algorithm to use')
    parser.add_argument('--loss', type=str, default='RoPCL',
                        help='Type of loss function')
    # Pseudo-Label Refinement module related
    parser.add_argument('--pseudo_label_weight', type=float, default=0.5,
                        help='Soft label weight in Pseudo-Label Refinement module')
    # OOD/ID discrimination related
    parser.add_argument('--tau', type=float, default=0.8,
                        help='Distance threshold for distinguishing OOD and ID samples')
    # Contrastive loss related
    parser.add_argument('--prototype_position', type=int, default=1,
                        help='Prototype position parameter')
    parser.add_argument('--margin', type=float, default=round(math.sqrt(2 * 1 ** 2), 1),
                        help='Margin of contrastive loss, calculated based on prototype_position')

    # Sampling strategy related
    parser.add_argument('--alpha', type=float, default=1.0,
                        help='Coefficient for controlling class distribution in Class-Balanced Sampling')

    # Monte Carlo cross-validation related
    parser.add_argument('--dataset_split_rate', type=float, default=0.0,
                        help='Split ratio of validation set and training set in Monte Carlo cross-validation')
    parser.add_argument('--noise_early_stopping', type=bool, default=False,
                        help='Whether to use Monte Carlo cross-validation to determine noise early stopping timing')
    parser.add_argument('--split_seed', type=int, default=0, help='Random seed for dataset splitting in Monte Carlo cross-validation')

    # 2. Training Parameters
    parser.add_argument('--n_epoch', type=int, default=15, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Initial learning rate')
    parser.add_argument('--optimizer', type=str, default='adam',
                        choices=['adam', 'sgd', 'rmsprop'], help='Optimizer')
    parser.add_argument('--weight_decay', type=float, default=0,
                        help='Weight decay (L2 regularization)')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum coefficient')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')

    # 3. Noise Settings
    parser.add_argument('--noise_rate', type=float, default=0.4,
                        help='Label noise rate')
    parser.add_argument('--noise_type', type=str, default='imb',
                        choices=['sym', 'asym', 'inst', 'imb'],
                        help='Noise type: sym(symmetric)/asym(asymmetric)/instance(instance-related)/imb(imbalance-induced)')
    parser.add_argument('--outlier_rate', type=float, default=0.0,
                        help='Proportion of open-set noise in total noise')
    parser.add_argument('--imbalance_coefficient', type=float, default=0.1,
                        help='Data imbalance coefficient (minimum class/maximum class)')

    # 4. Model Structure
    parser.add_argument('--hidden_dim', type=int, nargs='+', default=[100, 300, 100],
                        help='MLP hidden layer dimensions')
    parser.add_argument('--dropout', type=float, default=0.0,
                        help='Dropout rate')

    # 5. Random Seeds
    parser.add_argument('--seed', type=int, default=0, help='Random seed for model training')

    # 6. Device and Data Loading
    parser.add_argument('--device', type=str, default=get_device(),
                        help='Computing device')
    parser.add_argument('--dataset', type=str, default='news',
                        choices=['cifar10', 'news', 'Zircon3', 'Apatite12', 'Garnet7', 'Garnet5', 'Basalt6'],
                        help='Dataset selection:\n'
                             '  - Benchmark datasets: cifar10, news\n'
                             '  - Geological datasets: Zircon3, Apatite12, Garnet7, Garnet5, Basalt6')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='Number of data loading threads')
    parser.add_argument('--save_features', type=bool, default=False,
                        help='Whether to save features')

    return parser


def initialize_experiment_args(args):
    """Initialize experiment parameters (only keep specified core parameters)"""
    # 1. Fixed seed related
    args.seed = 0  # Random seed for model training
    args.split_seed = 0  # Random seed for dataset splitting in Monte Carlo cross-validation

    # 2. Core algorithm parameters
    args.method = "NLCer"
    args.loss = "RoPCL"
    args.n_epoch = 10

    # 3. Dataset related (retained)
    args.dataset_type = 'geological'  # geological /  Benchmark

    if args.dataset_type == 'geological':
        args.dataset = 'Apatite12'  # Zircon3, Apatite12, Garnet7, Garnet5, Basalt6
    else:
        args.dataset = 'news'  # news, cifar10
        args.noise_type = 'inst'  # sym / asym / inst / imb
        args.noise_rate = 0.1
        args.outlier_rate = 0.0
        args.imbalance_coefficient = 1

    # 4. Runtime environment configuration (retained)
    args.device = get_device()
    args.num_workers = 0 if args.device == "cpu" else 8

    return args


# ===================== Main Function =====================
def main(args):
    """Main training process"""
    # 1. Initialize random seed
    setup_random_seed(args.seed)

    # 2. Initialize experiment parameters
    args = initialize_experiment_args(args)

    # 3. Noise early stopping related parameters (retained)
    args.initial_learning_ep, args.lam, args.alpha = nes(args)
    args.start_time = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 4. Start main training logic
    nlcer(args)


if __name__ == '__main__':
    # Parse command line arguments
    parser = create_arg_parser()
    args = parser.parse_args()

    # Execute main program
    main(args)