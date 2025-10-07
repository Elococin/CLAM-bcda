from __future__ import print_function

import argparse
import pdb
import os
import math

# internal imports
from utils.file_utils import save_pkl, load_pkl
from utils.utils import *
from utils.core_utils import train
from dataset_modules.dataset_generic import Generic_WSI_Classification_Dataset, Generic_MIL_Dataset

# pytorch imports
import torch
from torch.utils.data import DataLoader, sampler
import torch.nn as nn
import torch.nn.functional as F

import pandas as pd
import numpy as np
import yaml


# -----------------------------------------------------------
# Utility: YAML loader
# -----------------------------------------------------------
def load_yaml(path):
    """Simple YAML loader for config files."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


# -----------------------------------------------------------
# Argument parsing
# -----------------------------------------------------------
parser = argparse.ArgumentParser(description='Configurations for WSI Training')
parser.add_argument('--data_root_dir', type=str, default=None, help='data directory')
parser.add_argument('--config', type=str, default=None, help='Path to YAML config file for multi-dataset setup')
parser.add_argument('--embed_dim', type=int, default=1024)
parser.add_argument('--max_epochs', type=int, default=200, help='maximum number of epochs to train')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--label_frac', type=float, default=1.0)
parser.add_argument('--reg', type=float, default=1e-5)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--k', type=int, default=10)
parser.add_argument('--k_start', type=int, default=-1)
parser.add_argument('--k_end', type=int, default=-1)
parser.add_argument('--results_dir', default='./results')
parser.add_argument('--split_dir', type=str, default=None)
parser.add_argument('--log_data', action='store_true', default=False)
parser.add_argument('--testing', action='store_true', default=False)
parser.add_argument('--early_stopping', action='store_true', default=False)
parser.add_argument('--opt', type=str, choices=['adam', 'sgd'], default='adam')
parser.add_argument('--drop_out', type=float, default=0.25)
parser.add_argument('--bag_loss', type=str, choices=['svm', 'ce'], default='ce')
parser.add_argument('--model_type', type=str, choices=['clam_sb', 'clam_mb', 'mil'], default='clam_sb')
parser.add_argument('--exp_code', type=str)
parser.add_argument('--weighted_sample', action='store_true', default=False)
parser.add_argument('--model_size', type=str, choices=['small', 'big'], default='small')
parser.add_argument('--task', type=str, choices=['task_1_tumor_vs_normal', 'task_2_tumor_subtyping', 'ER_status'])
parser.add_argument('--csv_path', type=str, default=None)
parser.add_argument('--label_dir', type=str, default=None)
parser.add_argument('--no_inst_cluster', action='store_true', default=False)
parser.add_argument('--inst_loss', type=str, choices=['svm', 'ce', None], default=None)
parser.add_argument('--subtyping', action='store_true', default=False)
parser.add_argument('--bag_weight', type=float, default=0.7)
parser.add_argument('--B', type=int, default=8)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print('\nLoad Dataset')


# -----------------------------------------------------------
# Dataset loading
# -----------------------------------------------------------
if args.task == 'task_1_tumor_vs_normal':
    args.n_classes = 2
    dataset = Generic_MIL_Dataset(
        csv_path='dataset_csv/tumor_vs_normal_dummy_clean.csv',
        data_dir=os.path.join(args.data_root_dir, 'tumor_vs_normal_resnet_features'),
        shuffle=False,
        seed=args.seed,
        print_info=True,
        label_dict={'normal_tissue': 0, 'tumor_tissue': 1},
        patient_strat=False,
        ignore=[]
    )

elif args.task == 'task_2_tumor_subtyping':
    args.n_classes = 3
    dataset = Generic_MIL_Dataset(
        csv_path='dataset_csv/tumor_subtyping_dummy_clean.csv',
        data_dir=os.path.join(args.data_root_dir, 'tumor_subtyping_resnet_features'),
        shuffle=False,
        seed=args.seed,
        print_info=True,
        label_dict={'subtype_1': 0, 'subtype_2': 1, 'subtype_3': 2},
        patient_strat=False,
        ignore=[]
    )

    if args.model_type in ['clam_sb', 'clam_mb']:
        assert args.subtyping

elif args.task == 'ER_status':
    args.n_classes = 2

    # --- Begin YAML-based multi-dataset patch ---
    if args.config is not None and os.path.exists(args.config):
        cfg = load_yaml(args.config)
    else:
        raise FileNotFoundError(f"Config file {args.config} not found.")

    datasets = cfg.get("datasets", [])
    held_out = cfg.get("held_out", None)
    args.split_dir = cfg.get("split_dir", args.split_dir)


    merged_csvs = []
    for dname in datasets:
        label_csv = cfg["label_csvs"][dname]
        feat_dir = cfg["feature_dirs"][dname]
        df = pd.read_csv(label_csv)
        df["dataset"] = dname
        df["feat_dir"] = feat_dir
        merged_csvs.append(df)

    merged_df = pd.concat(merged_csvs, ignore_index=True)
    os.makedirs(cfg["results_dir"], exist_ok=True)
    tmp_csv_path = os.path.join(cfg["results_dir"], "merged_train_labels.csv")
    merged_df.to_csv(tmp_csv_path, index=False)

    print(f"[INFO] Combined {len(datasets)} datasets into {tmp_csv_path}")
    print(f"[INFO] Held-out dataset: {held_out}")

    dataset = Generic_WSI_Classification_Dataset(
        csv_path=tmp_csv_path,
        shuffle=False,
        seed=args.seed,
        print_info=True,
        label_dict={'0': 0, '1': 1},
        patient_strat=False,
        ignore=[]
    )
    # --- End YAML-based multi-dataset patch ---

else:
    raise NotImplementedError


# -----------------------------------------------------------
# Helper: seed setup
# -----------------------------------------------------------
def seed_torch(seed=7):
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


seed_torch(args.seed)


# -----------------------------------------------------------
# Main training loop
# -----------------------------------------------------------
def main(args):
    # create results directory if necessary
    if not os.path.isdir(args.results_dir):
        os.mkdir(args.results_dir)

    if args.k_start == -1:
        start = 0
    else:
        start = args.k_start
    if args.k_end == -1:
        end = args.k
    else:
        end = args.k_end

    all_test_auc, all_val_auc, all_test_acc, all_val_acc = [], [], [], []
    folds = np.arange(start, end)
    for i in folds:
        seed_torch(args.seed)
        train_dataset, val_dataset, test_dataset = dataset.return_splits(
            from_id=False, csv_path='{}/splits_{}.csv'.format(args.split_dir, i)
        )

        datasets = (train_dataset, val_dataset, test_dataset)
        results, test_auc, val_auc, test_acc, val_acc = train(datasets, i, args)
        all_test_auc.append(test_auc)
        all_val_auc.append(val_auc)
        all_test_acc.append(test_acc)
        all_val_acc.append(val_acc)

        # write results to pkl
        filename = os.path.join(args.results_dir, f'split_{i}_results.pkl')
        save_pkl(filename, results)

    final_df = pd.DataFrame({
        'folds': folds,
        'test_auc': all_test_auc,
        'val_auc': all_val_auc,
        'test_acc': all_test_acc,
        'val_acc': all_val_acc
    })

    save_name = (
        f'summary_partial_{start}_{end}.csv'
        if len(folds) != args.k else 'summary.csv'
    )
    final_df.to_csv(os.path.join(args.results_dir, save_name))


# -----------------------------------------------------------
# Experiment setup
# -----------------------------------------------------------
encoding_size = 768
settings = {
    'num_splits': args.k,
    'k_start': args.k_start,
    'k_end': args.k_end,
    'task': args.task,
    'max_epochs': args.max_epochs,
    'results_dir': args.results_dir,
    'lr': args.lr,
    'experiment': args.exp_code,
    'reg': args.reg,
    'label_frac': args.label_frac,
    'bag_loss': args.bag_loss,
    'seed': args.seed,
    'model_type': args.model_type,
    'model_size': args.model_size,
    'use_drop_out': args.drop_out,
    'weighted_sample': args.weighted_sample,
    'opt': args.opt
}

if args.model_type in ['clam_sb', 'clam_mb']:
    settings.update({'bag_weight': args.bag_weight, 'inst_loss': args.inst_loss, 'B': args.B})

# safe exp_code handling
exp_name = args.exp_code if args.exp_code else "default"
args.results_dir = os.path.join(args.results_dir, f"{exp_name}_s{args.seed}")
os.makedirs(args.results_dir, exist_ok=True)

if args.split_dir is None:
    args.split_dir = os.path.join('splits', args.task + '_{}'.format(int(args.label_frac * 100)))
else:
    args.split_dir = os.path.join('splits', args.split_dir)

print('split_dir: ', args.split_dir)
assert os.path.isdir(args.split_dir)
settings.update({'split_dir': args.split_dir})

# write experiment summary
with open(os.path.join(args.results_dir, f'experiment_{exp_name}.txt'), 'w') as f:
    print(settings, file=f)

print("################# Settings ###################")
for key, val in settings.items():
    print(f"{key}:  {val}")

# -----------------------------------------------------------
# Run training
# -----------------------------------------------------------
if __name__ == "__main__":
    results = main(args)
    print("finished!")
    print("end script")
