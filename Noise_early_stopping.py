import os
import pandas as pd
import re
from pathlib import Path
import argparse
from typing import List, Tuple, Optional

# Ensure main_nldtor function can be imported correctly
from train import nlcer
from datetime import datetime
import torch
import math


# ======================== Core Utility Functions ========================
def get_device():
    """Get available computing device (CUDA if available, otherwise CPU)"""
    return 'cuda:0' if torch.cuda.is_available() else 'cpu'


def extract_hyperparams_from_filename(filename):
    """Extract core hyperparameters (only keep lam/alpha) from filename"""
    lam = alpha = None

    loss_match = re.search(r'lam(\d+(?:\.\d+)?)', filename)
    if loss_match:
        lam = float(loss_match.group(1))

    sample_match = re.search(r'alpha(\d+(?:\.\d+)?)', filename)
    if sample_match:
        alpha = float(sample_match.group(1))

    return {'lam': lam, 'alpha': alpha}


def build_folder_path(dataset, noise_type, noise_rate, outlier_rate, imbalance_coefficient):
    """Build target folder path (retain core logic)"""
    noise_rate_str = f"{noise_rate:g}"
    outlier_rate_str = f"{outlier_rate:g}"
    imbalance_coeff_str = f"{imbalance_coefficient:g}"
    folder_name = f"{noise_type}_{noise_rate_str}_o{outlier_rate_str}_im{imbalance_coeff_str}"
    return Path(dataset) / folder_name


# ======================== File Checking and Completion ========================
def check_required_files(
        base_dir: str,
        dataset: str,
        noise_type: str,
        noise_rate: float,
        outlier_rate: float,
        imbalance_coefficient: float,
        seeds: List[int],
        lam_values: List[float],
        alpha_values: List[float]
) -> Tuple[bool, List[dict], int, int]:
    """Check if required files exist (only retain core checking logic)"""
    target_folder = build_folder_path(dataset, noise_type, noise_rate, outlier_rate, imbalance_coefficient)
    full_input_dir = Path(base_dir) / target_folder if base_dir else target_folder

    missing_files = []
    found_count = 0
    total_should_check = 0

    # Automatically create missing folders
    if not full_input_dir.exists():
        full_input_dir.mkdir(parents=True, exist_ok=True)

    # Load all csv files and extract hyperparameters
    csv_files = []
    for file in full_input_dir.glob("*.csv"):
        if file.is_file():
            hp = extract_hyperparams_from_filename(file.name)
            seed_match = re.search(r'seed(\d+)', file.name)
            csv_files.append({
                'lam': hp['lam'],
                'alpha': hp['alpha'],
                'seed': int(seed_match.group(1)) if seed_match else None
            })

    # Check if parameter combinations are complete
    for seed in seeds:
        for lo in lam_values:
            for sr in alpha_values:
                total_should_check += 1
                found = any(f['seed'] == seed and f['lam'] == lo and f['alpha'] == sr for f in csv_files)
                if found:
                    found_count += 1
                else:
                    missing_files.append({'seed': seed, 'lam': lo, 'alpha': sr})

    all_ok = len(missing_files) == 0
    return all_ok, missing_files, total_should_check, found_count


def run_main_nldtor(args, missing_list):
    """Run main_nldtor to complete missing files (only retain core logic)"""
    # Set default experiment parameters (avoid missing when passed externally)
    n_epoch = args.n_epoch
    args.noise_early_stopping = True
    args.dataset_split_rate = 0.3
    args.initial_learning_ep = 1000
    args.n_epoch = args.nes_epoch

    success_count = 0
    fail_count = 0
    for missing_item in missing_list:
        # Extract parameters
        args.split_seed = missing_item['seed']
        args.lam = missing_item['lam']
        args.alpha = missing_item['alpha']
        args.start_time = datetime.now().strftime("%Y%m%d_%H%M%S")

        nlcer(args)  # Call core experiment function
        success_count += 1

        # try:
        #     main_nlcer(args)  # Call core experiment function
        #     success_count += 1
        # except Exception as e:
        #     print(f"❌ Completion failed seed={args.split_seed}, lam={args.lam}, alpha={args.alpha}: {str(e)}")
        #     fail_count += 1

    print(f"\n📊 Completion finished: {success_count} succeeded, {fail_count} failed")
    # Restore original parameters (if any)
    args.n_epoch = n_epoch
    args.noise_early_stopping = False
    args.dataset_split_rate = 0.0


# ======================== Optimal Parameter Selection (Core) ========================
def get_best_params_with_ep(
        input_dir: str,
        seeds: List[int],
        lam_values: List[float],
        alpha_values: List[float],
        net_types: List[str],
        window_size: int = 5,  # Default sliding window size
        max_epoch: Optional[int] = 100  # Default maximum epoch
) -> dict:
    """
    Core logic: Manually calculate moving average based on original noisy_acc/clean_acc to select optimal hyperparameters
    Core metric: Maximum value of validation set accuracy (noisy_acc) after moving average
    """
    # 1. Filter files that meet the criteria
    file_records = []
    seed_pattern = re.compile(r'seed(\d+)')
    net_pattern = re.compile(r'(net1|net2|net3)')  # Extended support for more net types

    # Validate input directory
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    all_filenames = os.listdir(input_dir)
    for filename in all_filenames:
        if not filename.endswith(".csv"):
            continue

        # Filter by net type
        net_match = net_pattern.search(filename)
        if not net_match or net_match.group(1) not in net_types:
            continue
        # Filter by seed
        seed_match = seed_pattern.search(filename)
        if not seed_match or int(seed_match.group(1)) not in seeds:
            continue

        # Extract hyperparameters and filter
        hyperparams = extract_hyperparams_from_filename(filename)
        if hyperparams['lam'] not in lam_values or hyperparams['alpha'] not in alpha_values:
            continue

        file_records.append({
            "filepath": os.path.join(input_dir, filename),
            "seed": int(seed_match.group(1)),
            "net": net_match.group(1),
            "lam": hyperparams['lam'],
            "alpha": hyperparams['alpha']
        })

    # if not file_records:
    #     raise FileNotFoundError(
    #         f"No matching experiment files found: {input_dir} (Filter criteria: seeds={seeds}, net_types={net_types}, lam={lam_values}, alpha={alpha_values})")

    # 2. Group statistics by hyperparameters
    groups = {}
    for rec in file_records:
        key = (rec['lam'], rec['alpha'])
        if key not in groups:
            groups[key] = []
        groups[key].append(rec)

    # 3. Calculate core metrics for each group (manual moving average based on original acc)
    group_metrics = []
    for hyper_key, group_files in groups.items():
        lam, alpha = hyper_key
        clean_epochs = []
        noise_epochs = []
        val_acc_list = []  # List of validation set accuracy (after moving average)

        for rec in group_files:
            try:
                # Handle Windows long path issue
                abs_path = os.path.abspath(rec["filepath"])
                unc_path = '\\\\?\\' + abs_path if os.name == 'nt' and not abs_path.startswith('\\\\?\\') else abs_path
                df = pd.read_csv(unc_path)
            except Exception as e:
                print(f"⚠️  Skip unreadable file: {rec['filepath']} | Error: {e}")
                continue

            # Filter by maximum epoch
            if max_epoch is not None:
                df = df[df['epoch'] <= max_epoch].copy()
            if df.empty:
                print(f"⚠️  No valid data in file (epoch≤{max_epoch}): {rec['filepath']}")
                continue

            # ========== Core: Manually calculate moving average based on original noisy_acc ==========
            if 'noisy_acc' not in df.columns:
                print(f"⚠️  No noisy_acc column in file: {rec['filepath']}")
                continue
            if df['noisy_acc'].isna().all():
                print(f"⚠️  All values in noisy_acc column are empty: {rec['filepath']}")
                continue

            # Calculate rolling window average (min_periods=1 ensures first row has value)
            df['noisy_acc_rolling'] = df['noisy_acc'].rolling(
                window=window_size,
                min_periods=1,
                center=False  # Right-aligned window (common approach)
            ).mean()
            # Take the maximum value after moving average as the validation set accuracy of this file
            best_val_acc = df['noisy_acc_rolling'].max()
            val_acc_list.append(best_val_acc)
            # Extract corresponding epoch
            best_epoch_idx = df['noisy_acc_rolling'].idxmax()
            noise_epochs.append(int(df.loc[best_epoch_idx, 'epoch']))

            # Manually calculate moving average based on original clean_acc
            if 'clean_acc' in df.columns and not df['clean_acc'].isna().all():
                df['clean_acc_rolling'] = df['clean_acc'].rolling(
                    window=window_size,
                    min_periods=1,
                    center=False
                ).mean()
                clean_epoch_idx = df['clean_acc_rolling'].idxmax()
                clean_epochs.append(int(df.loc[clean_epoch_idx, 'epoch']))
            # ========== End of moving average calculation ==========

        # Calculate statistical values for this hyperparameter combination (only keep combinations with valid data)
        if val_acc_list:
            avg_val_acc = round(pd.Series(val_acc_list).mean(), 4)
            clean_ep_mean = round(pd.Series(clean_epochs).mean(), 2) if clean_epochs else None
            clean_ep_median = round(pd.Series(clean_epochs).median(), 1) if clean_epochs else None
            noise_ep_mean = round(pd.Series(noise_epochs).mean(), 2) if noise_epochs else None
            noise_ep_median = round(pd.Series(noise_epochs).median(), 1) if noise_epochs else None

            group_metrics.append({
                "lam": lam,
                "alpha": alpha,
                "avg_val_acc": avg_val_acc,  # Mean validation set accuracy (core)
                "clean_ep_mean": clean_ep_mean,
                "clean_ep_median": clean_ep_median,
                "noise_ep_mean": noise_ep_mean,
                "noise_ep_median": noise_ep_median,
                "val_acc_count": len(val_acc_list)  # Amount of valid data
            })

    # 4. Find hyperparameters with the highest validation set accuracy
    # if not group_metrics:
    #     raise ValueError("No valid experimental data to statistics (all files have no valid noisy_acc data)")

    best_group = max(group_metrics, key=lambda x: x['avg_val_acc'])
    return best_group


# ======================== Core Process Entry (Adapted for External Calls) ========================
def nes(args):
    """
    Core process: Check files → Complete missing ones → Select optimal parameters
    Adapted for external calls: Automatically assign default values to missing key parameters
    """
    # ========== 1. Assign default values to core parameters (use these values if missing when passed externally) ==========
    # Default values for hyperparameter ranges
    n_epoch = args.n_epoch
    args.seeds = getattr(args, 'seeds', [0, 1] if args.dataset == "cifar10" else [0, 1, 2])  # , 1, 2
    args.nes_epoch = getattr(args, 'nes_epoch', 100)
    args.lam_values = getattr(args, 'lam_values', [1.0, 1.25, 1.5])  # , 1.25, 1.5
    args.alpha_values = getattr(args, 'alpha_values', [0.25, 0.5, 0.75])  # , 0.5 , 0.75
    # Default values for filtering configuration
    args.net_types = getattr(args, 'net_types', ["net1", "net2"])
    args.window_size = getattr(args, 'window_size', 5)  # Default sliding window size is 5
    args.max_epoch = getattr(args, 'max_epoch', 100)  # Default maximum epoch is 100
    # Default value for base path
    args.base_dir = getattr(args, 'base_dir', 'outputs/NES/')
    if args.dataset in ['Zircon3', 'Apatite12', 'Garnet7', 'Garnet5', 'Basalt6']:
        args.noise_type, args.noise_rate, args.outlier_rate, args.imbalance_coefficient = 'imb', 0.0, 0.0, 1

    # Validate required parameters (dataset/noise_type/noise_rate, etc.)
    required_args = ['dataset', 'noise_type', 'noise_rate', 'outlier_rate', 'imbalance_coefficient']
    missing_required = [arg for arg in required_args if not hasattr(args, arg)]
    if missing_required:
        raise ValueError(f"Missing required parameters: {missing_required} (must be passed when calling externally)")

    # ========== 2. Check and complete missing files ==========
    all_ok, missing_list, total, found = check_required_files(
        base_dir=args.base_dir,
        dataset=args.dataset,
        noise_type=args.noise_type,
        noise_rate=args.noise_rate,
        outlier_rate=args.outlier_rate,
        imbalance_coefficient=args.imbalance_coefficient,
        seeds=args.seeds,
        lam_values=args.lam_values,
        alpha_values=args.alpha_values
    )

    print(f"\n🔍 File check result: {total} combinations to check, {found} found, {len(missing_list)} missing")
    if not all_ok:
        run_main_nldtor(args, missing_list)

    # ========== 3. Select optimal parameters ==========
    experiment_dir = build_folder_path(args.dataset, args.noise_type, args.noise_rate, args.outlier_rate, args.imbalance_coefficient)
    full_experiment_dir = Path(args.base_dir) / experiment_dir if args.base_dir else experiment_dir

    best_params = get_best_params_with_ep(
        input_dir=str(full_experiment_dir.absolute()),
        seeds=args.seeds,
        lam_values=args.lam_values,
        alpha_values=args.alpha_values,
        net_types=args.net_types,
        window_size=args.window_size,
        max_epoch=args.max_epoch
    )

    # Output results
    print("\n" + "=" * 60)
    print("🎯 Optimal Hyperparameters and Statistical Results")
    print("=" * 60)
    print(f"  Optimal lam        = {best_params['lam']}")
    print(f"  Optimal alpha      = {best_params['alpha']}")
    print(f"  Mean validation accuracy    = {best_params['avg_val_acc']}")
    print(f"  Valid data count            = {best_params.get('val_acc_count', 0)}")
    print(f"  clean_ep - mean     = {best_params['clean_ep_mean']}")
    print(f"  clean_ep - median   = {best_params['clean_ep_median']}")
    print(f"  noisy_ep - mean     = {best_params['noise_ep_mean']}")
    print(f"  noisy_ep - median   = {best_params['noise_ep_median']}")
    print("=" * 60)

    return int(best_params['noise_ep_median']), best_params['lam'], best_params['alpha']
