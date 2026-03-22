## Overview

**NLCer** (Noisy Label Cleaner) is a deep learning framework designed to identify and filter mislabeled samples in real-world geological datasets, yielding cleaner data for downstream tasks.

![](imge\NCLer.png)

## Feature Highlights

- **Multi-modal Support**: Compatible with various data modalities, including images and tabular data.
- **Robustness**: Effectively handles open-set, closed-set, and long-tail distribution problems simultaneously. 
- **Prior-free**: Operates without requiring prior knowledge of noise rates or distributions.

## Install

1. Install PyTorch 2.8.0 following the [official instructions](https://pytorch.org/).

   ```
   git clone https://github.com/moji1017/NLCer.git
   cd NLCer
   pip install -r requirements.txt
   ```

   Dependencies: `pandas`, `numpy`, `scikit-learn`, `scipy`

## Quick Start

We demonstrate NLCer's capabilities using both Benchmark datasets (**CIFAR-10**, **20 Newsgroups**) and real-world Geological datasets (**Zircon3, Apatite12, Garnet7, Garnet5, Basalt6**).

### 1. Dataset Preparation

- **Benchmark Datasets**: CIFAR-10 and 20 Newsgroups can be automatically downloaded by running the scripts `.data/cifar10.py` and `.data/news.py`.
- **Geological Datasets**: Collected from previous literature and organized under `.data/geological_dataset/`. This directory includes features, labels, and references.

### 2. Noise Early Stopping (NES)

Before training, determine the optimal hyperparameters and the timing for pseudo-labeling using Monte Carlo cross-validation.

Run `Noise_early_stopping.py`:

```
# Configuration example
args.dataset = 'news'  # Options: 'news', 'cifar10', 'Zircon3', 'Apatite12', 'Garnet7', 'Garnet5', 'Basalt6'
args.noise_type = 'inst'  # Options: 'sym', 'asym', 'inst', 'imb'
args.noise_rate = 0.1     # Proportion of corrupted labels (Ignored for real geological datasets)
args.outlier_rate = 0.0   # Proportion of open-set outliers
args.imbalance_coefficient = 1 # Controls severity of class imbalance
```

- Results are saved in `./output/NES/{dataset}/`.
- **Note**: For real-world geological datasets, which inherently contain noise, do not inject artificial noise. The algorithm estimates the optimal stopping point based on the inherent noise.

### 3. Training

After NES completes, run `main.py` for training and noise identification.

```
python main.py
```

- **Important**: Ensure the configuration in `main.py` matches `Noise_early_stopping.py`. If configurations differ, the script will automatically re-run the `Noise_early_stopping.py` to obtain hyperparameters.

### 4. Evaluation

#### For Benchmark Datasets

- Per-epoch metrics: `./output/NLCer/{dataset}/`
- Final filtering results (sample-wise clean/noisy predictions): `./output/NLCer_filtering/{dataset}/`

#### For Geological Datasets

Since ground-truth noise labels are unavailable, we evaluate filtering performance via **5-fold cross-validation**.

```
python k_fold_evaluation.py
```

- Evaluation results are saved in `./output/evaluations/{dataset}/`.
