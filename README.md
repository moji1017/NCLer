# **NLCer: A deep learning framework for automated label cleaning on real-world geological dataset**

## **Overivew**

**NLCer** is a deep learning framework designed to identify and filter mislabeled samples in real-world geological datasets, which are frequently characterized by long-tail distributions and coexisting closed-set and open-set noise.



## **Installation**

Install PyTorch 2.8.0 following the [official instructions](https://pytorch.org/).


```
git clone https://github.com/moji1017/NLCer.git
cd NLCer
pip install -r requirements.txt
```

Dependencies:
`pandas`, `numpy`, `scikit-learn`, `scipy`

## **Quick Start**

### **Synthetic Noisy Datasets (**`cifar10` **and** `newsgroups20`**)**

Place the datasets under the `data/` directory. For `newsgroups20`, GloVe embeddings are required for text preprocessing. For `cifar10`, MoCo v2 features are used to enable instance-dependent noise injection.

Configure noise settings in `main.py`:

python



```
args.noise_type = 'inst'                # Options: 'sym', 'asym', 'inst', 'imb'
args.noise_rate = 0.1                   # Proportion of corrupted labels
args.outlier_rate = 0.0                 # Proportion of open-set outliers
args.imbalance_coefficient = 1          # Controls severity of class imbalance
```

Run the pipeline:




```
python main.py
```

The process consists of two stages:

1. **Noise Early Stopping (NES)**
   Performs Monte Carlo cross-validation to determine:
   - `alpha`: coefficient for controlling effective class distribution in Class-Balanced Sampling
   - `lam`: weight of the prototype-negative term in the loss function
     Results are saved in `./output/NES/{dataset}/`.
2. **Training and Noise Filtering**
   Trains the model and identifies clean versus noisy samples.
   - Per-epoch training and evaluation metrics are saved in `./output/NLCer/{dataset}/`.
   - Final noise filtering results (sample-wise clean/noisy predictions) are saved in `./output/NLCer_filtering/{dataset}/`.

### **Real-World Geological Datasets**

We provide five real geological datasets in `data/geological_dataset/`:

- **Zircon3**
- **Apatite12**
- **Garnet7**
- **Garnet5**
- **Basalt6**

Each dataset includes:

- **Features**: concentrations of major and trace elements
- **Labels**: mineral or rock genesis categories assigned by domain experts (inherently noisy)
- **Data source**: reference to the original publication or database

To assess filtering performance in the absence of ground-truth noise labels, we use 5-fold cross-validation. Run:




```
python k_fold_evaluation.py
```

Evaluation results are saved in `./output/evaluations/`.

