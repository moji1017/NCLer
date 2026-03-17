import os, sys, random
import torch, math, copy
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score

from utils import *
from networks import get_model_function
from dataloader import get_dataloader
from loss import RobustPrototypeContrastiveLoss
import argparse

parser = argparse.ArgumentParser()

# Method options
parser.add_argument('--method', type=str, default='RoCL', help='name of the algorithm to use')
parser.add_argument('--lam', type=float, default=0.4,
                    help='The proportion of the open-set noisy labels in the loss function')
parser.add_argument('--begin', type=int, default=20,
                    help='When to begin updating labels')
# Training parameters
parser.add_argument('--n_epoch', type=int, default=200)
parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'sgd', 'rmsprop'],
                    help='optimizer to use (e.g., adam, sgd, rmsprop)')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay (L2 regularization)')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum factor for SGD optimizer')
parser.add_argument('--batch_size', type=int, default=128, help='input batch size for training')

# Noise settings
parser.add_argument('--noise_rate', type=float, help='corruption rate, should be less than 1', default=0.3)
parser.add_argument('--noise_type', type=str, choices=['sym', 'asym', 'instance', 'imb'], default='sym',
                    help='type of label noise: sym (symmetric), asym (asymmetric), instance (instance-dependent), imb (imbalance-induced noise)')
parser.add_argument('--outlier_rate', type=float, default=0.5, help='proportion of open-set noise among total noise')
parser.add_argument('--imbalance_coefficient', type=float,
                    help='imbalance ratio (min_class_count / max_class_count), used to control label imbalance', default=1.0)

# Model architecture
parser.add_argument('--hidden_dim', type=int, nargs='+', default=[100, 300, 100],
                    help='hidden layer dimensions for MLP, e.g., --hidden_dim 100 300 100')
parser.add_argument('--dropout', type=float, default=0.0, help='dropout rate in deep network model (0 means no dropout)')

# Randomness and device
parser.add_argument('--seed', type=int, default=0, help='random seed for reproducibility')
parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu',
                    help='device to use for training (e.g., cpu, cuda:0)')

# Data loading
parser.add_argument('--dataset', type=str, choices=['cifar10', 'cifar10_feature', 'newsgroup20'], default='newsgroup20',
                    help='dataset to use: cifar10 (raw images), cifar10_feature (Moco_V2 pre-extracted features), newsgroup20 (text data)')
parser.add_argument('--reduced_rate', type=float, default=1.0,
                    help='proportion of dataset to use for training (e.g., 0.1 means 10% of data)')
parser.add_argument('--sample_mode', type=str, choices=['base', 'augment', 'contrastive'], default='contrastive',
                    help='sampling strategy in dataset: base (normal), augment (with data augmentation), contrastive (for contrastive learning)')
parser.add_argument('--alpha', type=float, default=1.0,
                    help='sampling balance ratio for balanced sampling strategies (e.g., 0.5 means resample to 50% of original class distribution)')
parser.add_argument('--num_workers', type=int, default=0,
                    help='number of subprocesses to use for data loading (0 means load in main process)')

args = parser.parse_args()


def random_set(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)


def shuffle_center(num, prototype_position=1):
    # 使用 random 模块打乱索引列表, 创建有序索引列表
    index_list = list(range(num))
    onehot_tensor = torch.zeros((num, num))
    for i, j in enumerate(index_list):
        onehot_tensor[i, j] = prototype_position
    return onehot_tensor


def normalized_softmax(x, num_classes, max_margin, q_desire=0.99):
    # 计算缩放因子
    k = (1 - q_desire) / (num_classes - 1)
    scale = np.log(k) / max_margin

    # 对每一行分别应用 softmax
    e_x = np.exp(x * scale - np.max(x * scale, axis=1, keepdims=True))  # 减去每行的最大值以稳定计算

    # 计算 e_x 的总和，防止出现除以 0 的情况
    e_x_sum = e_x.sum(axis=1, keepdims=True)
    e_x_sum[e_x_sum == 0] = 1e-12  # 防止除以零

    return e_x / e_x_sum


class ClassPrediction:
    def __init__(self, class_centers, noisy_labels, clean_labels, margin, num_epochs, save_features=False):
        """
        Args:
            class_centers: (num_classes, feature_dim)
            noisy_labels: (num_samples,)
            clean_labels: (num_samples,)
            margin: float
            num_epochs: int
            save_features: bool, 是否在最后一个 epoch 保存特征
        """
        self.class_centers = class_centers
        self.num_classes = class_centers.shape[0]
        self.num_samples = len(noisy_labels)
        self.num_epochs = num_epochs
        self.margin = margin
        self.noisy_labels = noisy_labels
        self.clean_labels = clean_labels

        # 存储每一轮的预测结果
        self.pred_storage = np.zeros((num_epochs, self.num_samples), dtype=int)
        self.min_distance_storage = np.zeros((num_epochs, self.num_samples), dtype=float)

        # 是否保存最后一轮的特征
        self.save_features = save_features
        self._final_features = None  # 只保存最后一轮的特征 (N, feature_dim)

    def compute_prob(self, output_features, indices, epoch):
        """
        计算概率、预测标签，并存储结果。
        如果是最后一个 epoch 且 save_features=True，则保存特征。
        """
        device = output_features.device
        dists = torch.cdist(output_features.detach(), self.class_centers.to(device)).cpu().numpy()

        # 转换为概率（基于归一化 softmax）
        prob = normalized_softmax(dists, self.num_classes, self.margin)
        pred = np.argmax(prob, axis=1)
        min_dist = dists.min(axis=1)

        # 存储当前 epoch 的预测和距离
        self.pred_storage[epoch, indices] = pred
        self.min_distance_storage[epoch, indices] = min_dist

        # 如果是最后一个 epoch 且需要保存特征
        if self.save_features and (epoch == self.num_epochs - 1):
            features_np = output_features.detach().cpu().numpy()
            if self._final_features is None:
                self._final_features = [None] * self.num_samples
            for i, idx in enumerate(indices.cpu().numpy()):
                self._final_features[idx] = features_np[i]

        return prob, pred, min_dist

    def get_epoch_predictions(self, epoch):
        """获取指定 epoch 的预测结果"""
        if epoch < 0 or epoch >= self.num_epochs:
            raise ValueError(f"Epoch {epoch} 不合法")
        return self.pred_storage[epoch].copy(), self.min_distance_storage[epoch].copy()

    def get_saved_features(self, args=None):
        """
        获取在最后一个 epoch 保存的特征及相关预测结果（仅当 save_features=True 时才有数据）
        Returns:
            dict 包含 'features', 'clean_labels', 'noisy_labels', 'indices',
                    'predicted_labels', 'min_distance', 'is_clean'
            若未保存或无数据，返回 None
        """
        if not self.save_features or self._final_features is None:
            return None

        # 提取有效样本
        valid_mask = [f is not None for f in self._final_features]
        if not any(valid_mask):
            return None

        features = np.array([f for f in self._final_features if f is not None])
        clean_labels = self.clean_labels[valid_mask]
        noisy_labels = self.noisy_labels[valid_mask]
        indices = np.where(valid_mask)[0]

        # 获取最后一个 epoch 的预测和距离
        last_epoch = self.num_epochs - 1
        pred = self.pred_storage[last_epoch][valid_mask]
        min_dist = self.min_distance_storage[last_epoch][valid_mask]

        # 计算 is_clean：clean 当且仅当 (d <= threshold 且 p == n)
        if args is not None:
            threshold = args.tau * self.margin
            is_clean = np.array([
                1 if (d <= threshold and p == n) else 0
                for d, p, n in zip(min_dist, pred, noisy_labels)
            ])
        else:
            # 如果没有 args，保守设为全 0 或跳过
            is_clean = np.zeros_like(pred)

        return {
            'features': features,
            'clean_labels': clean_labels,
            'noisy_labels': noisy_labels,
            'predicted_labels': pred,
            'min_distance': min_dist,
            'is_clean': is_clean,
            'indices': indices
        }

    def save_features_and_predictions(self, args, net=1):
        """
        将最后一轮的特征、标签、预测结果等保存为 CSV。
        特征本身不直接存入 CSV（维度高），但可后续用于 t-SNE；这里只保存标量信息。
        """
        results = self.get_saved_features(args)
        if results is None:
            print("No features to save.")
            return

        # 构建 DataFrame（不包含高维 features，避免 CSV 膨胀）
        df = pd.DataFrame({
            'index': results['indices'],
            'clean_labels': results['clean_labels'],
            'noisy_labels': results['noisy_labels'],
            'predicted_labels': results['predicted_labels'],
            'min_distance': results['min_distance'],
            'is_clean': results['is_clean']
        })

        save_path = get_save_path(args, save_type="feature", net=net)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df.to_csv(save_path, index=False)
        print(f"Features-related metadata saved to {save_path}")

        # 可选：单独保存 features.npy 用于可视化（如 t-SNE）
        features_save_path = save_path.replace(".csv", ".npy")
        np.save(features_save_path, results['features'])
        print(f"High-dimensional features saved to {features_save_path}")

    def get_saved_features_old(self):
        """
        获取在最后一个 epoch 保存的特征（仅当 save_features=True 时才有数据）
        Returns:
            dict 包含 'features', 'clean_labels', 'noisy_labels', 'indices'
            若未保存或无数据，返回 None
        """
        if not self.save_features or self._final_features is None:
            return None

        # 提取有效样本
        valid_mask = [f is not None for f in self._final_features]
        if not any(valid_mask):
            return None  # 没有实际保存任何特征

        features = np.array([f for f in self._final_features if f is not None])
        clean_labels = self.clean_labels[valid_mask]
        noisy_labels = self.noisy_labels[valid_mask]
        indices = np.where(valid_mask)[0]

        return {
            'features': features,
            'clean_labels': clean_labels,
            'noisy_labels': noisy_labels,
            'indices': indices
        }

    def save_noise_detection_results(self, args, net=1, epoch=None):
        """保存噪声检测结果到 CSV，包含 detection_result 对应的标签和原始噪声标签；
        若指定了 args.initial_learning_ep 且合法，则同时保存 initial_learning_ep 和最后一个 epoch 的结果，中间用 '-' 列隔开。"""

        def _build_epoch_data(epoch_val):
            """内部函数：根据指定 epoch 构建该轮次的检测结果字典"""
            pred = self.pred_storage[epoch_val]
            min_dist = self.min_distance_storage[epoch_val]
            num_classes = self.num_classes

            # detection_result 和 detect_labels
            detection_results = []
            detect_labels = []
            for d, p, n in zip(min_dist, pred, self.noisy_labels):
                if d > args.tau * self.margin:
                    result = "OOD noise"
                    label = 2
                elif p != n:
                    result = "ID noise"
                    label = 1
                else:
                    result = "clean"
                    label = 0
                detection_results.append(result)
                detect_labels.append(label)

            # is_clean 列
            is_clean = [
                0 if (d > args.tau * self.margin or p != n) else 1
                for d, p, n in zip(min_dist, pred, self.noisy_labels)
            ]

            return {
                "predicted_labels": pred,
                "min_distance": min_dist,
                "detection_result": detection_results,
                "detect_labels": detect_labels,
                "is_clean": is_clean,
            }

        # 原始标签部分（不随 epoch 变化）
        raw_results = []
        raw_labels = []
        for clean, noisy in zip(self.clean_labels, self.noisy_labels):
            if clean == noisy:
                raw_result = "clean"
                raw_label = 0
            else:
                if clean == self.num_classes:
                    raw_result = "OOD noise"
                    raw_label = 2
                else:
                    raw_result = "ID noise"
                    raw_label = 1
            raw_results.append(raw_result)
            raw_labels.append(raw_label)

        # 最后一个 epoch（默认）
        last_epoch = self.num_epochs - 1
        if last_epoch < 0 or last_epoch >= self.num_epochs:
            raise ValueError(f"Last epoch {last_epoch} 不合法")

        data_dict = {
            "clean_labels": self.clean_labels,
            "noisy_labels": self.noisy_labels,
            "raw": raw_results,
            "raw_labels": raw_labels,
        }

        # 添加最后一个 epoch 的结果（带后缀）
        last_data = _build_epoch_data(last_epoch)
        for key, val in last_data.items():
            data_dict[f"{key}_ep{last_epoch}"] = val

        # 尝试添加 warmup epoch 的结果（如果存在且合法）
        initial_learning_epoch = getattr(args, 'initial_learning_ep', None)
        initial_learning_epoch = int(initial_learning_epoch)
        print(f"initial_learning_epoch {initial_learning_epoch}")
        if initial_learning_epoch is not None and 0 <= initial_learning_epoch < self.num_epochs:
            initial_learning_data = _build_epoch_data(initial_learning_epoch)
            # 插入分隔列
            sep_col_name = f"--- between ep{initial_learning_epoch}_and_ep{last_epoch} ---"
            data_dict[sep_col_name] = ["-"] * len(self.noisy_labels)
            # 添加 warmup epoch 的结果（带后缀）
            for key, val in initial_learning_data.items():
                data_dict[f"{key}_ep{initial_learning_epoch}"] = val
        else:
            print(f"Warning: initial_learning_ep {initial_learning_epoch} not valid or not set. Only saving final epoch results.")

        # 构建最终 DataFrame
        df = pd.DataFrame(data_dict)

        save_path = get_save_path(args, save_type="filtering", net=net)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df.to_csv(save_path, index=False)
        print(f"Noise detection results saved to {save_path} (epochs: {initial_learning_epoch} and {last_epoch})")

    def save_noise_detection_results_old(self, args, net=1, epoch=None):
        """保存噪声检测结果到 CSV，包含 detection_result 对应的标签和原始噪声标签"""
        if epoch is None:
            epoch = self.num_epochs - 1
        if epoch < 0 or epoch >= self.num_epochs:
            raise ValueError(f"Epoch {epoch} 不合法")

        pred = self.pred_storage[epoch]
        min_dist = self.min_distance_storage[epoch]
        num_classes = self.num_classes

        # 构建 detection_result 和 detect_label
        detection_results = []
        detect_labels = []
        for d, p, n in zip(min_dist, pred, self.noisy_labels):
            if d > args.tau * self.margin:
                result = "OOD noise"
                label = 2
            elif p != n:
                result = "ID noise"
                label = 1
            else:
                result = "clean"
                label = 0
            detection_results.append(result)
            detect_labels.append(label)

        # 构建 raw（原始噪声标签）和 raw_label
        raw_results = []
        raw_labels = []
        for clean, noisy in zip(self.clean_labels, self.noisy_labels):
            if clean == noisy:
                raw_result = "clean"
                raw_label = 0
            else:
                # 假设 clean_labels == num_classes 表示该样本的真实标签是 OOD
                if clean == num_classes:
                    raw_result = "OOD noise"
                    raw_label = 2
                else:
                    raw_result = "ID noise"
                    raw_label = 1
            raw_results.append(raw_result)
            raw_labels.append(raw_label)

        # 创建 DataFrame
        df = pd.DataFrame({
            "clean_labels": self.clean_labels,
            "noisy_labels": self.noisy_labels,
            "predicted_labels": pred,
            "min_distance": min_dist,
            "detection_result": detection_results,
            "detect_labels": detect_labels,
            "raw": raw_results,
            "raw_labels": raw_labels,
            "is_clean": [
                0 if (d > args.tau * self.margin or p != n) else 1
                for d, p, n in zip(min_dist, pred, self.noisy_labels)
            ]
        })

        save_path = get_save_path(args, save_type="detect", net=net)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df.to_csv(save_path, index=False)
        print(f"Noise detection results saved to {save_path} (epoch {epoch})")


class LabelCorrection:
    def __init__(self, noisy_labels, num_classes, margin, tau, pseudo_label_weight):
        """
        将 numpy 硬标签转为 one-hot soft labels
        Args:
            noisy_labels: numpy.ndarray, shape [N]
            num_classes: int
        """
        # 直接转为 torch.LongTensor
        labels_tensor = torch.from_numpy(noisy_labels).long()
        N = labels_tensor.size(0)
        self.noisy_labels = noisy_labels
        self.num_classes = num_classes
        self.margin = margin
        self.tau = tau
        self.pseudo_label_weight = pseudo_label_weight
        # 创建 one-hot soft labels
        self.soft_labels = torch.zeros(N, num_classes)
        self.soft_labels.scatter_(1, labels_tensor.unsqueeze(1), 1.0)

    def get_soft_labels(self, indices):
        return self.soft_labels[indices]

    def update_soft_labels(self, pred1, pred2, min_dist1, min_dist2):
        noisy_labels = self.noisy_labels
        tau_dist = self.tau * self.margin

        # == OOD soft_label
        ood_penalty = np.full(len(noisy_labels), 1.0)
        ood_set_indices = np.where(((min_dist1 > tau_dist) | (min_dist2 > tau_dist)))[0]
        ood_penalty[ood_set_indices] = 0.

        low_confident_ood_indices2 = np.where(((min_dist1 > tau_dist) | (min_dist2 > tau_dist))
                                              & ((pred1 == noisy_labels) | (pred2 == noisy_labels))
                                              & ((min_dist1 < self.margin) | (min_dist2 < self.margin))
                                              )[0]
        ood_penalty[low_confident_ood_indices2] = min(1, self.pseudo_label_weight)

        low_confident_ood_indices = np.where(((min_dist1 > tau_dist) | (min_dist2 > tau_dist))
                                             & ((pred1 == noisy_labels) | (pred2 == noisy_labels))
                                             & ((min_dist1 < self.margin) & (min_dist2 < self.margin))
                                             )[0]
        ood_penalty[low_confident_ood_indices] = min(1, self.pseudo_label_weight)

        # == ID soft_label
        pseudo_label = noisy_labels.copy()
        id_set_indices = np.where(((min_dist1 <= tau_dist) & (min_dist2 <= tau_dist)))[0]
        pseudo_label[id_set_indices] = pred2[id_set_indices]

        pseudo_label_other = noisy_labels.copy()
        pseudo_label_other[id_set_indices] = pred1[id_set_indices]

        pseudo_label_pro = np.full(len(noisy_labels), self.pseudo_label_weight)  # / 2
        high_confident_id_indices = np.where(min_dist2 <= 0.5 * self.margin)[0]
        pseudo_label_pro[high_confident_id_indices] = self.pseudo_label_weight

        original_label_one_hot = np.eye(self.num_classes, dtype=np.float32)[noisy_labels]
        pseudo_soft_label = np.eye(self.num_classes, dtype=np.float32)[pseudo_label]
        pseudo_soft_label_other = np.eye(self.num_classes, dtype=np.float32)[pseudo_label_other]

        soft_labels_new = ood_penalty[:, np.newaxis] * (
                0.5 * pseudo_label_pro[:, np.newaxis] * pseudo_soft_label +
                0.5 * pseudo_label_pro[:, np.newaxis] * pseudo_soft_label_other +
                (1 - pseudo_label_pro)[:, np.newaxis] * original_label_one_hot).copy()
        self.soft_labels = torch.tensor(soft_labels_new, dtype=torch.float32)


"""
# === 划分验证集测试集
"""


def get_train_dataset_split_indices(train_dataset, split_rate, seed=42):
    # 假设这些是你的数据和标签
    noisy_labels = train_dataset.train_noisy_labels

    train_indices, val_indices = train_test_split(
        np.arange(len(noisy_labels)),
        test_size=split_rate,  # 验证集的比例
        random_state=seed,  # 设置随机种子以便复现
        stratify=noisy_labels  # 确保类别分布的一致性
    )

    # 转换为numpy数组并排序
    train_indices = np.array(train_indices)
    val_indices = np.array(val_indices)
    # 确保训练集和验证集没有交集
    assert len(set(train_indices) & set(val_indices)) == 0, "Training and validation indices overlap."

    return train_indices, val_indices


def train_dataset_split_by_indices(dataset, split_indices):
    # == 更新主要的数据集
    dataset.train_noisy_labels = dataset.train_noisy_labels[split_indices]
    dataset.train_labels = dataset.train_labels[split_indices]  # = dataset.labels
    dataset.train_data = dataset.train_data[split_indices]  # = dataset.dataset

    # == 更新辅助信息
    dataset.noise_or_not = np.where(dataset.train_labels == dataset.train_noisy_labels, 1, 0)
    dataset.update_class_weights()  # 更新数据库信息

    return dataset


def train(train_loader, model1, model2, optimizer, criterion, epoch, class_prediction1, class_prediction2, label_correction1,
          label_correction2, args,
          run_type="train"):
    device = args.device

    total_losses = []
    p_losses = []
    n_losses = []
    predicted_indices = []
    predicted_labels1 = []
    predicted_labels2 = []
    clean_labels = []
    noisy_labels = []

    model1.train()
    model2.train()
    all_batch = len(train_loader)
    for i, (x, y_noisy, y_clean, indices) in enumerate(train_loader):
        x1, x2 = x[0].to(device), x[1].to(device)
        y_noisy1, y_noisy2 = y_noisy[0].to(device), y_noisy[1].to(device)
        y_clean1, y_clean2 = y_clean[0].to(device), y_clean[1].to(device)
        indices1, indices2 = indices[0], indices[1]

        soft_y_noisy1, soft_y_noisy2 = label_correction1.get_soft_labels(indices1).to(device), label_correction1.get_soft_labels(
            indices2).to(device)
        outputs1_model1, outputs2_model1 = model1(x1), model1(x2)
        probs1_model1, pred1_model1, min_dist1_model1 = class_prediction1.compute_prob(outputs1_model1, indices1, epoch)
        probs2_model1, pred2_model1, min_dist2_model1 = class_prediction1.compute_prob(outputs2_model1, indices2, epoch)
        loss_model1, p_loss_model1, n_loss_model1 = criterion(outputs1_model1, outputs2_model1, probs1_model1, probs2_model1, soft_y_noisy1,
                                                              soft_y_noisy2)

        # == 设置ood标签
        pred1_model1_adjusted = pred1_model1.copy()
        pred1_model1_adjusted[min_dist1_model1 > args.tau * args.margin] = args.num_classes

        soft_y_noisy1, soft_y_noisy2 = label_correction2.get_soft_labels(indices1).to(device), label_correction2.get_soft_labels(
            indices2).to(device)
        outputs1_model2, outputs2_model2 = model2(x1), model2(x2)
        probs1_model2, pred1_model2, min_dist1_model2 = class_prediction2.compute_prob(outputs1_model2, indices1, epoch)
        probs2_model2, pred2_model2, min_dist2_model2 = class_prediction2.compute_prob(outputs2_model2, indices2, epoch)
        loss_model2, p_loss_model2, n_loss_model2 = criterion(outputs1_model2, outputs2_model2, probs1_model2, probs2_model2, soft_y_noisy1,
                                                              soft_y_noisy2)

        # == 设置ood标签
        pred1_model2_adjusted = pred1_model2.copy()
        pred1_model2_adjusted[min_dist1_model2 > args.tau * args.margin] = args.num_classes

        if run_type == "train":
            optimizer.zero_grad()
            loss_model1.backward()
            loss_model2.backward()
            optimizer.step()
            n = int((i + 1) / (all_batch * 0.05))  # 计算要有多少个小格子显示
            print('\rEp_{4:02d}/{5}-Iteration_{2:02d}/{3}-{0} {1:.0f}% '.format('▉' * n, (i + 1) * (100 / all_batch), i + 1, all_batch,
                                                                                epoch + 1, args.n_epoch), end='')  # :03d 格式代表 001

        # == 保存结果
        total_losses.append(loss_model1.detach().item())
        p_losses.append(p_loss_model1.detach().item())
        n_losses.append(n_loss_model1.detach().item())
        predicted_indices.extend(indices1.detach().tolist())
        predicted_labels1.extend(pred1_model1_adjusted.tolist())  # pred1_model1 / pred1_model1_adjusted
        predicted_labels2.extend(pred1_model2_adjusted.tolist())  # pred1_model1 / pred1_model1_adjusted
        clean_labels.extend(y_clean1.detach().tolist())
        noisy_labels.extend(y_noisy1.detach().tolist())

    total_loss = sum(total_losses) / len(total_losses)
    return clean_labels, noisy_labels, predicted_labels1, predicted_labels2, total_loss


def evaluation(valid_loader, model, ep, args):
    device = args.device
    branch_centers = args.prototypes
    predicted_labels = np.full(len(valid_loader.dataset), -1)
    clean_labels = np.full(len(valid_loader.dataset), -1)
    noisy_labels = np.full(len(valid_loader.dataset), -1)
    min_distances = np.full(len(valid_loader.dataset), 0.0)

    all_batch = len(valid_loader)
    model.eval()
    for i, (x, y_noisy, y_clean, indices) in enumerate(valid_loader):
        x1 = x[0].to(device)
        indices1 = indices[0]

        # == 记录原始列表
        clean_labels[indices1] = y_clean[0].detach().numpy()
        noisy_labels[indices1] = y_noisy[0].detach().numpy()

        # == 获取输出计算预测结果
        x1_outputs = model(x1)

        distance_to_centers = torch.cdist(x1_outputs.detach().cpu(), branch_centers).detach().numpy()
        # 使用 NumPy 的 min 函数找到每一行的最小值
        min_distances[indices1] = np.min(distance_to_centers, axis=1)
        # 使用 NumPy 的 argmin 函数找到每一行最小值的索引
        predicted_labels[indices1] = np.argmin(distance_to_centers, axis=1)

        n = int((i + 1) / (all_batch * 0.05))  # 计算要有多少个小格子显示
        print('\rEp_{4:02d}/{5}-Iteration_{2:02d}/{3}-{0} {1:.0f}% '.format('▉' * n, (i + 1) * (100 / all_batch), i + 1, all_batch,
                                                                            ep + 1, args.n_epoch), end='')  # :03d 格式代表 001

    macro_noisy_acc = balanced_accuracy_score(noisy_labels, predicted_labels)
    macro_clean_acc = balanced_accuracy_score(clean_labels, predicted_labels)
    return macro_noisy_acc, macro_clean_acc


def nlcer(args):
    # == 获取训练集
    train_dataset, train_loader, noisy_labels, clean_labels = get_dataloader(args)

    # == noise_early_stopping
    if args.noise_early_stopping:
        valid_dataset = copy.deepcopy(train_dataset)
        train_indices, val_indices = get_train_dataset_split_indices(train_dataset, args.dataset_split_rate, seed=args.split_seed)
        train_dataset = train_dataset_split_by_indices(train_dataset, train_indices)
        valid_dataset = train_dataset_split_by_indices(valid_dataset, val_indices)

        noisy_labels = train_dataset.train_noisy_labels.copy()
        clean_labels = train_dataset.train_labels.copy()

        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                   batch_size=args.batch_size,
                                                   num_workers=args.num_workers,
                                                   drop_last=False,
                                                   shuffle=True)

        valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset,
                                                   batch_size=args.batch_size,
                                                   num_workers=args.num_workers,
                                                   drop_last=False,
                                                   shuffle=False)
        print(f"--> dataset split")
        print(f"__ split num: train {len(train_dataset)}; valid {len(valid_dataset)}")
        early_stop_logger1 = NoiseEarlyStoppingLogger()
        early_stop_logger2 = NoiseEarlyStoppingLogger()

    log_initial(args, noisy_labels, clean_labels)
    # == 定义类原型
    args.prototypes = shuffle_center(args.num_classes, args.prototype_position)

    # == 定义 model optimizer
    model1 = get_model_function(args).to(args.device)
    model2 = get_model_function(args).to(args.device)
    criterion = RobustPrototypeContrastiveLoss(args.margin, args.num_classes, args)
    parameters = list(model1.parameters()) + list(model2.parameters())
    optimizer = optim.Adam(parameters, lr=args.lr, weight_decay=args.weight_decay)

    # 初始化两个模块
    class_prediction1 = ClassPrediction(args.prototypes, noisy_labels, clean_labels, args.margin, args.n_epoch,
                                        save_features=args.save_features)
    class_prediction2 = ClassPrediction(args.prototypes, noisy_labels, clean_labels, args.margin, args.n_epoch,
                                        save_features=args.save_features)

    label_correction1 = LabelCorrection(noisy_labels, args.num_classes, args.margin, args.tau, args.pseudo_label_weight)
    label_correction2 = LabelCorrection(noisy_labels, args.num_classes, args.margin, args.tau, args.pseudo_label_weight)

    # 初始化储存类
    metric_logger1 = MetricStorage()
    metric_logger2 = MetricStorage()

    # == 训练 + 评估
    for epoch in range(args.n_epoch):
        clean_labels_train, noisy_labels_train, predicted_labels1, predicted_labels2, train_loss = train(train_loader, model1, model2,
                                                                                                         optimizer,
                                                                                                         criterion, epoch,
                                                                                                         class_prediction1,
                                                                                                         class_prediction2,
                                                                                                         label_correction1,
                                                                                                         label_correction2, args=args)
        results1 = eval_noise_ood(args, train_loss, clean_labels_train, noisy_labels_train, predicted_labels1)
        metric_logger1.update(epoch, results1)
        print_eval_results(epoch, results1, args, net=1)  # 详细打印以检测结果

        results2 = eval_noise_ood(args, train_loss, clean_labels_train, noisy_labels_train, predicted_labels2)
        metric_logger2.update(epoch, results2)
        print_eval_results(epoch, results2, args, net=2)  # 详细打印以检测结果

        print(f"___")

        # == 实施早停
        if args.noise_early_stopping:
            macro_noisy_acc1, macro_clean_acc1 = evaluation(valid_loader, model1, epoch, args)
            macro_noisy_acc2, macro_clean_acc2 = evaluation(valid_loader, model2, epoch, args)
            print(f"\r___ Noise valid Macro Accuracy: net1_{macro_noisy_acc1:.3f} | net2_{macro_noisy_acc2:.3f}")

            early_stop_logger1.log(epoch, macro_noisy_acc1, macro_clean_acc1, results1['clean_screen_macro'],
                                   results1['noise_detect_macro'])
            early_stop_logger2.log(epoch, macro_noisy_acc2, macro_clean_acc2, results2['clean_screen_macro'],
                                   results2['noise_detect_macro'])

        # == 跟新伪标签 - 实施标签校正
        if epoch >= args.initial_learning_ep:
            if epoch == args.initial_learning_ep:
                print("Soft label correction started")
            pred1, min_dist1 = class_prediction1.get_epoch_predictions(epoch)
            pred2, min_dist2 = class_prediction2.get_epoch_predictions(epoch)
            label_correction1.update_soft_labels(pred1, pred2, min_dist1, min_dist2)
            label_correction2.update_soft_labels(pred2, pred1, min_dist2, min_dist1)

    # 训练结束后保存
    metric_logger1.save_to_csv(args, net=1)
    metric_logger2.save_to_csv(args, net=2)
    if args.noise_early_stopping:
        early_stop_logger1.save_to_csv(args, net=1)
        early_stop_logger2.save_to_csv(args, net=2)
    else:
        class_prediction1.save_noise_detection_results(args, net=1)
        class_prediction1.save_noise_detection_results(args, net=1)

    # == 保存对应的 t_sne
    if args.save_features:
        results_dict = class_prediction1.get_saved_features(args)
        class_prediction1.save_features_and_predictions(args, net=1)
