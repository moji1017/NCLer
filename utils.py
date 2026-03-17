import numpy as np
from sklearn.metrics import precision_recall_fscore_support
import pandas as pd
import os
import csv
import re
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


# _new
def eval_noise_ood(args, total_loss, clean_labels, noise_labels, predicted_labels):
    """
    扩展版本：返回更丰富的评估指标
    返回: dict 包含各类 macro/micro/per-class 指标
    噪声检测部分按照之前代码的逻辑改进
    """
    clean_labels = np.array(clean_labels)
    noise_labels = np.array(noise_labels)
    predicted_labels = np.array(predicted_labels)
    num_classes = args.num_classes
    ood_label = args.num_classes  # 假设 OOD label 是 num_classes

    results = {}

    # ================================
    # 1. Noise Detection: Macro & Micro (per ID class) - 改进逻辑
    # ================================
    # 找出噪声样本（标签被翻转的样本）
    noise_mask = clean_labels != noise_labels

    # 找出模型正确检测出的噪声样本
    correctly_detected = noise_mask & (predicted_labels != noise_labels)

    # 找出模型错误检测为噪声的样本（实际是干净样本）
    false_detection = ~noise_mask & (predicted_labels != noise_labels)

    # 存储每个类别的指标
    precisions_noise_macro, recalls_noise_macro, f1s_noise_macro = [], [], []
    all_y_true_noise, all_y_pred_noise = [], []

    for c in range(num_classes):
        idx_c = (noise_labels == c)
        if not np.any(idx_c):
            continue

        # 当前类别中的噪声样本
        cls_noise = noise_mask & idx_c

        # 当前类别中被正确检测的噪声样本
        cls_tp = correctly_detected & idx_c

        # 当前类别中被错误检测为噪声的样本
        cls_fp = false_detection & idx_c

        # 当前类别中未被检测出的噪声样本
        cls_fn = cls_noise & ~correctly_detected

        # 计算TP, FP, FN
        tp = cls_tp.sum()
        fp = cls_fp.sum()
        fn = cls_fn.sum()

        # 计算精确率、召回率和F1
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        precisions_noise_macro.append(precision)
        recalls_noise_macro.append(recall)
        f1s_noise_macro.append(f1)

        # 为micro指标准备数据
        y_true_noise_cls = (clean_labels[idx_c] != noise_labels[idx_c]).astype(int)
        y_pred_noise_cls = (predicted_labels[idx_c] != noise_labels[idx_c]).astype(int)
        all_y_true_noise.extend(y_true_noise_cls)
        all_y_pred_noise.extend(y_pred_noise_cls)

    # Macro - 使用改进后的计算
    results['noise_detect_macro'] = {
        'p': np.mean(precisions_noise_macro) if precisions_noise_macro else 0.0,
        'r': np.mean(recalls_noise_macro) if recalls_noise_macro else 0.0,
        'f1': np.mean(f1s_noise_macro) if f1s_noise_macro else 0.0
    }

    # Micro (global over all samples)
    if len(all_y_true_noise) > 0:
        p_mi, r_mi, f1_mi, _ = precision_recall_fscore_support(
            all_y_true_noise, all_y_pred_noise,
            average='binary', pos_label=1, zero_division=0
        )
        results['noise_detect_micro'] = {'p': p_mi, 'r': r_mi, 'f1': f1_mi}
    else:
        results['noise_detect_micro'] = {'p': 0.0, 'r': 0.0, 'f1': 0.0}

    # Per-class noise detection - 使用改进后的逻辑
    per_class_noise = {}
    for c in range(num_classes):
        idx_c = (noise_labels == c)
        if not np.any(idx_c):
            per_class_noise[f'noise_detect_p_{c}'] = 0.0
            per_class_noise[f'noise_detect_r_{c}'] = 0.0
            per_class_noise[f'noise_detect_f1_{c}'] = 0.0
            continue

        # 当前类别中的噪声样本
        cls_noise = noise_mask & idx_c

        # 当前类别中被正确检测的噪声样本
        cls_tp = correctly_detected & idx_c

        # 当前类别中被错误检测为噪声的样本
        cls_fp = false_detection & idx_c

        # 当前类别中未被检测出的噪声样本
        cls_fn = cls_noise & ~correctly_detected

        # 计算TP, FP, FN
        tp = cls_tp.sum()
        fp = cls_fp.sum()
        fn = cls_fn.sum()

        # 计算精确率、召回率和F1
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        per_class_noise[f'noise_detect_p_{c}'] = precision
        per_class_noise[f'noise_detect_r_{c}'] = recall
        per_class_noise[f'noise_detect_f1_{c}'] = f1
    results.update(per_class_noise)

    # ================================
    # 2. Screening: Retain Clean Samples (Macro & Micro & Per-Class)
    # ================================
    # 找出干净样本
    clean_mask = clean_labels == noise_labels

    # 找出模型正确保留的干净样本
    correctly_retained = clean_mask & (predicted_labels == noise_labels)

    # 找出模型错误保留的噪声样本
    incorrectly_retained = ~clean_mask & (predicted_labels == noise_labels)

    precisions_screen_macro, recalls_screen_macro, f1s_screen_macro = [], [], []
    all_y_true_clean, all_y_pred_keep = [], []

    for c in range(num_classes):
        idx_c = (noise_labels == c)
        if not np.any(idx_c):
            continue

        # 当前类别中的干净样本
        cls_clean = clean_mask & idx_c

        # 当前类别中被正确保留的干净样本
        cls_tp = correctly_retained & idx_c

        # 当前类别中被错误保留的噪声样本
        cls_fp = incorrectly_retained & idx_c

        # 当前类别中被错误过滤的干净样本
        cls_fn = cls_clean & (predicted_labels != noise_labels)

        # 计算TP, FP, FN
        tp = cls_tp.sum()
        fp = cls_fp.sum()
        fn = cls_fn.sum()

        # 计算精确率、召回率和F1
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        precisions_screen_macro.append(precision)
        recalls_screen_macro.append(recall)
        f1s_screen_macro.append(f1)

        # 为micro指标准备数据
        y_true_clean_cls = (clean_labels[idx_c] == noise_labels[idx_c]).astype(int)
        y_pred_keep_cls = (predicted_labels[idx_c] == noise_labels[idx_c]).astype(int)
        all_y_true_clean.extend(y_true_clean_cls)
        all_y_pred_keep.extend(y_pred_keep_cls)

    results['clean_screen_macro'] = {
        'p': np.mean(precisions_screen_macro) if precisions_screen_macro else 0.0,
        'r': np.mean(recalls_screen_macro) if recalls_screen_macro else 0.0,
        'f1': np.mean(f1s_screen_macro) if f1s_screen_macro else 0.0
    }

    if len(all_y_true_clean) > 0:
        p_mi, r_mi, f1_mi, _ = precision_recall_fscore_support(
            all_y_true_clean, all_y_pred_keep,
            average='binary', pos_label=1, zero_division=0
        )
        results['clean_screen_micro'] = {'p': p_mi, 'r': r_mi, 'f1': f1_mi}
    else:
        results['clean_screen_micro'] = {'p': 0.0, 'r': 0.0, 'f1': 0.0}

    # Per-class screening - 使用改进后的逻辑
    per_class_screen = {}
    for c in range(num_classes):
        idx_c = (noise_labels == c)
        if not np.any(idx_c):
            per_class_screen[f'clean_screen_p_{c}'] = 0.0
            per_class_screen[f'clean_screen_r_{c}'] = 0.0
            per_class_screen[f'clean_screen_f1_{c}'] = 0.0
            continue

        # 当前类别中的干净样本
        cls_clean = clean_mask & idx_c

        # 当前类别中被正确保留的干净样本
        cls_tp = correctly_retained & idx_c

        # 当前类别中被错误保留的噪声样本
        cls_fp = incorrectly_retained & idx_c

        # 当前类别中被错误过滤的干净样本
        cls_fn = cls_clean & (predicted_labels != noise_labels)

        # 计算TP, FP, FN
        tp = cls_tp.sum()
        fp = cls_fp.sum()
        fn = cls_fn.sum()

        # 计算精确率、召回率和F1
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        per_class_screen[f'clean_screen_p_{c}'] = precision
        per_class_screen[f'clean_screen_r_{c}'] = recall
        per_class_screen[f'clean_screen_f1_{c}'] = f1
    results.update(per_class_screen)

    # ================================
    # 3. OOD Detection: Per-Class, Micro, Macro (保持不变)
    # ================================
    y_true_cls = clean_labels
    y_pred_cls = predicted_labels
    all_labels = list(range(ood_label + 1))

    try:
        p_list, r_list, f1_list, support = precision_recall_fscore_support(
            y_true_cls, y_pred_cls,
            labels=all_labels,
            average=None,
            zero_division=0
        )
    except Exception:
        p_list = r_list = f1_list = [0.0] * len(all_labels)

    # OOD-specific metrics
    results['ood'] = {
        'p': p_list[ood_label],
        'r': r_list[ood_label],
        'f1': f1_list[ood_label]
    }

    # OOD Macro F1 across all classes (including OOD)
    results['ood_macro'] = {
        'p': np.mean(p_list),
        'r': np.mean(r_list),
        'f1': np.mean(f1_list)
    }

    # OOD Micro F1 (global accuracy-like)
    p_mi_all, r_mi_all, f1_mi_all, _ = precision_recall_fscore_support(
        y_true_cls, y_pred_cls,
        average='micro',
        zero_division=0
    )
    results['ood_micro'] = {'p': p_mi_all, 'r': r_mi_all, 'f1': f1_mi_all}

    # Per-class OOD prediction F1 (for ID classes and OOD)
    per_class_ood = {}
    for i in range(len(p_list)):
        name = "ood" if i == ood_label else f"cls_{i}"
        per_class_ood[f'ood_p_{name}'] = p_list[i]
        per_class_ood[f'ood_r_{name}'] = r_list[i]
        per_class_ood[f'ood_f1_{name}'] = f1_list[i]
    results.update(per_class_ood)

    # ================================
    # 4. Average Loss (保持不变)
    # ================================
    results['loss'] = float(np.mean(total_loss))

    return results


def print_eval_results(epoch, results, args, net=None):
    """
    简洁打印关键指标（单行覆盖式输出）
    所有信息在一行输出，若 net 非 None，则开头显示 Net 值。
    """
    # 提取指标
    noise_macro = results.get('noise_detect_macro', {'p': 0.0, 'r': 0.0, 'f1': 0.0})
    screen_macro = results.get('clean_screen_macro', {'p': 0.0, 'r': 0.0, 'f1': 0.0})
    ood_detect = results.get('ood', {'p': 0.0, 'r': 0.0, 'f1': 0.0})
    avg_loss = results.get('loss', 0.0)

    # 构造单行输出字符串
    output = f"Epoch [{epoch + 1:2d}/{args.n_epoch}] | " \
             f"Loss={avg_loss:.3f} | " \
             f"Noise Detect(MP={noise_macro['p']:.3f},MR={noise_macro['r']:.3f},MF1={noise_macro['f1']:.3f}) | " \
             f"Clean Screen(MP={screen_macro['p']:.3f},MR={screen_macro['r']:.3f},MF1={screen_macro['f1']:.3f}) | " \
             f"OOD Detect(P={ood_detect['p']:.3f},R={ood_detect['r']:.3f},F1={ood_detect['f1']:.3f})"

    # 如果 net 存在，在最前面加上 [Net=x]
    if net is not None:
        output = f"[Net={net}] " + output

    # 单行覆盖输出
    print(f"\r{output}", flush=True)


def get_save_path(args, save_type="train", net=1):
    noise_setting = f"{args.noise_type}_{args.noise_rate:g}_o{args.outlier_rate:g}_im{args.imbalance_coefficient:g}"
    # 基础路径：按 method 分文件夹
    if save_type == "train":
        base_dir = f"outputs/{args.method}/{args.dataset}/{noise_setting}"
        print(f"train csv")
    elif save_type == "valid":
        base_dir = f"outputs/NES/{args.dataset}/{noise_setting}"
        print(f"valid csv")
    elif save_type == "filtering":
        base_dir = f"outputs/{args.method}_filtering/{args.dataset}"
    elif save_type == "figure":
        base_dir = f"figures/{args.method}_tsne/{args.dataset}"
    elif save_type == "tsne_csv":
        base_dir = f"figures/_tsne_csv/{args.dataset}"
    elif save_type == "feature":
        base_dir = f"outputs/{args.method}_feature/{args.dataset}"
    else:
        raise ValueError(f"Unknown save_type: {save_type}")

    os.makedirs(base_dir, exist_ok=True)

    # 文件名：time_method_dataset(noise_info)_network.csv
    noise_info = f"{args.noise_type}_{args.noise_rate}_o{args.outlier_rate}_im{args.imbalance_coefficient}"
    dataset = args.dataset

    # 如果是 MLPs，加上 hidden_dim 标记
    seed = args.seed if args.seed == args.split_seed else args.split_seed
    network_str = args.network
    if args.network == 'MLPs':
        network_str += f"{args.hidden_dim}".replace(" ", "")
    method_str = args.method
    if args.method == 'NLCer':
        method_str += (
            f"_margin{args.margin:.1f}_lam{args.lam}_pseudo_label_weight{args.pseudo_label_weight}_alpha{args.alpha}_tau{args.tau}")
    if save_type == "train":
        filename = f"{args.start_time}_{method_str}_{dataset}_({noise_info})_{network_str} net{net}_seed{seed}.csv"
    elif save_type == "figure":
        filename = f"{args.start_time}_{method_str}_{dataset}_({noise_info})_{network_str} net{net}_seed{seed}.jpg"
    elif save_type == "tsne_csv":
        filename = f"{args.start_time}_tsne_{method_str}_{dataset}_({noise_info})_{network_str} net{net}_seed{seed}.csv"
    elif save_type == "valid":
        filename = f"{args.start_time}_valid_{method_str}_{dataset}_({noise_info})_{network_str} net{net}_seed{seed}.csv"
    elif save_type == "filtering":
        if args.dataset_type == 'geological':
            filename = f"filtering_{dataset}.csv"
        else:
            filename = f"{args.start_time}_filtering_{method_str}_{dataset}_({noise_info})_{network_str} net{net}_seed{seed}.csv"
    elif save_type == "feature":
        filename = f"{args.start_time}_feature_{method_str}_{dataset}_({noise_info})_{network_str} net{net}_seed{seed}.csv"

    return os.path.join(base_dir, filename)


def sanitize_filename(s):
    return re.sub(r'[<>:"/\\|?*\[\]]+', '_', s)


"""绘制特征"""
# 定义每个数据集的类别名称
DATASET_CLASS_NAMES = {
    'cifar10': [
        'Airplane', 'Automobile', 'Bird', 'Cat', 'Deer',
        'Dog', 'Frog', 'Horse', 'Ship', 'Truck'
    ],

    'newsgroup20': [
        'Talk Politics Guns', 'Talk Politics Mideast', 'Talk Politics Misc',
        'Talk Religion Christian', 'Talk Religion Misc', 'Religion Christianity',
        'Sci Space', 'Sci Medicine', 'Sci Crypt', 'Sci Electronics',
        'Rec Sports Baseball', 'Rec Sports Hockey', 'Rec Motorcycles', 'Rec Autos',
        'Comp OS', 'Comp Windows', 'Comp XWindow', 'Comp System', 'Comp Graphics',
        'Misc Marketplace'
    ],  # 按主题争议性/重要性从高到低排序
}


def visualize_tsne_and_save(features_list, clean_labels, noisy_labels, args, net=1, epoch=None):
    """
    对收集到的特征进行 t-SNE 可视化，并保存图像和CSV，支持按数据集显示真实类名。

    Args:
        features_list: list of array-like, 长度为 num_samples，每个元素是特征向量
        clean_labels: numpy array (num_samples,)
        noisy_labels: numpy array (num_samples,)
        args: 参数对象（需包含 dataset 和 get_save_path 方法）
        net: 网络编号
        epoch: 当前 epoch（用于标题）
    """
    # 过滤掉未采集的样本
    valid_mask = [f is not None for f in features_list]
    features = np.array([f for f, m in zip(features_list, valid_mask) if m])
    clean_labels = clean_labels[valid_mask]
    noisy_labels = noisy_labels[valid_mask]

    # t-SNE 降维
    X_embedded = TSNE(n_components=2, random_state=42).fit_transform(features)

    # if len(features) > 5000:
    #     idx_subsample = np.random.choice(len(features), size=5000, replace=False)
    #     X_embedded = TSNE(n_components=2, random_state=42).fit_transform(features[idx_subsample])
    #     clean_labels = clean_labels[idx_subsample]
    #     noisy_labels = noisy_labels[idx_subsample]
    # else:
    #     X_embedded = TSNE(n_components=2, random_state=42).fit_transform(features)

    # 保存 CSV
    df = pd.DataFrame({
        "tsne_x": X_embedded[:, 0],
        "tsne_y": X_embedded[:, 1],
        "clean_label": clean_labels,
        "noisy_label": noisy_labels
    })
    csv_path = get_save_path(args, save_type="tsne_csv", net=net)
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df.to_csv(csv_path, index=False)

    # 获取类名映射
    dataset = args.dataset.lower()
    if dataset in DATASET_CLASS_NAMES:
        class_names = DATASET_CLASS_NAMES[dataset]
    else:
        # 默认使用 Class 0, Class 1...
        class_names = [f"Class {i}" for i in range(len(np.unique(clean_labels)))]

    # 绘图
    plt.figure(figsize=(10, 10))
    unique_labels = np.sort(np.unique(clean_labels))  # 保证顺序一致
    colors = plt.cm.get_cmap('tab20', len(unique_labels))

    for i, label in enumerate(unique_labels):
        mask = clean_labels == label
        label_name = class_names[label] if label < len(class_names) else f"Class {label}"
        plt.scatter(X_embedded[mask, 0], X_embedded[mask, 1],
                    c=[colors(i)], label=label_name, alpha=0.7, s=30)

    plt.legend(title="Classes", loc='upper right', fontsize=9, ncol=2)
    plt.title(f"t-SNE Feature Visualization - {dataset.upper()} (Epoch {epoch})", fontsize=14)
    plt.tight_layout()

    # 保存图像
    img_path = get_save_path(args, save_type="figure", net=net)
    plt.savefig(img_path, dpi=150)
    plt.close()

    print(f"✅ t-SNE figure saved to {img_path}")
    print(f"✅ t-SNE data saved to {csv_path}")


class MetricStorage:
    def __init__(self):
        self.history = []  # 存储每个 epoch 的结果

    def _flatten_metrics(self, metrics_dict):
        """
        将嵌套的 metrics 字典展平，所有浮点数保留三位小数
        """
        flat = {}
        for k, v in metrics_dict.items():
            if isinstance(v, dict):
                for sub_k, sub_v in v.items():
                    flat[f"{k}_{sub_k}"] = round(float(sub_v), 3)
            elif isinstance(v, (float, int)):
                flat[k] = round(float(v), 3)
            else:
                # 其他类型（如 list、array）尝试转 float 或设默认值
                try:
                    flat[k] = round(float(v), 3)
                except:
                    flat[k] = 0.0
        return flat

    def update(self, epoch, metrics_dict):
        """
        接收一个 epoch 的 metrics 并加入历史记录
        :param epoch: int, 当前 epoch
        :param metrics_dict: dict from eval_noise_ood
        """
        flat_metrics = self._flatten_metrics(metrics_dict)
        record = {'epoch': epoch}
        record.update(flat_metrics)
        self.history.append(record)

    def save_to_csv(self, args, net=1):
        if not self.history:
            print("No data to save.")
            return

        file_path = get_save_path(args, save_type="train", net=net)
        directory, filename = os.path.split(file_path)
        name, ext = os.path.splitext(filename)

        counter = 1
        while os.path.exists(file_path):
            new_filename = f"{filename}_{counter}{ext}"
            unique_file_path = os.path.join(directory, new_filename)
            counter += 1

        # --- 构建 DataFrame 并插入 param_all（仅第一行有值）---
        df = pd.DataFrame(self.history)
        df['param_all'] = ''  # 先创建空列
        if len(df) > 0:
            df.loc[0, 'param_all'] = args.param_all  # 第一行写入参数信息

        df.to_csv(file_path, index=False)
        print(f"Metrics saved to {file_path}")

    def clear(self):
        """清空存储"""
        self.history.clear()


class NoiseEarlyStoppingLogger:
    def __init__(self):
        self.data = []  # 存 (epoch, noisy_acc, clean_acc, p, r, f1)

    def log(self, epoch, noisy_acc, clean_acc=None, clean_screen_macro=None, noise_detect_macro=None):
        """
        记录每轮训练结果
        :param epoch: 当前训练轮次
        :param noisy_acc: 在噪声标签数据上的准确率
        :param clean_acc: 在干净验证集上的准确率（可选）
        :param clean_screen_macro: 筛选出的 clean 样本的分类报告宏指标，dict 形式 {'p': ..., 'r': ..., 'f1': ...}
        """
        clean_screen_macro = clean_screen_macro or {'p': None, 'r': None, 'f1': None}
        noise_detect_macro = noise_detect_macro or {'p': None, 'r': None, 'f1': None}
        self.data.append((
            epoch,
            noisy_acc,
            clean_acc,
            clean_screen_macro['p'],
            clean_screen_macro['r'],
            clean_screen_macro['f1'],
            noise_detect_macro['p'],
            noise_detect_macro['r'],
            noise_detect_macro['f1'],
        ))

    def save_to_csv(self, args, net):
        if not self.data:
            raise ValueError("No data to save.")

        columns = ['epoch', 'noisy_acc', 'clean_acc', 'screen_p', 'screen_r', 'screen_f1', 'detect_p', 'detect_r', 'detect_f1']
        df = pd.DataFrame(self.data, columns=columns)
        df = df.sort_values('epoch').drop_duplicates(subset='epoch').reset_index(drop=True)
        # 计算滑动平均（略，保持原逻辑）
        epochs = df['epoch'].values
        noisy_accs = df['noisy_acc'].values
        clean_accs = df['clean_acc'].values.astype('float64')
        n = len(df)

        df['noisy_mean_w1'] = None
        df['noisy_mean_w2'] = None
        df['clean_mean_w1'] = None
        df['clean_mean_w2'] = None

        for i in range(n):
            left1_n, right1_n = max(0, i - 1), min(n, i + 2)
            left2_n, right2_n = max(0, i - 2), min(n, i + 3)

            df.loc[i, 'noisy_mean_w1'] = round(noisy_accs[left1_n:right1_n].mean(), 4)
            df.loc[i, 'noisy_mean_w2'] = round(noisy_accs[left2_n:right2_n].mean(), 4)

            if pd.notna(clean_accs[i]):
                clean_slice_w1 = clean_accs[left1_n:right1_n]
                clean_slice_w2 = clean_accs[left2_n:right2_n]
                valid_w1 = clean_slice_w1[~pd.isna(clean_slice_w1)]
                valid_w2 = clean_slice_w2[~pd.isna(clean_slice_w2)]
                if len(valid_w1) > 0:
                    df.loc[i, 'clean_mean_w1'] = round(valid_w1.mean(), 4)
                if len(valid_w2) > 0:
                    df.loc[i, 'clean_mean_w2'] = round(valid_w2.mean(), 4)

        col_order = [
            'epoch',
            'screen_p', 'screen_r', 'screen_f1',
            'detect_p', 'detect_r', 'detect_f1',
            'noisy_acc', 'clean_acc',
            'noisy_mean_w1', 'noisy_mean_w2',
            'clean_mean_w1', 'clean_mean_w2'
        ]
        df = df[col_order]

        # --- 插入 param_all 列：仅第一行有值 ---
        df['param_all'] = ''  # 初始化为空字符串
        if len(df) > 0:
            df.loc[0, 'param_all'] = args.param_all

        # --- 文件名中加入 param 标识 ---
        file_path = get_save_path(args, save_type="valid", net=net)
        directory, filename = os.path.split(file_path)
        name, ext = os.path.splitext(filename)

        counter = 1
        while os.path.exists(file_path):
            new_filename = f"{filename}_{counter}{ext}"
            unique_file_path = os.path.join(directory, new_filename)
            counter += 1

        df.to_csv(file_path, index=False)
        print(f"Saved enhanced results with symmetric moving averages to {file_path}")


def log_initial(args, noisy_labels, clean_labels):
    """简洁版参数汇总函数：仅关注参数组织与日志字符串生成"""

    # 1. 数据集信息
    dataset_info = f"{args.dataset} ip{args.input_features} {args.noise_type}_{args.noise_rate} ol{args.outlier_rate} im{args.imbalance_coefficient}"

    # 2. 损失信息
    loss_info = f"lr_{args.lr}_op{args.optimizer}"

    # 3. 网络信息
    if "MLP" in args.network:
        network_info = f"MLP{args.hidden_dim}"
    else:
        network_info = args.network
    network_info = network_info.replace(" ", "")
    network_info = f"{network_info}(bs{args.batch_size}_out{args.output})"

    # 4. 方法信息（动态获取）
    method_params_map = {
        "NLCer": ['margin', 'initial_learning_ep', "alpha", "lam", 'tau', "pseudo_label_weight", "split_seed"],
    }
    method_params = method_params_map.get(args.method, [])
    method_info = args.method + "_" + " ".join([f"{k}_{getattr(args, k)}" for k in method_params])

    # 5. 数据集详细统计信息
    actual_noise_cls_by_noise = calculate_noise_rate(clean_labels, noisy_labels)  # 每类真实噪声率
    actual_noise_cls_by_raw = calculate_noise_rate(noisy_labels, clean_labels)  # 每类观测噪声率
    train_num = len(noisy_labels)
    noise_cls_num = np.bincount(noisy_labels).tolist()  # 每类含噪声样本数
    raw_cls_num = np.bincount(clean_labels).tolist()  # 每类真实样本数
    train_noise_actually = (noisy_labels != clean_labels).mean()  # 整体闭集噪声率
    train_noise_open = (clean_labels == (noisy_labels.max() + 1)).mean()  # 开集噪声率

    # 转换为字符串（避免 list 打印影响格式）
    stat_info_parts = [
        f"train_num_{train_num}",
        f"overall_noise_rate_{train_noise_actually:.3f}",
        f"open_set_noise_rate_{train_noise_open:.3f}",
        f"noise_cls_num_{noise_cls_num}",
        f"raw_cls_num_{raw_cls_num}",
        f"noise_rate_per_class_observed_{actual_noise_cls_by_raw}",
        f"noise_rate_per_class_actual_{actual_noise_cls_by_noise}",
    ]
    stat_info = " ".join(stat_info_parts)

    # 6. 随机种子信息
    seed_info = f"seed_{args.seed}" if hasattr(args, 'seed') else "seed_unknown"

    # 7. 拼接完整参数字符串
    param_all = (
        f"{network_info} {args.loss}_{loss_info} "
        f"{dataset_info} "
        f"{method_info} "
        f"{stat_info} "
        f"{seed_info}"
    )
    args.param_all = param_all.strip()

    # 输出日志
    print(f"--> 数据集: {dataset_info}")
    print(f"--> 损失: {loss_info}")
    print(f"--> 网络: {network_info}")
    if method_info:
        print(f"--> 方法参数: {method_info}")
    print(f"--> 统计信息: {stat_info}")
    print(f"--> 随机种子: {seed_info}")

    return args


def calculate_noise_rate(labels, noisy_labels):
    """
    计算每个类别的实际噪声率

    参数:
    labels: 原始标签的 NumPy 数组
    noisy_labels: 带有噪声的标签的 NumPy 数组

    返回:
    每个类别的实际噪声率的字典
    """
    labels = np.array(labels)
    noisy_labels = np.array(noisy_labels)

    # 获取所有类别
    unique_labels = np.unique(noisy_labels)

    noise_rates = np.zeros(len(unique_labels))

    for label in unique_labels:
        # 当前类别的索引
        indices = noisy_labels == label

        # 计算噪声数量
        noise_count = np.sum(labels[indices] != noisy_labels[indices])

        # 计算总数量
        total_count = np.sum(indices)

        # 计算噪声率
        noise_rate = 0 if total_count == 0 else round(noise_count / total_count, 3)

        noise_rates[label] = noise_rate

    return noise_rates
