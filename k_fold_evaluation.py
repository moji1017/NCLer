import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from datetime import datetime
import os
import warnings
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from scipy.stats import ttest_rel
from data.geodataset import GeoDataset
from networks import get_model_function

warnings.filterwarnings("ignore")

# ====================== 全局配置项 =======================
# 数据集配置
DATASETS_TO_RUN = ['Zircon3']  # 可扩展为 ['Zircon3', 'Apatite12', 'Basalt6']
K_FOLDS = 5
RANDOM_SEED = 0
OUTPUT_DIR = 'outputs/evaluations'
SPLIT_MODE = 'rock_shuffle'  # 仅支持 no_shuffle/random/rock_shuffle
EVAL_MODE = 'clean_test'

# 噪声标签文件路径（根据你的实际路径修改）
NOISE_LABEL_PATHS = {
    'Zircon3': 'outputs/NLCer_filtering/Zircon3/filtering_Zircon3 ep120.csv',
    'Apatite12': 'path/to/apatite12_noise_labels.csv',
    'Garnet7': 'path/to/garnet7_noise_labels.csv',
    'Garnet5': 'path/to/garnet5_noise_labels.csv',
    'Basalt6': 'path/to/basalt6_noise_labels.csv'
}

# PyTorch 训练参数（移除LOSS_TYPE配置）
LEARNING_RATE = 1e-3
EPOCHS = 100
BATCH_SIZE = 128
DROPOUT_PROB = 0.0
HIDDEN_LAYERS = [100, 200, 100]  # MLP隐藏层配置
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
USE_EARLY_STOPPING = True
PATIENCE = 10
MIN_DELTA = 1e-4
VAL_SPLIT_RATIO = 0.2

# ====================== 辅助类/函数 =======================
# 移除FocalLoss类（不再使用）

# 模拟命令行参数对象，适配get_model_function的参数要求
class Args:
    def __init__(self, input_features, output, hidden_dim, dropout, network='MLPs', input_channel=None):
        self.input_features = input_features  # MLPs输入维度
        self.output = output                  # 输出类别数
        self.hidden_dim = hidden_dim          # 隐藏层配置
        self.dropout = dropout                # dropout概率
        self.network = network                # 模型类型（固定为MLPs）
        self.input_channel = input_channel    # CNN用，MLPs设为None

def train_mlp_on_fold(X_train, y_train, X_test, y_test, num_classes, device):
    """
    单折数据的MLP训练与预测（仅使用CE损失）
    返回：测试集预测标签
    """
    # 转换为Tensor
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)

    # 划分训练/验证集（早停用）
    if USE_EARLY_STOPPING:
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train_tensor, y_train_tensor, test_size=VAL_SPLIT_RATIO,
            random_state=RANDOM_SEED, stratify=y_train_tensor.cpu().numpy()
        )
    else:
        X_tr, y_tr = X_train_tensor, y_train_tensor
        X_val, y_val = None, None

    # 创建参数对象并初始化你的MLPs模型
    args = Args(
        input_features=X_train.shape[1],
        output=num_classes,
        hidden_dim=HIDDEN_LAYERS,
        dropout=DROPOUT_PROB
    )
    model = get_model_function(args).to(device)

    # 固定使用交叉熵损失（移除LOSS_TYPE判断）
    criterion = nn.CrossEntropyLoss().to(device)

    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 早停初始化
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    # 训练循环
    for epoch in range(EPOCHS):
        # 训练阶段
        model.train()
        optimizer.zero_grad()
        outputs = model(X_tr)
        loss = criterion(outputs, y_tr)
        loss.backward()
        optimizer.step()

        # 早停验证
        if USE_EARLY_STOPPING:
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val)
                val_loss = criterion(val_outputs, y_val).item()

            # 更新最佳模型
            if val_loss < best_val_loss - MIN_DELTA:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = model.state_dict()
            else:
                patience_counter += 1
                if patience_counter >= PATIENCE:
                    # 加载最佳模型
                    model.load_state_dict(best_model_state)
                    break

    # 测试阶段
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test_tensor)
        pred = torch.argmax(test_outputs, dim=1).cpu().numpy()

    return pred

def run_statistical_tests(results_dict):
    """
    统计检验：配对t检验（比较不同策略的性能）
    返回：统计检验结果DataFrame
    """
    test_results = []
    strategies = list(results_dict.keys())

    # 两两比较
    for i in range(len(strategies)):
        for j in range(i + 1, len(strategies)):
            s1, s2 = strategies[i], strategies[j]
            s1_vals = np.array(results_dict[s1])
            s2_vals = np.array(results_dict[s2])

            # 过滤NaN值
            valid_mask = ~(np.isnan(s1_vals) | np.isnan(s2_vals))
            if valid_mask.sum() < 2:
                test_results.append({
                    'Strategy1': s1,
                    'Strategy2': s2,
                    't-statistic': np.nan,
                    'p-value': np.nan,
                    'Significant (p<0.05)': False
                })
                continue

            # 配对t检验
            t_stat, p_val = ttest_rel(s1_vals[valid_mask], s2_vals[valid_mask])
            test_results.append({
                'Strategy1': s1,
                'Strategy2': s2,
                't-statistic': round(t_stat, 4),
                'p-value': round(p_val, 4),
                'Significant (p<0.05)': p_val < 0.05
            })

    return pd.DataFrame(test_results)

# ====================== 主程序 =======================
def main():
    # 初始化输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"{OUTPUT_DIR}/{timestamp}_cv_results.xlsx"

    # 创建Excel写入器
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        # 遍历每个数据集
        for dataset_name in DATASETS_TO_RUN:
            print(f"\n{'=' * 50}")
            print(f"处理数据集: {dataset_name}")
            print(f"{'=' * 50}")

            # 1. 初始化数据集
            try:
                dataset = GeoDataset(
                    root='./data',
                    data_mode='filter',
                    dataset_name=dataset_name,
                    split_mode=SPLIT_MODE,
                    random_seed=RANDOM_SEED,
                    eval_mode=EVAL_MODE,
                    noise_label_path=NOISE_LABEL_PATHS[dataset_name]
                )
            except Exception as e:
                print(f"初始化数据集失败: {e}")
                continue

            # 2. 初始化交叉验证
            dataset.init_cross_validation(k_folds=K_FOLDS)

            # 3. 存储每折结果
            fold_metrics = {
                'Fold': [],
                'RemoveNoise_Acc': [],
                'RemoveNoise_F1': [],
                'RandomRemove_Acc': [],
                'RandomRemove_F1': [],
                'NoRemove_Acc': [],
                'NoRemove_F1': []
            }

            # 4. 遍历每折
            for fold_idx in range(K_FOLDS):
                print(f"\n--- 第 {fold_idx + 1}/{K_FOLDS} 折 ---")
                fold_data = dataset.get_fold_data(fold_idx)

                # 跳过无干净测试样本的折
                if fold_data is None:
                    print(f"折 {fold_idx + 1}: 无干净测试样本，跳过")
                    continue

                X_train = fold_data['X_train']
                y_train = fold_data['y_train']
                noise_train = fold_data['noise_train']
                X_test = fold_data['X_test']
                y_test = fold_data['y_test']

                # 记录折号
                fold_metrics['Fold'].append(fold_idx + 1)

                # === 策略1: RemoveNoise（仅用干净训练样本）===
                clean_mask = (noise_train == 1)
                if clean_mask.sum() > 0:
                    X_tr_clean = X_train[clean_mask]
                    y_tr_clean = y_train[clean_mask]
                    pred = train_mlp_on_fold(
                        X_tr_clean, y_tr_clean, X_test, y_test,
                        dataset.num_classes, DEVICE
                    )
                    acc = balanced_accuracy_score(y_test, pred)
                    f1 = f1_score(y_test, pred, average='weighted')
                    fold_metrics['RemoveNoise_Acc'].append(round(acc, 4))
                    fold_metrics['RemoveNoise_F1'].append(round(f1, 4))
                    print(f"RemoveNoise - 平衡准确率: {acc:.4f}, F1: {f1:.4f}")
                else:
                    fold_metrics['RemoveNoise_Acc'].append(np.nan)
                    fold_metrics['RemoveNoise_F1'].append(np.nan)
                    print(f"RemoveNoise - 无干净训练样本")

                # === 策略2: RandomRemove（随机移除同等数量样本）===
                keep_mask = np.ones(len(y_train), dtype=bool)
                rng = np.random.default_rng(RANDOM_SEED + fold_idx)

                for label in np.unique(y_train):
                    class_mask = (y_train == label)
                    noisy_in_class = class_mask & (noise_train == 0)
                    n_noisy = noisy_in_class.sum()
                    n_total = class_mask.sum()

                    if n_noisy > 0 and n_noisy < n_total:
                        class_indices = np.where(class_mask)[0]
                        to_remove = rng.choice(class_indices, size=n_noisy, replace=False)
                        keep_mask[to_remove] = False

                if keep_mask.sum() > 0:
                    X_tr_rand = X_train[keep_mask]
                    y_tr_rand = y_train[keep_mask]
                    pred = train_mlp_on_fold(
                        X_tr_rand, y_tr_rand, X_test, y_test,
                        dataset.num_classes, DEVICE
                    )
                    acc = balanced_accuracy_score(y_test, pred)
                    f1 = f1_score(y_test, pred, average='weighted')
                    fold_metrics['RandomRemove_Acc'].append(round(acc, 4))
                    fold_metrics['RandomRemove_F1'].append(round(f1, 4))
                    print(f"RandomRemove - 平衡准确率: {acc:.4f}, F1: {f1:.4f}")
                else:
                    fold_metrics['RandomRemove_Acc'].append(np.nan)
                    fold_metrics['RandomRemove_F1'].append(np.nan)
                    print(f"RandomRemove - 无训练样本")

                # === 策略3: NoRemove（使用全部训练样本）===
                pred = train_mlp_on_fold(
                    X_train, y_train, X_test, y_test,
                    dataset.num_classes, DEVICE
                )
                acc = balanced_accuracy_score(y_test, pred)
                f1 = f1_score(y_test, pred, average='weighted')
                fold_metrics['NoRemove_Acc'].append(round(acc, 4))
                fold_metrics['NoRemove_F1'].append(round(f1, 4))
                print(f"NoRemove - 平衡准确率: {acc:.4f}, F1: {f1:.4f}")

            # 5. 整理每折结果
            fold_df = pd.DataFrame(fold_metrics)

            # 6. 计算汇总统计
            summary_data = {
                'Strategy': ['RemoveNoise', 'RandomRemove', 'NoRemove'],
                'Mean_Acc': [
                    fold_df['RemoveNoise_Acc'].mean(),
                    fold_df['RandomRemove_Acc'].mean(),
                    fold_df['NoRemove_Acc'].mean()
                ],
                'Std_Acc': [
                    fold_df['RemoveNoise_Acc'].std(),
                    fold_df['RandomRemove_Acc'].std(),
                    fold_df['NoRemove_Acc'].std()
                ],
                'Mean_F1': [
                    fold_df['RemoveNoise_F1'].mean(),
                    fold_df['RandomRemove_F1'].mean(),
                    fold_df['NoRemove_F1'].mean()
                ],
                'Std_F1': [
                    fold_df['RemoveNoise_F1'].std(),
                    fold_df['RandomRemove_F1'].std(),
                    fold_df['NoRemove_F1'].std()
                ]
            }
            summary_df = pd.DataFrame(summary_data)
            summary_df = summary_df.round(4)

            # 7. 统计检验
            acc_results = {
                'RemoveNoise': fold_df['RemoveNoise_Acc'].tolist(),
                'RandomRemove': fold_df['RandomRemove_Acc'].tolist(),
                'NoRemove': fold_df['NoRemove_Acc'].tolist()
            }
            f1_results = {
                'RemoveNoise': fold_df['RemoveNoise_F1'].tolist(),
                'RandomRemove': fold_df['RandomRemove_F1'].tolist(),
                'NoRemove': fold_df['NoRemove_F1'].tolist()
            }

            acc_test_df = run_statistical_tests(acc_results)
            f1_test_df = run_statistical_tests(f1_results)

            # 8. 保存到Excel
            fold_df.to_excel(writer, sheet_name=f'{dataset_name}_Folds', index=False)
            summary_df.to_excel(writer, sheet_name=f'{dataset_name}_Summary', index=False)
            acc_test_df.to_excel(writer, sheet_name=f'{dataset_name}_Acc_Test', index=False)
            f1_test_df.to_excel(writer, sheet_name=f'{dataset_name}_F1_Test', index=False)

            print(f"\n数据集 {dataset_name} 结果已保存")
            print(f"汇总结果:\n{summary_df}")

    print(f"\n所有结果已保存至: {output_file}")

# ====================== 运行入口 =======================
if __name__ == '__main__':
    # 设置随机种子
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(RANDOM_SEED)

    main()