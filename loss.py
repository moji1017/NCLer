import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import numpy as np
import random, sys, os
import math

import torch
import math


def js_divergence_matrix(A_softmax, B_softmax, device='cpu', js_type="count"):
    epsilon = 1e-8
    # Ensure non-zero division by adding epsilon
    # A_softmax = A_softmax + epsilon
    # B_softmax = B_softmax + epsilon
    if not isinstance(A_softmax, torch.Tensor):
        # 如果不是，则将其转换为 torch.Tensor
        A_softmax = torch.tensor(A_softmax + epsilon, device=device)
        B_softmax = torch.tensor(B_softmax + epsilon, device=device)
    else:
        A_softmax = A_softmax + epsilon
        B_softmax = B_softmax + epsilon

    # Calculate the average distributions M
    M = (A_softmax[:, None, :] + B_softmax[None, :, :]) / 2

    # Calculate KL divergences
    kl_div_AM = torch.sum(A_softmax[:, None, :] * torch.log(A_softmax[:, None, :] / M), dim=-1)
    kl_div_BM = torch.sum(B_softmax[None, :, :] * torch.log(B_softmax[None, :, :] / M), dim=-1)

    # Calculate JS divergence
    js_matrix = 0.5 * kl_div_AM + 0.5 * kl_div_BM
    if js_type == "loss":
        return js_matrix  # js_matrix  #

    return js_matrix.detach().cpu().numpy()


class RobustPrototypeContrastiveLoss(nn.Module):
    def __init__(self, margin, num_classes, args=None):
        super(RobustPrototypeContrastiveLoss, self).__init__()
        self.margin = margin
        self.num_classes = num_classes
        self.branch_centers = args.prototypes.to(args.device)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.args = args
        self.lam = args.lam  # 根据是否存在开集噪声决定是否启动 ood_loss 0 if args.outlier_rate == 0 else

    def forward(self, x1, x2, x_prob1, x_prob2, l1, l2):
        device = self.device

        # === 计算正负损失
        posi_loss = self.positive_loss(x1, x2, x_prob1, x_prob2, l1, l2, device)
        nega_loss = self.negative_loss(x1, x2, l1, l2, x_prob1, x_prob2, device)
        ood_loss = self.ood_loss(x1, x2, x_prob1, x_prob2, l1, l2, device)
        loss = posi_loss + nega_loss + self.lam * ood_loss

        return loss, posi_loss, nega_loss

    """
    OOD Loss
    """

    def ood_loss(self, x1, x2, x_prob1, x_prob2, l1, l2, device="cpu"):
        # === 添加上对应的ood损失
        x1_ood_loss = self.get_single_ood_loss(x1, l1, device=device)
        x2_ood_loss = self.get_single_ood_loss(x2, l2, device=device)
        ood_loss = (torch.sum(x1_ood_loss) + torch.sum(x2_ood_loss)) / 2
        return ood_loss

    def get_single_ood_loss(self, x, l, device="cpu"):
        # === 计算单组 ood 样品的损失
        """计算每一行的方差 / 标准差"""
        x = torch.cdist(x, self.branch_centers)
        ood_margin = self.margin
        one_hot_labels = torch.full((len(l), self.num_classes), 1.0).float().to(device)
        x_ood_distance = torch.clamp(ood_margin - x, min=torch.tensor(0., device=device)) * one_hot_labels
        x_ood_coffi = torch.min(1 - x_ood_distance / ood_margin, dim=1)[0].detach()

        variance = torch.sum((x_ood_distance / ood_margin) ** 2, dim=1)

        return variance * x_ood_coffi

    """
    Positive Loss
    """

    def positive_loss(self, x1, x2, x_prob1, x_prob2, l1, l2, device="cpu"):
        x1_distance = self.get_all_positive_loss(x1, l1, x_prob1, device=device)
        x2_distance = self.get_all_positive_loss(x2, l2, x_prob2, device=device)
        positive_loss = (torch.sum(x1_distance) + torch.sum(x2_distance)) / 2
        return positive_loss

    # == 原始代码
    def get_all_positive_loss(self, x, l_soft, x_prob, device="cpu"):
        # == 这里的l是软标签格式而非标签格式
        # === 计算单组正样品的损失
        onehot_matrix = torch.eye(self.num_classes, self.num_classes).to(device)
        # # == 按照原始方式获取JS散度
        x_l_js = js_divergence_matrix(x_prob, onehot_matrix.detach().cpu().numpy())
        x_l_js = torch.tensor(1 - x_l_js).to(device)

        positive_distance = torch.norm(x.unsqueeze(1) - onehot_matrix * self.args.prototype_position, dim=2)

        positive_distance = positive_distance * l_soft * x_l_js
        # 对每个输入 x 和所有中心点的距离求和
        positive_distance = torch.sum(positive_distance, dim=1)

        return positive_distance

    """
    Negative Loss
    """

    def negative_loss(self, x1, x2, l1, l2, x1_prob, x2_prob, device="cpu"):
        eps = torch.tensor(1e-10, device=device)

        negative_coffi = 1 - (l1[:, np.newaxis, :] * l2[np.newaxis, :, :]).sum(dim=2)
        is_negative = negative_coffi.to(device)

        x1_x2_matrix_js = js_divergence_matrix(x1_prob, x2_prob, device=device, js_type="count")
        x1_x2_js_matrix = torch.tensor(x1_x2_matrix_js).detach().to(device)

        euclidean_distance = torch.cdist(x1, x2) + eps  # 防止存在 0 值导致后续求损失出错 #**2
        negative_margin = self.args.margin  # * self.args.negative_threshold
        negative_pairs = torch.clamp(negative_margin - euclidean_distance, min=torch.tensor(0.0, device=device))

        negative_pairs = negative_pairs * is_negative.to(torch.float) * x1_x2_js_matrix
        # 计算每一行非零元素的个数
        non_zero_counts = (negative_pairs > 0).sum(dim=1)
        # 计算每一行的和
        negative_pairs = negative_pairs.sum(dim=1)
        non_zero_counts[non_zero_counts == 0] = 1
        # 用每一行的和除以每一行对应的非零的个数
        # 为了避免除以零的错误，可以使用torch.where来处理
        negative_pairs = negative_pairs / non_zero_counts
        # negative_pairs = negative_pairs / is_negative.sum(dim=1)

        return torch.sum(negative_pairs)  # / is_negative_num.sum().item()
