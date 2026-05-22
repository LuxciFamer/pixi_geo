"""
cnn1d.py — 1D-CNN 基线分类模型
================================
架构：
  Input (1, L)
  → Conv1D × 4（含残差捷径）
  → GlobalAvgPool
  → Dropout → Linear(2)

采用残差连接（ResNet-style）防止梯度消失，适合较深的 1D 卷积网络。
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock1D(nn.Module):
    """1D 残差块：两层深度可分离卷积 + 跳跃连接。"""

    def __init__(self, channels: int, kernel_size: int = 7, dropout: float = 0.1):
        super().__init__()
        pad = kernel_size // 2
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=pad, bias=False),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size, padding=pad, bias=False),
            nn.BatchNorm1d(channels),
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.net(x) + x)


class CNN1D(nn.Module):
    """
    轻量 1D-ResNet 分类器。

    Parameters
    ----------
    input_length : 输入信号长度（时间步数），默认 4096
    num_classes  : 分类数，二分类 = 2
    base_channels: 第一层卷积通道数，后续逐层翻倍
    dropout      : Dropout 比率
    """

    def __init__(
        self,
        input_length: int = 4096,
        num_classes: int = 2,
        base_channels: int = 32,
        dropout: float = 0.3,
    ):
        super().__init__()

        # 主干：逐阶段下采样
        self.stem = nn.Sequential(
            nn.Conv1d(1, base_channels, kernel_size=15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(base_channels),
            nn.GELU(),
        )

        self.stage1 = nn.Sequential(
            ResBlock1D(base_channels, kernel_size=7, dropout=0.1),
            nn.Conv1d(base_channels, base_channels * 2, kernel_size=3,
                      stride=2, padding=1, bias=False),
            nn.BatchNorm1d(base_channels * 2),
            nn.GELU(),
        )

        self.stage2 = nn.Sequential(
            ResBlock1D(base_channels * 2, kernel_size=7, dropout=0.1),
            nn.Conv1d(base_channels * 2, base_channels * 4, kernel_size=3,
                      stride=2, padding=1, bias=False),
            nn.BatchNorm1d(base_channels * 4),
            nn.GELU(),
        )

        self.stage3 = nn.Sequential(
            ResBlock1D(base_channels * 4, kernel_size=5, dropout=0.1),
            ResBlock1D(base_channels * 4, kernel_size=5, dropout=0.1),
        )

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(base_channels * 4, 64),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 1, L) → logits: (B, num_classes)"""
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.pool(x)
        return self.head(x)

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """返回类别概率 (B, num_classes)。"""
        return F.softmax(self.forward(x), dim=-1)
