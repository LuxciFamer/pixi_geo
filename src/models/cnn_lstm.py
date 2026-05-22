"""
cnn_lstm.py — CNN + 双向 LSTM 混合模型
=========================================
架构：
  Input (1, L)
  → 1D-CNN 特征提取器（局部时频模式）
  → 分帧重塑：(B, T, C)
  → BiLSTM × 2 层（全局时序依赖）
  → 注意力加权聚合
  → Dropout → Linear(2)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNEncoder(nn.Module):
    """轻量 1D-CNN 前端特征提取器。"""

    def __init__(self, out_channels: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            # Block 1
            nn.Conv1d(1, 16, kernel_size=15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(16), nn.GELU(),
            # Block 2
            nn.Conv1d(16, 32, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(32), nn.GELU(),
            # Block 3
            nn.Conv1d(32, out_channels, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm1d(out_channels), nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 1, L) → (B, C, L//8)"""
        return self.net(x)


class SelfAttention1D(nn.Module):
    """简单的点积自注意力池化（单头）。"""

    def __init__(self, hidden: int):
        super().__init__()
        self.score = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, H) → (B, H)"""
        attn = torch.softmax(self.score(x), dim=1)   # (B, T, 1)
        return (x * attn).sum(dim=1)                  # (B, H)


class CNNLSTM(nn.Module):
    """
    CNN + 双向 LSTM 分类模型。

    Parameters
    ----------
    input_length  : 输入信号长度
    num_classes   : 分类数
    cnn_channels  : CNN 编码器输出通道
    lstm_hidden   : LSTM 隐层维度
    lstm_layers   : LSTM 层数
    dropout       : Dropout 比率
    """

    def __init__(
        self,
        input_length: int = 4096,
        num_classes: int = 2,
        cnn_channels: int = 64,
        lstm_hidden: int = 128,
        lstm_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.cnn = CNNEncoder(out_channels=cnn_channels)

        self.lstm = nn.LSTM(
            input_size=cnn_channels,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )

        self.attn = SelfAttention1D(lstm_hidden * 2)  # bidirectional → *2

        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden * 2, 64),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 1, L) → logits: (B, num_classes)"""
        feat = self.cnn(x)              # (B, C, T)
        feat = feat.permute(0, 2, 1)   # (B, T, C)
        out, _ = self.lstm(feat)        # (B, T, 2H)
        ctx = self.attn(out)            # (B, 2H)
        return self.head(ctx)

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.forward(x), dim=-1)
