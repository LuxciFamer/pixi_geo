"""
transformer1d.py — 1D Patch-Transformer 分类模型
==================================================
架构（参考 Vision Transformer 思路，适配 1D 信号）：

  Input (1, L)
  → 分 Patch（固定长度子序列）
  → Patch Embedding（Linear / Conv 投影）
  → [CLS] token 拼接 + Positional Embedding
  → N × TransformerEncoder Block（Multi-Head Attention + FFN）
  → [CLS] token → MLP Head → num_classes

相较于 CNN/LSTM：
  - 无归纳偏置，依赖数据驱动的全局关系建模
  - 在样本量足够时通常优于局部感受野模型
  - 配合数据增强和预训练权重在小数据集表现良好
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────────────────────
# Patch Embedding
# ──────────────────────────────────────────────────────────────────────────────
class PatchEmbedding1D(nn.Module):
    """
    将 1D 信号划分为不重叠的 Patch 并投影到 embed_dim 维空间。

    Parameters
    ----------
    input_length : 信号总长度 L
    patch_size   : 每个 Patch 包含的时间步数
    in_channels  : 输入通道（通常 = 1）
    embed_dim    : 嵌入维度
    """

    def __init__(
        self,
        input_length: int = 4096,
        patch_size: int = 64,
        in_channels: int = 1,
        embed_dim: int = 128,
    ):
        super().__init__()
        assert input_length % patch_size == 0, (
            f"input_length ({input_length}) 必须是 patch_size ({patch_size}) 的整数倍"
        )
        self.num_patches = input_length // patch_size
        # 用卷积实现分 Patch + 线性投影（等价，但更高效）
        self.proj = nn.Conv1d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size, bias=False
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 1, L) → (B, num_patches, embed_dim)"""
        x = self.proj(x)          # (B, embed_dim, num_patches)
        x = x.permute(0, 2, 1)   # (B, num_patches, embed_dim)
        return self.norm(x)


# ──────────────────────────────────────────────────────────────────────────────
# 位置编码（可学习）
# ──────────────────────────────────────────────────────────────────────────────
class LearnablePositionalEncoding(nn.Module):
    def __init__(self, seq_len: int, embed_dim: int):
        super().__init__()
        self.pe = nn.Parameter(torch.zeros(1, seq_len, embed_dim))
        nn.init.trunc_normal_(self.pe, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1), :]


# ──────────────────────────────────────────────────────────────────────────────
# Transformer Block
# ──────────────────────────────────────────────────────────────────────────────
class TransformerBlock(nn.Module):
    """Pre-LN Transformer Block（更稳定的训练）。"""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        attn_dropout: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads,
            dropout=attn_dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout),
        )
        self.drop_path = nn.Dropout(dropout)  # 简化的 drop-path

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention with residual
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed)
        x = x + self.drop_path(attn_out)
        # FFN with residual
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


# ──────────────────────────────────────────────────────────────────────────────
# 完整 Transformer1D 模型
# ──────────────────────────────────────────────────────────────────────────────
class Transformer1D(nn.Module):
    """
    1D Patch-Transformer 分类器（ViT-style）。

    Parameters
    ----------
    input_length : 输入信号长度
    patch_size   : Patch 大小（建议 32 或 64）
    embed_dim    : 嵌入维度
    depth        : Transformer 层数
    num_heads    : 注意力头数
    mlp_ratio    : FFN 扩展比率
    dropout      : Dropout 比率
    num_classes  : 分类数
    """

    def __init__(
        self,
        input_length: int = 4096,
        patch_size: int = 64,
        embed_dim: int = 128,
        depth: int = 4,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.2,
        num_classes: int = 2,
    ):
        super().__init__()

        # 确保可以整除
        if input_length % patch_size != 0:
            # 自动调整 patch_size
            for ps in [32, 64, 128, 16, 8]:
                if input_length % ps == 0:
                    patch_size = ps
                    break

        self.patch_embed = PatchEmbedding1D(input_length, patch_size, 1, embed_dim)
        num_patches = input_length // patch_size

        # [CLS] token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # 位置编码（num_patches + 1 for CLS）
        self.pos_enc = LearnablePositionalEncoding(num_patches + 1, embed_dim)

        self.drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)

        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 1, L) → logits: (B, num_classes)"""
        B = x.size(0)

        # Patch embedding
        x = self.patch_embed(x)           # (B, N, D)

        # Prepend CLS token
        cls = self.cls_token.expand(B, -1, -1)  # (B, 1, D)
        x = torch.cat([cls, x], dim=1)    # (B, N+1, D)

        # Positional encoding
        x = self.pos_enc(x)
        x = self.drop(x)

        # Transformer blocks
        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)

        # 取 CLS token 输出分类
        cls_out = x[:, 0]                 # (B, D)
        return self.head(cls_out)

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.forward(x), dim=-1)
