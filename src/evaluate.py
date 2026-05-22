"""
evaluate.py — 评估指标与可视化
================================
功能：
  1. 混淆矩阵（带百分比标注）
  2. ROC 曲线（多模型对比）
  3. 训练历史曲线（Loss / F1 / AUC）
  4. t-SNE 特征可视化
  5. Grad-CAM（1D 信号显著性分析）
  6. 汇总报告（分类报告 + JSON 指标文件）
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")  # 无 GUI 环境
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, f1_score,
    precision_score, recall_score,
)
from sklearn.manifold import TSNE

# 使用支持中文的字体
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False

CLASS_NAMES = ["Background(0)", "DebrisFlow(1)"]
PALETTE = ["#3B82F6", "#EF4444"]  # 蓝/红


# ──────────────────────────────────────────────────────────────────────────────
# 混淆矩阵
# ──────────────────────────────────────────────────────────────────────────────
def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: Path,
    title: str = "Confusion Matrix",
) -> None:
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    n = len(CLASS_NAMES)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(CLASS_NAMES, fontsize=10)
    ax.set_yticklabels(CLASS_NAMES, fontsize=10)
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("True", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")

    for i in range(n):
        for j in range(n):
            color = "white" if cm_norm[i, j] > 0.6 else "black"
            ax.text(j, i, f"{cm[i,j]}\n({cm_norm[i,j]:.1%})",
                    ha="center", va="center", fontsize=10, color=color)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Confusion matrix saved: {save_path}")


# ──────────────────────────────────────────────────────────────────────────────
# ROC Curves (Model Comparison)
# ──────────────────────────────────────────────────────────────────────────────
def plot_roc_curves(
    results: Dict[str, dict],
    save_path: Path,
) -> None:
    """
    results: {model_name: {"labels": ..., "probs": ...}}
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    colors = ["#3B82F6", "#EF4444", "#10B981", "#F59E0B", "#8B5CF6"]

    for idx, (name, res) in enumerate(results.items()):
        fpr, tpr, _ = roc_curve(res["labels"], res["probs"])
        auc = roc_auc_score(res["labels"], res["probs"])
        color = colors[idx % len(colors)]
        ax.plot(fpr, tpr, color=color, lw=2, label=f"{name} (AUC={auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Random")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_title("ROC Curves — Model Comparison", fontsize=12, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ROC curves saved: {save_path}")


# ──────────────────────────────────────────────────────────────────────────────
# 训练历史曲线
# ──────────────────────────────────────────────────────────────────────────────
def plot_training_history(
    history: dict,
    save_path: Path,
    model_name: str = "Model",
) -> None:
    fig = plt.figure(figsize=(12, 4))
    gs = gridspec.GridSpec(1, 3, figure=fig)

    keys_pairs = [
        ("train_loss", "val_loss", "Loss"),
        ("train_f1", "val_f1", "F1 Score"),
        (None, "val_auc", "Val AUC"),
    ]

    for col, (tr_key, va_key, ylabel) in enumerate(keys_pairs):
        ax = fig.add_subplot(gs[col])
        if tr_key and tr_key in history:
            ax.plot(history[tr_key], color=PALETTE[0], lw=2, label="Train")
        if va_key and va_key in history:
            ax.plot(history[va_key], color=PALETTE[1], lw=2, label="Val")
        ax.set_xlabel("Epoch", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(f"{model_name} — {ylabel}", fontsize=10, fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Training history saved: {save_path}")


# ──────────────────────────────────────────────────────────────────────────────
# t-SNE 特征可视化
# ──────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def plot_tsne(
    model: nn.Module,
    loader,
    device: torch.device,
    save_path: Path,
    model_name: str = "Model",
    max_samples: int = 500,
) -> None:
    """提取倒数第二层特征 → t-SNE 降维可视化。"""
    model.eval()

    # 挂载 hook 获取 backbone 输出
    features_list, labels_list = [], []

    def hook_fn(module, inp, out):
        features_list.append(out.detach().cpu())

    # 尝试在最后一个线性层前挂钩
    hook = None
    for name, module in reversed(list(model.named_modules())):
        if isinstance(module, nn.Linear) and module.out_features == 2:
            # 挂在上一个线性层（分类头入口前的最后激活后）
            break
    # 简单方式：在 forward 前截获 global pool 或 CLS 特征
    # 这里直接对输入做池化特征代替（更稳健）
    for X, y in loader:
        X = X.to(device)
        # 取 RMS 和频域统计作为降维源（14维）
        x_np = X.squeeze(1).cpu().numpy()
        from src.features import extract_all_features
        feats = np.array([extract_all_features(xi) for xi in x_np])
        features_list.append(feats)
        labels_list.extend(y.numpy())
        if sum(len(f) for f in features_list) >= max_samples:
            break

    feats_all = np.concatenate(features_list, axis=0)[:max_samples]
    labels_all = np.array(labels_list[:max_samples])

    # 处理 NaN/Inf
    feats_all = np.nan_to_num(feats_all, nan=0.0, posinf=0.0, neginf=0.0)
    # 标准化
    mean = feats_all.mean(axis=0, keepdims=True)
    std = feats_all.std(axis=0, keepdims=True) + 1e-8
    feats_all = (feats_all - mean) / std

    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(feats_all) // 4))
    emb = tsne.fit_transform(feats_all)

    fig, ax = plt.subplots(figsize=(6, 5))
    for cls_idx, (cls_name, color) in enumerate(zip(CLASS_NAMES, PALETTE)):
        mask = labels_all == cls_idx
        ax.scatter(emb[mask, 0], emb[mask, 1], c=color, label=cls_name,
                   alpha=0.7, s=20, edgecolors="none")

    ax.set_title(f"t-SNE Feature Visualization — {model_name}", fontsize=11, fontweight="bold")
    ax.set_xlabel("t-SNE Dim 1")
    ax.set_ylabel("t-SNE Dim 2")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  t-SNE plot saved: {save_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Grad-CAM（1D 版本）
# ──────────────────────────────────────────────────────────────────────────────
class GradCAM1D:
    """对 1D-CNN 计算 Grad-CAM 显著性图。"""

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self._gradients = None
        self._activations = None

        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_backward_hook(self._save_gradient)

    def _save_activation(self, module, inp, out):
        self._activations = out.detach()

    def _save_gradient(self, module, grad_inp, grad_out):
        self._gradients = grad_out[0].detach()

    def compute(self, x: torch.Tensor, class_idx: int = 1) -> np.ndarray:
        """返回与输入信号等长的 CAM 热力图 (normalized 0~1)。"""
        self.model.eval()
        x = x.requires_grad_(True)
        logits = self.model(x)
        self.model.zero_grad()
        logits[0, class_idx].backward()

        # 权重 = 梯度全局平均池化
        weights = self._gradients.mean(dim=-1, keepdim=True)  # (B, C, 1)
        cam = (weights * self._activations).sum(dim=1)         # (B, T)
        cam = torch.clamp(cam, min=0)
        cam = cam / (cam.max() + 1e-8)
        return cam.squeeze().cpu().numpy()


def plot_gradcam(
    signal: np.ndarray,
    cam: np.ndarray,
    save_path: Path,
    fs: float = 200.0,
    title: str = "Grad-CAM Saliency",
) -> None:
    """将原始信号和 CAM 热力图叠加可视化。"""
    t = np.arange(len(signal)) / fs
    # 将 CAM 插值到信号长度
    cam_interp = np.interp(
        np.linspace(0, 1, len(signal)),
        np.linspace(0, 1, len(cam)),
        cam,
    )

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 4), sharex=True)
    ax1.plot(t, signal, color="#3B82F6", lw=0.8, alpha=0.9)
    ax1.set_ylabel("Amplitude", fontsize=9)
    ax1.set_title(title, fontsize=11, fontweight="bold")
    ax1.grid(alpha=0.3)

    ax2.fill_between(t, cam_interp, alpha=0.7, color="#EF4444")
    ax2.plot(t, cam_interp, color="#B91C1C", lw=1)
    ax2.set_ylabel("CAM Score", fontsize=9)
    ax2.set_xlabel("Time (s)", fontsize=9)
    ax2.set_ylim(0, 1.05)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Grad-CAM saved: {save_path}")


# ──────────────────────────────────────────────────────────────────────────────
# 综合评估报告
# ──────────────────────────────────────────────────────────────────────────────
def generate_report(
    model_name: str,
    labels: np.ndarray,
    preds: np.ndarray,
    probs: np.ndarray,
    output_dir: Path,
) -> dict:
    """打印分类报告并保存 JSON 指标文件。"""
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    # 文字报告
    report = classification_report(labels, preds,
                                   target_names=CLASS_NAMES,
                                   output_dict=True, zero_division=0)
    f1 = f1_score(labels, preds, average="binary", zero_division=0)
    auc = roc_auc_score(labels, probs) if len(np.unique(labels)) > 1 else 0.5
    prec = precision_score(labels, preds, average="binary", zero_division=0)
    rec = recall_score(labels, preds, average="binary", zero_division=0)
    acc = float(np.mean(preds == labels))

    metrics = {
        "model": model_name,
        "accuracy": round(acc, 4),
        "f1_binary": round(f1, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "roc_auc": round(auc, 4),
        "classification_report": report,
    }

    # 保存 JSON
    metrics_path = output_dir / f"{model_name}_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    # 混淆矩阵
    plot_confusion_matrix(
        labels, preds,
        save_path=plot_dir / f"{model_name}_confusion_matrix.png",
        title=f"{model_name} — Confusion Matrix",
    )

    print(f"\n{'─'*50}")
    print(f" {model_name} Test Results")
    print(f"{'─'*50}")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  F1 Score : {f1:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall   : {rec:.4f}")
    print(f"  ROC-AUC  : {auc:.4f}")
    print(classification_report(labels, preds, target_names=CLASS_NAMES, zero_division=0))

    return metrics
