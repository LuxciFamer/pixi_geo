"""
train.py — 统一训练循环
=========================
支持：
  - 混合精度训练（AMP）
  - 早停（Early Stopping）
  - 余弦退火学习率调度
  - 自动保存最佳权重（按验证 F1）
  - 训练日志到 CSV
"""
from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from sklearn.metrics import f1_score, roc_auc_score
import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# 早停
# ──────────────────────────────────────────────────────────────────────────────
class EarlyStopping:
    def __init__(self, patience: int = 10, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score: Optional[float] = None
        self.should_stop = False

    def step(self, score: float) -> bool:
        """返回 True 表示需要触发早停。"""
        if self.best_score is None or score > self.best_score + self.min_delta:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


# ──────────────────────────────────────────────────────────────────────────────
# 单 epoch 训练
# ──────────────────────────────────────────────────────────────────────────────
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scaler: GradScaler,
    use_amp: bool = True,
) -> dict:
    model.train()
    total_loss = 0.0
    all_preds, all_labels = [], []

    for X, y in loader:
        X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad()

        with autocast('cuda', enabled=use_amp):
            logits = model(X)
            loss = criterion(logits, y)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * len(y)
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(y.cpu().numpy())

    n = len(all_labels)
    avg_loss = total_loss / n
    f1 = f1_score(all_labels, all_preds, average="binary", zero_division=0)
    acc = np.mean(np.array(all_preds) == np.array(all_labels))
    return {"loss": avg_loss, "f1": f1, "acc": float(acc)}


# ──────────────────────────────────────────────────────────────────────────────
# 验证/测试评估
# ──────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool = True,
) -> dict:
    model.eval()
    total_loss = 0.0
    all_probs, all_preds, all_labels = [], [], []

    for X, y in loader:
        X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with autocast('cuda', enabled=use_amp):
            logits = model(X)
            loss = criterion(logits, y)

        total_loss += loss.item() * len(y)
        probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        preds = logits.argmax(dim=1).cpu().numpy()
        all_probs.extend(probs)
        all_preds.extend(preds)
        all_labels.extend(y.cpu().numpy())

    n = len(all_labels)
    avg_loss = total_loss / n
    f1 = f1_score(all_labels, all_preds, average="binary", zero_division=0)
    acc = np.mean(np.array(all_preds) == np.array(all_labels))

    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = 0.5

    return {
        "loss": avg_loss,
        "f1": f1,
        "acc": float(acc),
        "auc": auc,
        "probs": np.array(all_probs),
        "preds": np.array(all_preds),
        "labels": np.array(all_labels),
    }


# ──────────────────────────────────────────────────────────────────────────────
# 完整训练函数
# ──────────────────────────────────────────────────────────────────────────────
def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    output_dir: Path,
    model_name: str = "model",
    epochs: int = 50,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 12,
    use_amp: bool = True,
    device: Optional[torch.device] = None,
) -> dict:
    """
    训练模型并保存最佳权重。

    Returns
    -------
    history: dict with lists 'train_loss', 'val_loss', 'val_f1', 'val_auc'
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*60}")
    print(f" Model: {model_name}  |  Device: {device}")
    print(f" Epochs: {epochs}  |  LR: {lr}  |  AMP: {use_amp}")
    print(f"{'='*60}")

    model = model.to(device)

    # 类别权重（针对不平衡）—— 此处已通过 WeightedSampler 处理，直接用 CE
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=lr * 0.01
    )
    scaler = GradScaler('cuda', enabled=use_amp and device.type == "cuda")
    early_stop = EarlyStopping(patience=patience)

    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt = ckpt_dir / f"{model_name}_best.pt"

    log_path = output_dir / f"{model_name}_train_log.csv"
    history = {k: [] for k in ["train_loss", "train_f1", "val_loss", "val_f1", "val_auc"]}

    best_val_f1 = -1.0
    t0 = time.time()

    with log_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "train_f1",
                                               "val_loss", "val_f1", "val_auc", "lr"])
        writer.writeheader()

        for epoch in range(1, epochs + 1):
            train_metrics = train_one_epoch(
                model, train_loader, optimizer, criterion, device, scaler, use_amp
            )
            val_metrics = evaluate(model, val_loader, criterion, device, use_amp)
            scheduler.step()

            cur_lr = optimizer.param_groups[0]["lr"]
            row = {
                "epoch": epoch,
                "train_loss": f"{train_metrics['loss']:.4f}",
                "train_f1": f"{train_metrics['f1']:.4f}",
                "val_loss": f"{val_metrics['loss']:.4f}",
                "val_f1": f"{val_metrics['f1']:.4f}",
                "val_auc": f"{val_metrics['auc']:.4f}",
                "lr": f"{cur_lr:.6f}",
            }
            writer.writerow(row)
            f.flush()

            for k in history:
                key_map = {
                    "train_loss": train_metrics["loss"],
                    "train_f1": train_metrics["f1"],
                    "val_loss": val_metrics["loss"],
                    "val_f1": val_metrics["f1"],
                    "val_auc": val_metrics["auc"],
                }
                history[k].append(key_map[k])

            # 保存最佳模型（按验证 F1）
            if val_metrics["f1"] > best_val_f1:
                best_val_f1 = val_metrics["f1"]
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_f1": best_val_f1,
                    "val_auc": val_metrics["auc"],
                }, best_ckpt)

            elapsed = time.time() - t0
            print(
                f"Ep {epoch:03d}/{epochs} | "
                f"TrLoss={train_metrics['loss']:.4f} TrF1={train_metrics['f1']:.3f} | "
                f"VaLoss={val_metrics['loss']:.4f} VaF1={val_metrics['f1']:.3f} "
                f"AUC={val_metrics['auc']:.3f} | "
                f"LR={cur_lr:.5f} | {elapsed:.0f}s"
            )

            if early_stop.step(val_metrics["f1"]):
                print(f"  >> Early stopping at epoch {epoch}. Best val_F1={best_val_f1:.4f}")
                break

    print(f"\nTraining complete. Best Val F1 = {best_val_f1:.4f}")
    print(f"Checkpoint saved: {best_ckpt}")
    return history
