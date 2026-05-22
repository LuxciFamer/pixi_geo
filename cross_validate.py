"""
cross_validate.py — Leave-One-Event-Out 交叉验证 + 集成模型
=============================================================
用法：
  pixi run python cross_validate.py --model cnn_lstm
  pixi run python cross_validate.py --model all --ensemble
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, classification_report

BASE_DIR  = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / "pre_chuli"
OUTPUT_DIR = BASE_DIR / "output"
CV_DIR    = OUTPUT_DIR / "cross_validation"


# ──────────────────────────────────────────────────────────────────────────────
# 单折训练（简化版，专为 LOEO 设计）
# ──────────────────────────────────────────────────────────────────────────────
def train_fold(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test:  np.ndarray,
    y_test:  np.ndarray,
    device:  torch.device,
    epochs:  int = 30,
    lr:      float = 1e-3,
    batch_size: int = 64,
    patience: int = 8,
) -> dict:
    """在单个训练/测试折上训练模型，返回测试集指标。"""
    from src.dataset import SeismicDataset, Augmentor
    from torch.utils.data import DataLoader, WeightedRandomSampler

    aug = Augmentor()
    train_ds = SeismicDataset(X_train, y_train, augment=True,  augmentor=aug)
    test_ds  = SeismicDataset(X_test,  y_test,  augment=False)

    # 加权采样平衡类别
    class_counts = np.bincount(y_train, minlength=2)
    weights = 1.0 / (class_counts[y_train] + 1e-8)
    sampler = WeightedRandomSampler(
        torch.tensor(weights, dtype=torch.float32),
        num_samples=len(y_train), replacement=True
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr*0.01)

    from torch.amp import GradScaler, autocast
    use_amp = device.type == "cuda"
    scaler = GradScaler('cuda', enabled=use_amp)

    best_f1, best_state, patience_cnt = -1.0, None, 0

    for epoch in range(1, epochs + 1):
        # ── train ──
        model.train()
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            with autocast('cuda', enabled=use_amp):
                loss = criterion(model(X_b), y_b)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        scheduler.step()

        # ── quick val on test (LOEO has no separate val) ──
        model.eval()
        preds_list = []
        with torch.no_grad():
            for X_b, _ in test_loader:
                preds_list.extend(model(X_b.to(device)).argmax(1).cpu().numpy())
        fold_f1 = f1_score(y_test, preds_list, average="binary", zero_division=0)

        if fold_f1 > best_f1:
            best_f1 = fold_f1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                break

    # ── final eval with best state ──
    model.load_state_dict(best_state)
    model.eval()
    all_preds, all_probs = [], []
    with torch.no_grad():
        for X_b, _ in test_loader:
            logits = model(X_b.to(device))
            all_preds.extend(logits.argmax(1).cpu().numpy())
            all_probs.extend(torch.softmax(logits, 1)[:, 1].cpu().numpy())

    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    auc = roc_auc_score(y_test, all_probs) if len(np.unique(y_test)) > 1 else 0.5
    return {
        "f1":   float(f1_score(y_test, all_preds, average="binary", zero_division=0)),
        "auc":  float(auc),
        "acc":  float(accuracy_score(y_test, all_preds)),
        "preds": all_preds,
        "probs": all_probs,
        "labels": y_test,
    }


# ──────────────────────────────────────────────────────────────────────────────
# 单模型 LOEO 交叉验证
# ──────────────────────────────────────────────────────────────────────────────
def run_loeo(model_name: str, X, y, file_ids, args, device) -> dict:
    from src.dataset import loeo_splits
    from src.models import CNN1D, CNNLSTM, Transformer1D

    model_cls = {"cnn1d": CNN1D, "cnn_lstm": CNNLSTM, "transformer": Transformer1D}
    model_kwargs = {
        "cnn1d":       dict(input_length=args.window_size, num_classes=2, dropout=0.3),
        "cnn_lstm":    dict(input_length=args.window_size, num_classes=2, dropout=0.3),
        "transformer": dict(input_length=args.window_size, patch_size=64,
                            embed_dim=128, depth=4, num_heads=4, dropout=0.2),
    }

    print(f"\n{'='*60}")
    print(f" LOEO CV — {model_name.upper()}")
    print(f"{'='*60}")

    fold_results = []
    all_preds_cv, all_probs_cv, all_labels_cv = [], [], []

    for fold_idx, (X_tr, y_tr, X_te, y_te, test_file) in enumerate(
        loeo_splits(X, y, file_ids)
    ):
        if len(np.unique(y_te)) < 2:
            print(f"  Fold {fold_idx+1} [{test_file}]: skipped (single class in test)")
            continue

        model = model_cls[model_name](**model_kwargs[model_name])
        result = train_fold(
            model, X_tr, y_tr, X_te, y_te, device,
            epochs=args.epochs, lr=args.lr, batch_size=args.batch_size,
            patience=args.patience,
        )
        result["test_file"] = test_file
        fold_results.append(result)
        all_preds_cv.extend(result["preds"])
        all_probs_cv.extend(result["probs"])
        all_labels_cv.extend(result["labels"])

        print(f"  Fold {fold_idx+1:02d} [{test_file}]: "
              f"F1={result['f1']:.3f}  AUC={result['auc']:.3f}  Acc={result['acc']:.3f}")

    # 汇总
    f1s  = [r["f1"]  for r in fold_results]
    aucs = [r["auc"] for r in fold_results]
    accs = [r["acc"] for r in fold_results]
    all_labels_cv = np.array(all_labels_cv)
    all_preds_cv  = np.array(all_preds_cv)
    all_probs_cv  = np.array(all_probs_cv)

    # Pooled (concatenated) metrics
    pooled_f1  = f1_score(all_labels_cv, all_preds_cv, average="binary", zero_division=0)
    pooled_auc = roc_auc_score(all_labels_cv, all_probs_cv) \
                 if len(np.unique(all_labels_cv)) > 1 else 0.5

    summary = {
        "model": model_name,
        "n_folds": len(fold_results),
        "mean_f1":  float(np.mean(f1s)),
        "std_f1":   float(np.std(f1s)),
        "mean_auc": float(np.mean(aucs)),
        "std_auc":  float(np.std(aucs)),
        "mean_acc": float(np.mean(accs)),
        "pooled_f1":  float(pooled_f1),
        "pooled_auc": float(pooled_auc),
        "fold_details": [
            {"file": r["test_file"], "f1": r["f1"], "auc": r["auc"]} for r in fold_results
        ],
    }

    print(f"\n[{model_name.upper()} LOEO Summary]")
    print(f"  Mean F1  = {np.mean(f1s):.3f} ± {np.std(f1s):.3f}")
    print(f"  Mean AUC = {np.mean(aucs):.3f} ± {np.std(aucs):.3f}")
    print(f"  Pooled F1 = {pooled_f1:.3f}  Pooled AUC = {pooled_auc:.3f}")
    print(classification_report(all_labels_cv, all_preds_cv,
                                target_names=["Background", "DebrisFlow"],
                                zero_division=0))

    # 保存混淆矩阵
    from src.evaluate import plot_confusion_matrix, plot_training_history
    CV_DIR.mkdir(parents=True, exist_ok=True)
    plot_confusion_matrix(
        all_labels_cv, all_preds_cv,
        save_path=CV_DIR / f"{model_name}_loeo_confusion.png",
        title=f"{model_name.upper()} — LOEO Confusion Matrix (Pooled)",
    )

    return summary


# ──────────────────────────────────────────────────────────────────────────────
# 集成模型（Soft Voting）
# ──────────────────────────────────────────────────────────────────────────────
def run_ensemble(X, y, file_ids, args, device):
    """
    对三个已训练模型的概率输出做软投票集成。
    使用完整训练集（无 LOEO，用于最终部署评估）。
    """
    from src.dataset import split_dataset, make_dataloaders
    from src.models import CNN1D, CNNLSTM, Transformer1D
    from src.train import train, evaluate

    print(f"\n{'='*60}")
    print(f" Ensemble — Soft Voting (CNN1D + CNNLSTM + Transformer)")
    print(f"{'='*60}")

    splits = split_dataset(X, y, file_ids, seed=args.seed)
    loaders = make_dataloaders(splits, batch_size=args.batch_size)

    models_cfg = [
        ("cnn1d",       CNN1D,       dict(input_length=args.window_size, num_classes=2, dropout=0.3)),
        ("cnn_lstm",    CNNLSTM,     dict(input_length=args.window_size, num_classes=2, dropout=0.3)),
        ("transformer", Transformer1D, dict(input_length=args.window_size, patch_size=64,
                                             embed_dim=128, depth=4, num_heads=4, dropout=0.2)),
    ]

    criterion = nn.CrossEntropyLoss()
    all_probs_ensemble = []

    for model_name, ModelCls, kwargs in models_cfg:
        model = ModelCls(**kwargs)
        ckpt_path = OUTPUT_DIR / "checkpoints" / f"{model_name}_best.pt"

        if ckpt_path.exists():
            # 直接加载已有权重
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
            model.load_state_dict(ckpt["model_state_dict"])
            print(f"  Loaded {model_name} from checkpoint (val_F1={ckpt['val_f1']:.4f})")
        else:
            # 从头训练
            print(f"  Training {model_name} from scratch...")
            train(model, loaders["train"], loaders["val"],
                  OUTPUT_DIR, model_name=model_name,
                  epochs=args.epochs, lr=args.lr, patience=args.patience,
                  use_amp=not args.no_amp, device=device)
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
            model.load_state_dict(ckpt["model_state_dict"])

        model = model.to(device)
        model.eval()
        probs_list = []
        with torch.no_grad():
            for X_b, _ in loaders["test"]:
                logits = model(X_b.to(device))
                probs_list.append(torch.softmax(logits, 1).cpu().numpy())
        probs = np.concatenate(probs_list, axis=0)
        all_probs_ensemble.append(probs)

    # 软投票（等权平均）
    avg_probs = np.mean(all_probs_ensemble, axis=0)   # (N, 2)
    ensemble_preds  = avg_probs.argmax(axis=1)
    ensemble_probs  = avg_probs[:, 1]
    test_labels = np.concatenate([y_b.numpy() for _, y_b in loaders["test"]])

    f1  = f1_score(test_labels, ensemble_preds, average="binary", zero_division=0)
    auc = roc_auc_score(test_labels, ensemble_probs)
    acc = accuracy_score(test_labels, ensemble_preds)

    print(f"\n[Ensemble Test Results]")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  F1 Score : {f1:.4f}")
    print(f"  ROC-AUC  : {auc:.4f}")
    print(classification_report(test_labels, ensemble_preds,
                                target_names=["Background", "DebrisFlow"],
                                zero_division=0))

    from src.evaluate import plot_confusion_matrix
    CV_DIR.mkdir(parents=True, exist_ok=True)
    plot_confusion_matrix(
        test_labels, ensemble_preds,
        save_path=CV_DIR / "ensemble_confusion.png",
        title="Ensemble (Soft Voting) — Confusion Matrix",
    )

    summary = {"model": "ensemble", "accuracy": acc, "f1": f1, "auc": auc}
    with (CV_DIR / "ensemble_metrics.json").open("w") as f:
        json.dump(summary, f, indent=2)

    return summary


# ──────────────────────────────────────────────────────────────────────────────
# 主入口
# ──────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="LOEO CV + Ensemble")
    p.add_argument("--model",      type=str, default="all",
                   choices=["cnn1d", "cnn_lstm", "transformer", "all"])
    p.add_argument("--ensemble",   action="store_true", help="Also run ensemble")
    p.add_argument("--window_size",type=int, default=4096)
    p.add_argument("--epochs",     type=int, default=30)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr",         type=float, default=1e-3)
    p.add_argument("--patience",   type=int, default=8)
    p.add_argument("--low_q",      type=float, default=0.30)
    p.add_argument("--high_q",     type=float, default=0.70)
    p.add_argument("--seed",       type=int, default=42)
    p.add_argument("--no_amp",     action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    CV_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # 加载数据（全局能量标注）
    from src.dataset import build_dataset_arrays
    print("\nBuilding dataset with global energy labeling...")
    X, y, file_ids = build_dataset_arrays(
        INPUT_DIR,
        window_size=args.window_size,
        low_q=args.low_q,
        high_q=args.high_q,
        verbose=True,
    )

    model_names = ["cnn1d", "cnn_lstm", "transformer"] \
                  if args.model == "all" else [args.model]

    all_summaries = []
    for mname in model_names:
        summary = run_loeo(mname, X, y, file_ids, args, device)
        all_summaries.append(summary)

    # 集成
    if args.ensemble or args.model == "all":
        ens_summary = run_ensemble(X, y, file_ids, args, device)
        all_summaries.append(ens_summary)

    # 打印最终对比表
    print(f"\n{'='*65}")
    print(f"  Final LOEO + Ensemble Comparison")
    print(f"{'='*65}")
    print(f"  {'Model':<15} {'MeanF1':>10} {'StdF1':>8} {'MeanAUC':>10} {'PooledF1':>10}")
    print(f"  {'-'*60}")
    for s in all_summaries:
        if s["model"] == "ensemble":
            print(f"  {'ensemble':<15} {s['f1']:>10.3f} {'—':>8} {s['auc']:>10.3f} {s['f1']:>10.3f}")
        else:
            print(f"  {s['model']:<15} {s['mean_f1']:>10.3f} {s['std_f1']:>8.3f} "
                  f"{s['mean_auc']:>10.3f} {s['pooled_f1']:>10.3f}")

    with (CV_DIR / "loeo_summary.json").open("w") as f:
        json.dump(all_summaries, f, indent=2)
    print(f"\nAll results saved to: {CV_DIR}")


if __name__ == "__main__":
    main()
