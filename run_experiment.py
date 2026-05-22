"""
run_experiment.py — 山洪泥石流地震波二分类实验主入口
=====================================================
用法示例：
  # 训练单个模型
  pixi run python run_experiment.py --model cnn1d

  # 对比所有模型
  pixi run python run_experiment.py --model all

  # 快速验证（少 epoch）
  pixi run python run_experiment.py --model cnn1d --epochs 10 --batch_size 32

  # 仅探索数据，不训练
  pixi run python run_experiment.py --explore_only
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ──────────────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / "pre_chuli"
OUTPUT_DIR = BASE_DIR / "output"


# ──────────────────────────────────────────────────────────────────────────────
# 数据探索（可单独运行）
# ──────────────────────────────────────────────────────────────────────────────
def explore_data(input_dir: Path, output_dir: Path) -> None:
    """可视化所有文件的信号波形和振幅分布。"""
    from scipy.io import loadmat

    output_dir.mkdir(parents=True, exist_ok=True)
    mat_files = sorted(input_dir.glob("*.mat"))

    print(f"\n{'='*60}")
    print(f" 数据探索: {len(mat_files)} 个 MAT 文件")
    print(f"{'='*60}")

    fig, axes = plt.subplots(4, 4, figsize=(20, 12))
    axes = axes.flatten()

    stds = {}
    for idx, fpath in enumerate(mat_files):
        mat = loadmat(str(fpath))
        signal = None
        fs = 200.0
        for key, val in mat.items():
            if key.startswith("__"):
                continue
            if key.lower() == "fs":
                try:
                    fs = float(np.asarray(val).reshape(-1)[0])
                except Exception:
                    pass
                continue
            if isinstance(val, np.ndarray) and val.size > 1:
                signal = np.asarray(val).reshape(-1)

        if signal is None:
            continue

        std = float(signal.std())
        stds[fpath.stem] = std
        duration = len(signal) / fs

        # 降采样到 2000 点显示
        step = max(1, len(signal) // 2000)
        t = np.arange(0, len(signal), step) / fs
        s = signal[::step]

        ax = axes[idx]
        ax.plot(t, s, lw=0.5, color="#3B82F6", alpha=0.8)
        ax.set_title(f"{fpath.stem}\nstd={std:.2e}", fontsize=7, fontweight="bold")
        ax.set_xlabel("s", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.grid(alpha=0.3)

        print(f"  {fpath.name}: len={len(signal)}, fs={fs}, std={std:.2e}")

    # 隐藏多余子图
    for ax in axes[len(mat_files):]:
        ax.set_visible(False)

    plt.suptitle("Seismic Signal Overview — All Files", fontsize=14, fontweight="bold")
    plt.tight_layout()
    waveform_path = output_dir / "data_overview.png"
    fig.savefig(waveform_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"\nWaveform overview saved: {waveform_path}")

    # 能量分布图
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    names = list(stds.keys())
    vals = [stds[n] for n in names]
    colors_bar = ["#EF4444" if v > np.median(vals) else "#3B82F6" for v in vals]
    ax2.bar(range(len(names)), vals, color=colors_bar, alpha=0.8, edgecolor="white")
    ax2.set_xticks(range(len(names)))
    ax2.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax2.set_ylabel("RMS Amplitude (std)", fontsize=10)
    ax2.set_title("Signal Energy per File  (red=high energy, blue=low energy)",
                  fontsize=11, fontweight="bold")
    ax2.set_yscale("log")
    ax2.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    energy_path = output_dir / "energy_distribution.png"
    fig2.savefig(energy_path, dpi=120, bbox_inches="tight")
    plt.close(fig2)
    print(f"Energy distribution saved: {energy_path}")


# ──────────────────────────────────────────────────────────────────────────────
# 数据集构建
# ──────────────────────────────────────────────────────────────────────────────
def build_data(args) -> tuple:
    from src.dataset import build_dataset_arrays, split_dataset, make_dataloaders

    print(f"\n{'='*60}")
    print(" Data Loading & Preprocessing")
    print(f"{'='*60}")

    X, y, file_ids = build_dataset_arrays(
        INPUT_DIR,
        window_size=args.window_size,
        stride=args.stride,
        low_q=args.low_q,
        high_q=args.high_q,
        verbose=True,
    )
    print(f"\nTotal samples: {len(y)}  |  Positive: {y.sum()}  |  Negative: {(y==0).sum()}")

    splits = split_dataset(X, y, file_ids, seed=args.seed)
    loaders = make_dataloaders(
        splits, batch_size=args.batch_size,
        augment_train=not args.no_augment,
    )
    return splits, loaders


# ──────────────────────────────────────────────────────────────────────────────
# 单模型训练与评估
# ──────────────────────────────────────────────────────────────────────────────
def run_single_model(
    model_name: str,
    args,
    loaders: dict,
    device: torch.device,
) -> dict:
    from src.models import CNN1D, CNNLSTM, Transformer1D
    from src.train import train, evaluate
    from src.evaluate import generate_report, plot_training_history, plot_tsne

    model_cls = {"cnn1d": CNN1D, "cnn_lstm": CNNLSTM, "transformer": Transformer1D}
    model_kwargs = {
        "cnn1d": dict(input_length=args.window_size, num_classes=2, dropout=0.3),
        "cnn_lstm": dict(input_length=args.window_size, num_classes=2, dropout=0.3),
        "transformer": dict(input_length=args.window_size, patch_size=64,
                            embed_dim=128, depth=4, num_heads=4, dropout=0.2),
    }

    model = model_cls[model_name](**model_kwargs[model_name])
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel: {model_name.upper()}  |  Parameters: {n_params:,}")

    import torch.nn as nn
    criterion = nn.CrossEntropyLoss()

    history = train(
        model=model,
        train_loader=loaders["train"],
        val_loader=loaders["val"],
        output_dir=OUTPUT_DIR,
        model_name=model_name,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        use_amp=not args.no_amp,
        device=device,
    )

    # 加载最佳权重评估测试集
    ckpt_path = OUTPUT_DIR / "checkpoints" / f"{model_name}_best.pt"
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"Loaded best checkpoint (epoch={ckpt['epoch']}, val_F1={ckpt['val_f1']:.4f})")

    from src.train import evaluate as eval_fn
    test_metrics = eval_fn(model, loaders["test"], criterion, device,
                           use_amp=not args.no_amp)

    report = generate_report(
        model_name=model_name,
        labels=test_metrics["labels"],
        preds=test_metrics["preds"],
        probs=test_metrics["probs"],
        output_dir=OUTPUT_DIR,
    )

    # 训练曲线
    plot_dir = OUTPUT_DIR / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    plot_training_history(
        history,
        save_path=plot_dir / f"{model_name}_training_history.png",
        model_name=model_name.upper(),
    )

    # t-SNE 可视化
    plot_tsne(
        model, loaders["test"], device,
        save_path=plot_dir / f"{model_name}_tsne.png",
        model_name=model_name.upper(),
    )

    return {
        "model_name": model_name,
        "model": model,
        "history": history,
        "test_metrics": test_metrics,
        "report": report,
    }


# ──────────────────────────────────────────────────────────────────────────────
# 多模型对比
# ──────────────────────────────────────────────────────────────────────────────
def compare_models(all_results: list, output_dir: Path) -> None:
    from src.evaluate import plot_roc_curves

    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    roc_data = {}
    summary_rows = []

    for res in all_results:
        name = res["model_name"]
        tm = res["test_metrics"]
        roc_data[name] = {"labels": tm["labels"], "probs": tm["probs"]}
        summary_rows.append({
            "model": name,
            "accuracy": f"{tm['acc']:.4f}",
            "f1": f"{tm['f1']:.4f}",
            "auc": f"{tm['auc']:.4f}",
        })

    # ROC 对比图
    plot_roc_curves(roc_data, save_path=plot_dir / "roc_comparison.png")

    # 打印汇总表
    print(f"\n{'='*55}")
    print(f"  Model Comparison Summary")
    print(f"{'='*55}")
    print(f"  {'Model':<15} {'Accuracy':>10} {'F1':>10} {'AUC':>10}")
    print(f"  {'-'*50}")
    for row in summary_rows:
        print(f"  {row['model']:<15} {row['accuracy']:>10} {row['f1']:>10} {row['auc']:>10}")
    print(f"{'='*55}")

    summary_path = output_dir / "model_comparison.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary_rows, f, ensure_ascii=False, indent=2)
    print(f"\nComparison summary saved: {summary_path}")


# ──────────────────────────────────────────────────────────────────────────────
# 主函数
# ──────────────────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="山洪泥石流地震波二分类实验"
    )
    # 数据参数
    parser.add_argument("--window_size", type=int, default=4096,
                        help="滑动窗口长度（采样点）")
    parser.add_argument("--stride", type=int, default=2048,
                        help="滑动步幅")
    parser.add_argument("--low_q", type=float, default=0.25,
                        help="负样本能量分位阈值")
    parser.add_argument("--high_q", type=float, default=0.75,
                        help="正样本能量分位阈值")
    # 模型参数
    parser.add_argument("--model", type=str, default="all",
                        choices=["cnn1d", "cnn_lstm", "transformer", "all"],
                        help="选择模型")
    # 训练参数
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=12,
                        help="早停耐心值")
    parser.add_argument("--seed", type=int, default=42)
    # 开关
    parser.add_argument("--no_amp", action="store_true",
                        help="禁用混合精度训练")
    parser.add_argument("--no_augment", action="store_true",
                        help="禁用数据增强")
    parser.add_argument("--explore_only", action="store_true",
                        help="仅探索数据，不训练")
    return parser.parse_args()


def main():
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    t_start = time.time()
    print("\n" + "=" * 60)
    print("  Debris Flow / Flash Flood Seismic Binary Classification")
    print("=" * 60)

    # 数据探索
    explore_data(INPUT_DIR, OUTPUT_DIR)

    if args.explore_only:
        print("\n[explore_only 模式] 跳过训练。")
        return

    # 确定设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(f"\nGPU: {torch.cuda.get_device_name(0)}")
    else:
        print("\nUsing CPU (no CUDA GPU detected)")

    # 固定随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # 构建数据集
    splits, loaders = build_data(args)

    # 确定要训练的模型列表
    if args.model == "all":
        model_names = ["cnn1d", "cnn_lstm", "transformer"]
    else:
        model_names = [args.model]

    # 逐模型训练
    all_results = []
    for model_name in model_names:
        result = run_single_model(model_name, args, loaders, device)
        all_results.append(result)

    # 多模型对比（仅当多于一个模型时）
    if len(all_results) > 1:
        compare_models(all_results, OUTPUT_DIR)

    elapsed = time.time() - t_start
    print(f"\nTotal elapsed: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"All outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
