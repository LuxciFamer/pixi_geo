"""
dataset.py — 山洪泥石流地震波数据集 (v2 - Global Energy Labeling)
===================================================================
v2 改进：
  - 使用全局能量阈值标注（跨所有文件统一分位数）
  - 避免低振幅文件内部被错误标注为泥石流
  - 新增 build_dataset_global() 函数

标注逻辑：
  - 第一遍：收集所有文件所有窗口的 RMS 能量
  - 第二遍：按全局 Q25/Q75 分位数统一划分正负样本
  - 正样本（泥石流）：RMS > 全局 Q_high
  - 负样本（背景）  ：RMS < 全局 Q_low
  - 中间模糊区间丢弃

其余功能同 v1（数据增强、PyTorch Dataset/DataLoader）
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import torch
from scipy.io import loadmat
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

# ──────────────────────────────────────────────────────────────────────────────
# 默认超参数
# ──────────────────────────────────────────────────────────────────────────────
DEFAULT_FS = 200
WINDOW_SIZE = 4096        # ~20.48s @ 200Hz
STRIDE = 2048             # 50% overlap
LOW_QUANTILE = 0.30       # 全局能量分位：低于此 → 背景
HIGH_QUANTILE = 0.70      # 全局能量分位：高于此 → 泥石流
RANDOM_SEED = 42


# ──────────────────────────────────────────────────────────────────────────────
# MAT 加载
# ──────────────────────────────────────────────────────────────────────────────
def _load_mat_signal(path: Path) -> Tuple[np.ndarray, float]:
    mat = loadmat(str(path))
    signal: Optional[np.ndarray] = None
    fs: float = DEFAULT_FS
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
            signal = np.asarray(val, dtype=np.float64).reshape(-1)
    if signal is None:
        raise ValueError(f"No valid signal in {path.name}")
    return signal, fs


# ──────────────────────────────────────────────────────────────────────────────
# 滑动窗口
# ──────────────────────────────────────────────────────────────────────────────
def _sliding_windows(signal: np.ndarray, window_size: int, stride: int) -> np.ndarray:
    n_windows = (len(signal) - window_size) // stride + 1
    idx = np.arange(n_windows)[:, None] * stride + np.arange(window_size)[None, :]
    return signal[idx]


# ──────────────────────────────────────────────────────────────────────────────
# 数据增强
# ──────────────────────────────────────────────────────────────────────────────
class Augmentor:
    def __init__(
        self,
        flip_prob: float = 0.5,
        noise_prob: float = 0.5,
        noise_snr_db: float = 25.0,
        scale_prob: float = 0.5,
        scale_range: Tuple[float, float] = (0.8, 1.2),
        shift_prob: float = 0.3,
    ):
        self.flip_prob = flip_prob
        self.noise_prob = noise_prob
        self.noise_snr_db = noise_snr_db
        self.scale_prob = scale_prob
        self.scale_range = scale_range
        self.shift_prob = shift_prob

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = x.copy()
        if random.random() < self.flip_prob:
            x = x[::-1].copy()
        if random.random() < self.scale_prob:
            x = x * random.uniform(*self.scale_range)
        if random.random() < self.noise_prob:
            rms = np.sqrt(np.mean(x ** 2))
            if rms > 0:
                noise_rms = rms / (10 ** (self.noise_snr_db / 20))
                x = x + np.random.randn(*x.shape) * noise_rms
        if random.random() < self.shift_prob:
            x = np.roll(x, random.randint(0, len(x) - 1))
        return x.astype(np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# 归一化
# ──────────────────────────────────────────────────────────────────────────────
def _normalize(x: np.ndarray) -> np.ndarray:
    std = x.std()
    if std < 1e-12:
        return np.zeros_like(x, dtype=np.float32)
    return ((x - x.mean()) / std).astype(np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# 全局能量标注（核心改进）
# ──────────────────────────────────────────────────────────────────────────────
def build_dataset_arrays(
    input_dir: Path,
    window_size: int = WINDOW_SIZE,
    stride: int = STRIDE,
    low_q: float = LOW_QUANTILE,
    high_q: float = HIGH_QUANTILE,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    全局能量阈值标注版本。
    第一遍收集全局 RMS → 计算全局分位数 → 第二遍统一标注。
    """
    mat_files = sorted(input_dir.glob("*.mat"))
    if not mat_files:
        raise FileNotFoundError(f"No .mat files in {input_dir}")

    # ── 第一遍：收集所有窗口及其 RMS ──────────────────────────────
    all_windows_list: List[np.ndarray] = []
    all_fids_list: List[str] = []
    all_rms_list: List[np.ndarray] = []

    for fpath in mat_files:
        signal, _ = _load_mat_signal(fpath)
        wins = _sliding_windows(signal, window_size, stride)
        rms = np.sqrt(np.mean(wins ** 2, axis=1))
        all_windows_list.append(wins)
        all_fids_list.extend([fpath.stem] * len(wins))
        all_rms_list.append(rms)

    all_rms = np.concatenate(all_rms_list)

    # ── 全局分位数阈值 ────────────────────────────────────────────
    q_low  = np.quantile(all_rms, low_q)
    q_high = np.quantile(all_rms, high_q)

    if verbose:
        print(f"\n[Global Energy Thresholds]")
        print(f"  Q{low_q*100:.0f}  (background ceiling) = {q_low:.3e}")
        print(f"  Q{high_q*100:.0f} (debris-flow floor)  = {q_high:.3e}")

    # ── 第二遍：按全局阈值标注 ────────────────────────────────────
    all_x: List[np.ndarray] = []
    all_y: List[np.ndarray] = []
    all_fids_out: List[str] = []

    offset = 0
    for fpath, wins in zip(mat_files, all_windows_list):
        n = len(wins)
        rms = all_rms[offset: offset + n]
        offset += n

        mask_pos = rms >= q_high
        mask_neg = rms <= q_low

        wins_pos = wins[mask_pos]
        wins_neg = wins[mask_neg]
        fids_pos = [fpath.stem] * len(wins_pos)
        fids_neg = [fpath.stem] * len(wins_neg)

        if verbose:
            print(f"  {fpath.name}: {n} windows -> "
                  f"{len(wins_pos)} pos / {len(wins_neg)} neg "
                  f"(mid={n - len(wins_pos) - len(wins_neg)} discarded)")

        if len(wins_pos) > 0:
            x_pos = np.stack([_normalize(w) for w in wins_pos])
            all_x.append(x_pos)
            all_y.append(np.ones(len(wins_pos), dtype=np.int64))
            all_fids_out.extend(fids_pos)

        if len(wins_neg) > 0:
            x_neg = np.stack([_normalize(w) for w in wins_neg])
            all_x.append(x_neg)
            all_y.append(np.zeros(len(wins_neg), dtype=np.int64))
            all_fids_out.extend(fids_neg)

    X = np.concatenate(all_x, axis=0)
    y = np.concatenate(all_y, axis=0)

    if verbose:
        print(f"\n[Dataset] Total={len(y)}, Pos={y.sum()}, Neg={(y==0).sum()}, "
              f"Ratio={y.mean():.2%}")

    return X, y, all_fids_out


# ──────────────────────────────────────────────────────────────────────────────
# PyTorch Dataset
# ──────────────────────────────────────────────────────────────────────────────
class SeismicDataset(Dataset):
    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        augment: bool = False,
        augmentor: Optional[Augmentor] = None,
    ):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.int64)
        self.augment = augment
        self.augmentor = augmentor or Augmentor()

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        x = self.X[idx]
        if self.augment:
            x = self.augmentor(x)
        return torch.tensor(x, dtype=torch.float32).unsqueeze(0), \
               torch.tensor(self.y[idx], dtype=torch.long)


# ──────────────────────────────────────────────────────────────────────────────
# 数据集划分
# ──────────────────────────────────────────────────────────────────────────────
def split_dataset(
    X: np.ndarray,
    y: np.ndarray,
    file_ids: List[str],
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
    seed: int = RANDOM_SEED,
) -> dict:
    file_ids_arr = np.array(file_ids)
    unique_files = np.unique(file_ids_arr)
    rng = np.random.default_rng(seed)
    rng.shuffle(unique_files)

    n = len(unique_files)
    n_test = max(1, int(n * test_ratio))
    n_val  = max(1, int(n * val_ratio))
    n_train = n - n_val - n_test

    train_files = set(unique_files[:n_train])
    val_files   = set(unique_files[n_train: n_train + n_val])
    test_files  = set(unique_files[n_train + n_val:])

    def _mask(fset):
        return np.array([fid in fset for fid in file_ids_arr])

    splits = {}
    for name, fset in [("train", train_files), ("val", val_files), ("test", test_files)]:
        m = _mask(fset)
        splits[name] = (X[m], y[m])

    print(f"\n[Split] train: {sorted(train_files)}")
    print(f"[Split] val  : {sorted(val_files)}")
    print(f"[Split] test : {sorted(test_files)}")
    for k, (xk, yk) in splits.items():
        print(f"  {k}: {len(yk)} samples (pos={yk.sum()}, neg={(yk==0).sum()})")

    return splits


# ──────────────────────────────────────────────────────────────────────────────
# Leave-One-Event-Out 交叉验证生成器
# ──────────────────────────────────────────────────────────────────────────────
def loeo_splits(
    X: np.ndarray,
    y: np.ndarray,
    file_ids: List[str],
):
    """
    Leave-One-Event-Out (LOEO) generator.
    每次留一个文件作为测试集，其余作为训练集（无验证集）。
    Yields: (X_train, y_train, X_test, y_test, test_file_name)
    """
    file_ids_arr = np.array(file_ids)
    unique_files = np.unique(file_ids_arr)

    for test_file in unique_files:
        test_mask  = file_ids_arr == test_file
        train_mask = ~test_mask
        yield (
            X[train_mask], y[train_mask],
            X[test_mask],  y[test_mask],
            test_file,
        )


# ──────────────────────────────────────────────────────────────────────────────
# DataLoader 工厂
# ──────────────────────────────────────────────────────────────────────────────
def make_dataloaders(
    splits: dict,
    batch_size: int = 64,
    num_workers: int = 0,
    augment_train: bool = True,
) -> dict:
    loaders = {}
    for split, (X_s, y_s) in splits.items():
        aug = augment_train and (split == "train")
        ds = SeismicDataset(X_s, y_s, augment=aug)
        if split == "train":
            class_counts = np.bincount(y_s, minlength=2)
            weights = 1.0 / (class_counts[y_s] + 1e-8)
            sampler = WeightedRandomSampler(
                torch.tensor(weights, dtype=torch.float32),
                num_samples=len(y_s), replacement=True,
            )
            loader = DataLoader(ds, batch_size=batch_size,
                                sampler=sampler, num_workers=num_workers,
                                pin_memory=True)
        else:
            loader = DataLoader(ds, batch_size=batch_size,
                                shuffle=False, num_workers=num_workers,
                                pin_memory=True)
        loaders[split] = loader
    return loaders
