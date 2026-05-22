"""
features.py — 地震波特征提取
==============================
提供两类特征：
  1. 时频谱特征：STFT → log-mel 谱图（用于 2D-CNN 输入，可选）
  2. 时域/频域统计特征（用于传统 ML 对比基线）
"""
from __future__ import annotations

from typing import Tuple

import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# 时域统计特征
# ──────────────────────────────────────────────────────────────────────────────
def time_domain_features(x: np.ndarray) -> np.ndarray:
    """
    提取 8 维时域统计特征：
    [rms, peak, crest_factor, kurtosis, skewness, zero_crossing_rate, mean_abs, std]
    """
    x = x.astype(np.float64)
    rms = np.sqrt(np.mean(x ** 2))
    peak = np.max(np.abs(x))
    crest = peak / (rms + 1e-12)
    # 峰度（Fisher 定义）
    n = len(x)
    mu = np.mean(x)
    sigma = np.std(x) + 1e-12
    kurtosis = np.mean(((x - mu) / sigma) ** 4) - 3
    skewness = np.mean(((x - mu) / sigma) ** 3)
    # 零穿越率
    zcr = np.sum(np.diff(np.sign(x)) != 0) / (n - 1)
    mean_abs = np.mean(np.abs(x))
    std = np.std(x)
    return np.array([rms, peak, crest, kurtosis, skewness, zcr, mean_abs, std],
                    dtype=np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# 频域统计特征（基于 FFT）
# ──────────────────────────────────────────────────────────────────────────────
def freq_domain_features(x: np.ndarray, fs: float = 200.0) -> np.ndarray:
    """
    提取 6 维频域统计特征：
    [spectral_centroid, spectral_bandwidth, spectral_rolloff_85,
     dominant_freq, spectral_entropy, high_freq_ratio]
    """
    x = x.astype(np.float64)
    n = len(x)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    mag = np.abs(np.fft.rfft(x))
    mag_sum = np.sum(mag) + 1e-12

    centroid = np.sum(freqs * mag) / mag_sum
    bandwidth = np.sqrt(np.sum(((freqs - centroid) ** 2) * mag) / mag_sum)

    # 85% 能量滚降频率
    cumsum = np.cumsum(mag)
    rolloff_idx = np.searchsorted(cumsum, 0.85 * cumsum[-1])
    rolloff = freqs[min(rolloff_idx, len(freqs) - 1)]

    dominant_freq = freqs[np.argmax(mag)]

    # 谱熵（归一化功率谱）
    psd = (mag ** 2) / (np.sum(mag ** 2) + 1e-12)
    spectral_entropy = -np.sum(psd * np.log(psd + 1e-12))

    # 高频（>5 Hz）能量占比（泥石流高频特征）
    high_mask = freqs > 5.0
    high_freq_ratio = np.sum(mag[high_mask]) / mag_sum

    return np.array([centroid, bandwidth, rolloff, dominant_freq,
                     spectral_entropy, high_freq_ratio], dtype=np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# STFT 谱图
# ──────────────────────────────────────────────────────────────────────────────
def compute_stft_spectrogram(
    x: np.ndarray,
    fs: float = 200.0,
    nperseg: int = 256,
    noverlap: int = 192,
    log_scale: bool = True,
) -> np.ndarray:
    """
    计算 STFT 幅度谱图。
    返回形状 (freq_bins, time_frames) 的 float32 数组。
    """
    from scipy.signal import stft as scipy_stft
    _, _, Zxx = scipy_stft(x, fs=fs, nperseg=nperseg, noverlap=noverlap,
                           window="hann", boundary=None)
    mag = np.abs(Zxx).astype(np.float32)
    if log_scale:
        mag = np.log1p(mag * 1e6)  # log 压缩
    return mag


# ──────────────────────────────────────────────────────────────────────────────
# 组合特征向量（传统 ML 用）
# ──────────────────────────────────────────────────────────────────────────────
def extract_all_features(x: np.ndarray, fs: float = 200.0) -> np.ndarray:
    """返回 14 维特征向量（时域8 + 频域6）。"""
    td = time_domain_features(x)
    fd = freq_domain_features(x, fs)
    return np.concatenate([td, fd])
