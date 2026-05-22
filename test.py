from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from scipy.io import loadmat


BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / "pre_chuli"
OUTPUT_DIR = BASE_DIR / "output"
SUMMARY_PATH = OUTPUT_DIR / "mat_summary.csv"


def to_scalar(value: object) -> object:
	"""Convert MATLAB-loaded values to a plain Python scalar when possible."""
	if isinstance(value, np.ndarray):
		if value.size == 1:
			return value.reshape(-1)[0].item()
		return value.tolist()
	if hasattr(value, "item"):
		try:
			return value.item()
		except Exception:
			return value
	return value


def extract_signal_and_fs(mat_data: dict[str, object]) -> tuple[str | None, np.ndarray | None, object | None]:
	signal_key = None
	signal_value = None
	fs_value = None

	for key, value in mat_data.items():
		if key.startswith("__"):
			continue
		if key.lower() == "fs":
			fs_value = to_scalar(value)
			continue
		if isinstance(value, np.ndarray):
			signal_key = key
			signal_value = value

	return signal_key, signal_value, fs_value


def summarize_mat_file(path: Path) -> dict[str, object]:
	mat_data = loadmat(path)
	signal_key, signal_value, fs_value = extract_signal_and_fs(mat_data)

	if signal_value is None:
		return {
			"file_name": path.name,
			"signal_key": "",
			"fs": fs_value if fs_value is not None else "",
			"length": 0,
			"dtype": "",
			"min": "",
			"max": "",
			"mean": "",
			"std": "",
		}

	signal_array = np.asarray(signal_value).reshape(-1)

	return {
		"file_name": path.name,
		"signal_key": signal_key or "",
		"fs": fs_value if fs_value is not None else "",
		"length": int(signal_array.size),
		"dtype": str(signal_array.dtype),
		"min": float(np.min(signal_array)),
		"max": float(np.max(signal_array)),
		"mean": float(np.mean(signal_array)),
		"std": float(np.std(signal_array)),
	}


def main() -> None:
	if not INPUT_DIR.exists():
		raise FileNotFoundError(f"Input directory not found: {INPUT_DIR}")

	OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

	mat_files = sorted(INPUT_DIR.glob("*.mat"))
	if not mat_files:
		raise FileNotFoundError(f"No .mat files found in: {INPUT_DIR}")

	summaries = [summarize_mat_file(path) for path in mat_files]

	with SUMMARY_PATH.open("w", newline="", encoding="utf-8-sig") as csv_file:
		fieldnames = ["file_name", "signal_key", "fs", "length", "dtype", "min", "max", "mean", "std"]
		writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
		writer.writeheader()
		writer.writerows(summaries)

	print(f"Processed {len(summaries)} files.")
	print(f"Summary saved to: {SUMMARY_PATH}")


if __name__ == "__main__":
	main()
