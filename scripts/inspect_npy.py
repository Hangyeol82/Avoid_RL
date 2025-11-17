#!/usr/bin/env python3
import argparse
import os
import numpy as np


def summarize_array(arr: np.ndarray, name: str):
    print(f"\n=== {name} ===")
    print(f"type: {type(arr).__name__}")
    print(f"dtype: {arr.dtype}")
    print(f"shape: {arr.shape}")

    # Numeric summary
    if np.issubdtype(arr.dtype, np.number):
        try:
            vmin = np.nanmin(arr)
            vmax = np.nanmax(arr)
            mean = np.nanmean(arr)
            print(f"min/max: {vmin} / {vmax}")
            print(f"mean: {mean}")
        except Exception:
            pass

    # Unique counts (small cardinality)
    try:
        uniques, counts = np.unique(arr, return_counts=True)
        if len(uniques) <= 20:
            kv = ", ".join(f"{u}:{c}" for u, c in zip(uniques.tolist(), counts.tolist()))
            print(f"unique values (<=20 shown): {kv}")
        else:
            print(f"unique values: {len(uniques)} kinds")
    except Exception:
        pass

    # Waypoints guess: Nx2
    if arr.ndim == 2 and arr.shape[1] == 2 and np.issubdtype(arr.dtype, np.number):
        print("looks like waypoints (N,2)")
        print("first 10 rows:")
        n = min(10, arr.shape[0])
        for i in range(n):
            print(arr[i].tolist())
        col0_min, col0_max = np.min(arr[:, 0]), np.max(arr[:, 0])
        col1_min, col1_max = np.min(arr[:, 1]), np.max(arr[:, 1])
        print(f"col0 (x) range: {col0_min}..{col0_max}")
        print(f"col1 (y) range: {col1_min}..{col1_max}")


def load_npy(path: str):
    try:
        return np.load(path, allow_pickle=False)
    except Exception as e:
        print(f"[warn] allow_pickle=False failed for {path}: {e}")
        try:
            return np.load(path, allow_pickle=True)
        except Exception as e2:
            print(f"[error] Failed to load {path} even with allow_pickle=True: {e2}")
            return None


def main():
    p = argparse.ArgumentParser(description="Inspect .npy files (shape, dtype, summary)")
    p.add_argument("files", nargs="+", help="Paths to .npy files")
    args = p.parse_args()

    for f in args.files:
        if not os.path.exists(f):
            print(f"[missing] {f}")
            continue
        size = os.path.getsize(f)
        print(f"\n----- {f} ({size} bytes) -----")
        arr = load_npy(f)
        if arr is None:
            continue
        summarize_array(arr, os.path.basename(f))


if __name__ == "__main__":
    main()
