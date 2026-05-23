#!/usr/bin/env python3
"""Fatia o memmap normalizado em um arquivo .npy por timestep (6h).

Para uso em plots, inferência manual e testes — formato amigável para
carregar um único timestep sem mmap. Roda apenas em val/test.

Layout de saída:
    <out-dir>/<split>/<YYYY-MM-DDTHH>.npy   shape=(49152, 235) float16
    <out-dir>/<split>_index.json            espelha o do memmap

Reutiliza o memmap criado por 03_normalize.py — não recomputa transforms.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np


def slice_split(memmap_dir: Path, split: str, out_dir: Path) -> None:
    arr_path = memmap_dir / f"{split}.npy"
    times_path = memmap_dir / f"{split}_times.npy"
    index_path = memmap_dir / f"{split}_index.json"

    if not arr_path.exists():
        raise FileNotFoundError(f"memmap ausente: {arr_path}")
    if not times_path.exists():
        raise FileNotFoundError(f"times ausente: {times_path}")

    arr = np.load(arr_path, mmap_mode="r")
    times = np.load(times_path)
    n, pix, nvar = arr.shape
    assert len(times) == n, f"len(times)={len(times)} != n_steps={n}"

    split_out = out_dir / split
    split_out.mkdir(parents=True, exist_ok=True)

    print(f"[{split}] memmap shape={arr.shape} dtype={arr.dtype}")
    print(f"[{split}] writing {n} files to {split_out}/")

    for i in range(n):
        ts = np.datetime64(times[i], "h")
        name = str(ts).replace(":", "")
        out_path = split_out / f"{name}.npy"
        np.save(out_path, np.asarray(arr[i]))
        if (i + 1) % 200 == 0 or i == n - 1:
            print(f"  [{split}] {i+1}/{n} -> {out_path.name}")

    if index_path.exists():
        shutil.copy2(index_path, out_dir / f"{split}_index.json")

    sample_size = (split_out / out_path.name).stat().st_size / 1e6
    print(f"[{split}] DONE — {n} files, ~{sample_size:.1f} MB each, "
          f"total ~{sample_size * n / 1024:.2f} GB")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--memmap-dir", required=True, type=Path,
                   help="Onde estão {split}.npy / {split}_times.npy")
    p.add_argument("--out-dir", required=True, type=Path,
                   help="Pasta-raiz para os arquivos por timestep")
    p.add_argument("--splits", nargs="+", default=["val", "test"])
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for split in args.splits:
        slice_split(args.memmap_dir, split, args.out_dir)


if __name__ == "__main__":
    main()
