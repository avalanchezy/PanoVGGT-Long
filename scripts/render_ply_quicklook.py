import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read_binary_ply(path):
    with open(path, "rb") as f:
        header = []
        while True:
            line = f.readline()
            if not line:
                raise ValueError("Invalid PLY: missing end_header")
            header.append(line.decode("ascii", errors="ignore").strip())
            if line.strip() == b"end_header":
                break
        n = 0
        for line in header:
            if line.startswith("element vertex"):
                n = int(line.split()[-1])
                break
        dtype = [
            ("x", np.float32), ("y", np.float32), ("z", np.float32),
            ("r", np.uint8), ("g", np.uint8), ("b", np.uint8),
        ]
        data = np.fromfile(f, dtype=dtype, count=n)
    xyz = np.column_stack([data["x"], data["y"], data["z"]])
    rgb = np.column_stack([data["r"], data["g"], data["b"]]).astype(np.float32) / 255.0
    return xyz, rgb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ply", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max_points", type=int, default=250000)
    args = parser.parse_args()

    xyz, rgb = read_binary_ply(args.ply)
    if len(xyz) > args.max_points:
        idx = np.linspace(0, len(xyz) - 1, args.max_points).astype(np.int64)
        xyz = xyz[idx]
        rgb = rgb[idx]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=180)
    axes[0].scatter(xyz[:, 0], xyz[:, 2], s=0.08, c=rgb)
    axes[0].set_title("Top-down: x-z")
    axes[0].set_aspect("equal", adjustable="box")
    axes[0].grid(True, alpha=0.2)

    axes[1].scatter(xyz[:, 0], xyz[:, 1], s=0.08, c=rgb)
    axes[1].set_title("Side view: x-y")
    axes[1].set_aspect("equal", adjustable="box")
    axes[1].grid(True, alpha=0.2)

    for ax in axes:
        ax.set_xlabel("x")
        ax.set_ylabel("z / y")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out)
    print(out)


if __name__ == "__main__":
    main()
