import argparse
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk_dir", required=True)
    args = parser.parse_args()
    chunk_dir = Path(args.chunk_dir)
    for path in sorted(chunk_dir.glob("chunk_*.npy")):
        chunk = np.load(path, allow_pickle=True).item()
        pts = chunk["world_points"]
        conf = chunk["world_points_conf"]
        print(
            path.name,
            "points", pts.shape, pts.dtype,
            "finite", float(np.isfinite(pts).mean()),
            "min", float(np.nanmin(pts)),
            "max", float(np.nanmax(pts)),
            "conf_mean", float(np.nanmean(conf)),
            "conf_min", float(np.nanmin(conf)),
            "conf_max", float(np.nanmax(conf)),
        )


if __name__ == "__main__":
    main()
