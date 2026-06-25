import argparse
import gc
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from loop_utils.config_utils import load_config
from loop_utils.sim3utils import accumulate_sim3_transforms, merge_ply_files, weighted_align_point_maps


def chunk_id(path: Path) -> int:
    return int(path.stem.split("_")[-1])


def write_header(f, n):
    f.write(b"ply\n")
    f.write(b"format binary_little_endian 1.0\n")
    f.write(f"element vertex {n}\n".encode("ascii"))
    f.write(b"property float x\n")
    f.write(b"property float y\n")
    f.write(b"property float z\n")
    f.write(b"property uchar red\n")
    f.write(b"property uchar green\n")
    f.write(b"property uchar blue\n")
    f.write(b"end_header\n")


def write_vertices(f, xyz, rgb):
    data = np.empty(
        len(xyz),
        dtype=[
            ("x", np.float32),
            ("y", np.float32),
            ("z", np.float32),
            ("r", np.uint8),
            ("g", np.uint8),
            ("b", np.uint8),
        ],
    )
    data["x"] = xyz[:, 0]
    data["y"] = xyz[:, 1]
    data["z"] = xyz[:, 2]
    data["r"] = rgb[:, 0]
    data["g"] = rgb[:, 1]
    data["b"] = rgb[:, 2]
    f.write(data.tobytes())


def selected_mask(points_hw3, conf_hw, threshold, stride):
    n = conf_hw.size
    idx_mask = (np.arange(n) % stride) == 0
    pts = points_hw3.reshape(-1, 3)
    conf = conf_hw.reshape(-1)
    return idx_mask & (conf >= threshold) & (conf > 1e-5) & np.isfinite(pts).all(axis=1)


def transform_points(xyz, sim3):
    if sim3 is None:
        return xyz.astype(np.float32, copy=False)
    s, R, t = sim3
    return (s * (R @ xyz.T)).T + t


def export_chunk_ply(chunk, out_path, config, sim3=None):
    pc_cfg = config["Model"]["Pointcloud_Save"]
    confs = chunk["world_points_conf"]
    threshold = (
        float(np.mean(confs) * pc_cfg["conf_threshold_coef"])
        if pc_cfg.get("use_conf_filter", True)
        else -1.0
    )
    sample_ratio = float(pc_cfg["sample_ratio"])
    stride = max(1, int(round(1.0 / sample_ratio)))

    counts = []
    for i in range(chunk["world_points"].shape[0]):
        mask = selected_mask(chunk["world_points"][i], confs[i], threshold, stride)
        counts.append(int(mask.sum()))
    total = int(sum(counts))

    with open(out_path, "wb") as f:
        write_header(f, total)
        for i in range(chunk["world_points"].shape[0]):
            mask = selected_mask(chunk["world_points"][i], confs[i], threshold, stride)
            xyz = chunk["world_points"][i].reshape(-1, 3)[mask]
            xyz = transform_points(xyz, sim3).astype(np.float32, copy=False)
            img = chunk["images"][i].transpose(1, 2, 0).reshape(-1, 3)
            rgb = np.clip(img[mask] * 255.0, 0, 255).astype(np.uint8)
            write_vertices(f, xyz, rgb)
    print(f"Saved {total} points -> {out_path}")


def save_camera_files(run_dir, chunks, sim3_abs):
    poses_all = []
    for i, chunk in enumerate(chunks):
        poses = chunk.get("extrinsic", chunk.get("camera_poses", None))
        if poses is None:
            return
        if i > 0:
            s, R, t = sim3_abs[i - 1]
            S = np.eye(4)
            S[:3, :3] = s * R
            S[:3, 3] = t
            transformed = []
            for pose in poses:
                p = S @ pose
                p[:3, :3] /= s
                transformed.append(p)
            poses = np.asarray(transformed)
        poses_all.extend(list(poses))

    with open(run_dir / "camera_poses.txt", "w") as f:
        for pose in poses_all:
            f.write(" ".join(str(x) for x in pose.flatten()) + "\n")

    with open(run_dir / "camera_poses.ply", "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(poses_all)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for pose in poses_all:
            x, y, z = pose[:3, 3]
            f.write(f"{x} {y} {z} 255 0 0\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    config = load_config(args.config)
    overlap = int(config["Model"]["overlap"])
    unaligned = run_dir / "_tmp_results_unaligned"
    pcd_dir = run_dir / "pcd_stream"
    pcd_dir.mkdir(exist_ok=True)

    chunk_paths = sorted(unaligned.glob("chunk_*.npy"), key=chunk_id)
    sim3_seq = []
    for i in range(len(chunk_paths) - 1):
        c1 = np.load(chunk_paths[i], allow_pickle=True).item()
        c2 = np.load(chunk_paths[i + 1], allow_pickle=True).item()
        conf1 = c1["world_points_conf"][-overlap:]
        conf2 = c2["world_points_conf"][:overlap]
        threshold = min(np.median(conf1), np.median(conf2)) * 0.1
        print(f"Aligning chunk {i} -> {i + 1}")
        sim3_seq.append(
            weighted_align_point_maps(
                c1["world_points"][-overlap:],
                conf1,
                c2["world_points"][:overlap],
                conf2,
                None,
                conf_threshold=threshold,
                config=config,
            )
        )
        del c1, c2
        gc.collect()

    sim3_abs = accumulate_sim3_transforms(sim3_seq)
    pose_chunks = []
    for i, path in enumerate(chunk_paths):
        chunk = np.load(path, allow_pickle=True).item()
        sim3 = None if i == 0 else sim3_abs[i - 1]
        export_chunk_ply(chunk, pcd_dir / f"{i}_pcd.ply", config, sim3=sim3)
        pose_chunks.append({k: chunk[k] for k in ("extrinsic", "intrinsic") if k in chunk})
        del chunk
        gc.collect()

    save_camera_files(run_dir, pose_chunks, sim3_abs)
    merge_ply_files(str(pcd_dir), str(pcd_dir / "combined_pcd.ply"))
    print(f"Done: {pcd_dir / 'combined_pcd.ply'}")


if __name__ == "__main__":
    main()
