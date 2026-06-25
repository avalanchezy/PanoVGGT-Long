import argparse
import gc
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from loop_utils.config_utils import load_config
from loop_utils.sim3utils import (
    accumulate_sim3_transforms,
    apply_sim3_direct,
    merge_ply_files,
    save_confident_pointcloud_batch,
    weighted_align_point_maps,
)


def chunk_id(path: Path) -> int:
    return int(path.stem.split("_")[-1])


def save_chunk_pcd(chunk, ply_path, config):
    points = chunk["world_points"].reshape(-1, 3)
    images = chunk["images"]
    colors = (images.transpose(0, 2, 3, 1).reshape(-1, 3) * 255).astype(np.uint8)
    confs = chunk["world_points_conf"].reshape(-1)
    pc_cfg = config["Model"]["Pointcloud_Save"]
    conf_threshold = (
        np.mean(confs) * pc_cfg["conf_threshold_coef"]
        if pc_cfg.get("use_conf_filter", True)
        else -1.0
    )
    save_confident_pointcloud_batch(
        points=points,
        colors=colors,
        confs=confs,
        output_path=str(ply_path),
        conf_threshold=conf_threshold,
        sample_ratio=pc_cfg["sample_ratio"],
    )


def save_camera_poses(run_dir: Path, chunks, sim3_abs):
    all_poses = []
    for idx, chunk in enumerate(chunks):
        poses = chunk.get("extrinsic", chunk.get("camera_poses", None))
        if poses is None:
            return
        if idx > 0:
            s, R, t = sim3_abs[idx - 1]
            S = np.eye(4)
            S[:3, :3] = s * R
            S[:3, 3] = t
            transformed = []
            for c2w in poses:
                pose = S @ c2w
                pose[:3, :3] /= s
                transformed.append(pose)
            poses = np.asarray(transformed)
        all_poses.extend(list(poses))

    with open(run_dir / "camera_poses.txt", "w") as f:
        for pose in all_poses:
            f.write(" ".join(str(x) for x in pose.flatten()) + "\n")

    with open(run_dir / "camera_poses.ply", "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(all_poses)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        for pose in all_poses:
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
    aligned = run_dir / "_tmp_results_aligned"
    pcd_dir = run_dir / "pcd"
    aligned.mkdir(exist_ok=True)
    pcd_dir.mkdir(exist_ok=True)

    chunk_paths = sorted(unaligned.glob("chunk_*.npy"), key=chunk_id)
    if not chunk_paths:
        raise FileNotFoundError(f"No chunk_*.npy files under {unaligned}")

    sim3_seq = []
    for i in range(len(chunk_paths) - 1):
        c1 = np.load(chunk_paths[i], allow_pickle=True).item()
        c2 = np.load(chunk_paths[i + 1], allow_pickle=True).item()
        conf1 = c1["world_points_conf"][-overlap:]
        conf2 = c2["world_points_conf"][:overlap]
        conf_threshold = (
            min(np.median(conf1), np.median(conf2)) * 0.1
            if config["Model"]["Pointcloud_Save"].get("use_conf_filter", True)
            else -1.0
        )
        print(f"Aligning chunk {i} -> {i + 1}")
        s, R, t = weighted_align_point_maps(
            c1["world_points"][-overlap:],
            conf1,
            c2["world_points"][:overlap],
            conf2,
            None,
            conf_threshold=conf_threshold,
            config=config,
        )
        print("Estimated Scale:", s)
        print("Estimated Rotation:\n", R)
        print("Estimated Translation:", t)
        sim3_seq.append((s, R, t))
        del c1, c2
        gc.collect()

    sim3_abs = accumulate_sim3_transforms(sim3_seq)
    chunks_for_pose = []
    for i, path in enumerate(chunk_paths):
        chunk = np.load(path, allow_pickle=True).item()
        if i > 0:
            s, R, t = sim3_abs[i - 1]
            chunk["world_points"] = apply_sim3_direct(chunk["world_points"], s, R, t)
        np.save(aligned / f"chunk_{i}.npy", chunk)
        save_chunk_pcd(chunk, pcd_dir / f"{i}_pcd.ply", config)
        chunks_for_pose.append(chunk)

    save_camera_poses(run_dir, chunks_for_pose, sim3_abs)
    merge_ply_files(str(pcd_dir), str(pcd_dir / "combined_pcd.ply"))
    print(f"Done: {pcd_dir / 'combined_pcd.ply'}")


if __name__ == "__main__":
    main()
