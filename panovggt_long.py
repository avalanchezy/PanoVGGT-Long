import argparse
import contextlib
import gc
import glob
import os
from datetime import datetime
from pathlib import Path
import shutil
import sys

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
import torch
from tqdm.auto import tqdm

try:
    import onnxruntime  # noqa: F401
except ImportError:
    print("onnxruntime not found. Sky segmentation may not work.")

from LoopModelDBoW.retrieval.retrieval_dbow import RetrievalDBOW
from LoopModels.LoopModel import LoopDetector
from loop_utils.config_utils import load_config
from loop_utils.sim3loop import Sim3LoopOptimizer
from loop_utils.sim3utils import (
    accumulate_sim3_transforms,
    apply_sim3_direct,
    compute_sim3_ab,
    merge_ply_files,
    process_loop_list,
    save_confident_pointcloud_batch,
    warmup_numba,
    weighted_align_point_maps,
)
from panovggt.models.panovggt_model import PanoVGGTModel


os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

_IMG_EXTS = (".jpg", ".JPG", ".jpeg", ".JPEG", ".png", ".PNG")
_INPUT_H = 518
_INPUT_W = 1036


def remove_duplicates(data_list):
    """
    Keep the first loop item for each chunk-pair.
    data_list: [(67, (3386, 3406), 48, (2435, 2455)), ...]
    """
    seen = {}
    result = []
    for item in data_list:
        if item[0] == item[2]:
            continue
        key = (item[0], item[2])
        if key not in seen:
            seen[key] = True
            result.append(item)
    return result


def copy_file(src_path, dst_dir):
    try:
        os.makedirs(dst_dir, exist_ok=True)
        dst_path = os.path.join(dst_dir, os.path.basename(src_path))
        shutil.copy2(src_path, dst_path)
        print(f"config yaml file has been copied to: {dst_path}")
        return dst_path
    except FileNotFoundError:
        print("File Not Found")
    except PermissionError:
        print("Permission Error")
    except Exception as e:
        print(f"Copy Error: {e}")
    return None


def resolve_repo_path(path_like, repo_root):
    if path_like is None:
        return None
    path = Path(path_like)
    if path.is_absolute():
        return str(path)
    return str((repo_root / path).resolve())


class PanoVGGTAdapter:
    """
    Adapter with the same public shape as VGGT-Long/Pi-Long local model adapters.

    infer_chunk(paths) returns tensors with a leading batch dimension:
      images            [1, S, 3, H, W]
      world_points      [1, S, H, W, 3]
      world_points_conf [1, S, H, W]
      camera_poses      [1, S, 4, 4] if predicted
      mask              None
    """

    def __init__(self, config, repo_root):
        self.config = config
        self.repo_root = Path(repo_root)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = (
            torch.bfloat16
            if self.device == "cuda" and torch.cuda.get_device_capability()[0] >= 8
            else torch.float16
        )
        self.model = None
        self.model_config_path = resolve_repo_path(config["Weights"]["PanoVGGT_config"], self.repo_root)
        self.checkpoint_path = resolve_repo_path(config["Weights"]["PanoVGGT"], self.repo_root)
        self.input_h = int(config["Model"].get("input_h", _INPUT_H))
        self.input_w = int(config["Model"].get("input_w", _INPUT_W))

    def load(self):
        cfg = OmegaConf.load(self.model_config_path)
        OmegaConf.resolve(cfg)
        mc = cfg.model
        model = PanoVGGTModel(
            img_size=cfg.img_size,
            patch_size=cfg.patch_size,
            embed_dim=cfg.embed_dim,
            enable_camera=mc.enable_camera,
            enable_depth=mc.enable_depth,
            enable_point=mc.enable_point,
            aggregator=OmegaConf.to_container(mc.aggregator, resolve=True),
        )
        ckpt = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        for key in ("model_state_dict", "model", "state_dict"):
            if key in ckpt:
                ckpt = ckpt[key]
                break
        sd = {(k[7:] if k.startswith("module.") else k): v for k, v in ckpt.items()}
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if missing:
            print(f"[PanoVGGT] missing keys  : {missing[:5]}{'...' if len(missing) > 5 else ''}")
        if unexpected:
            print(f"[PanoVGGT] unexpected keys: {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")
        self.model = model.eval().to(self.device)
        print(f"PanoVGGT model loaded on {self.device}.")

    def load_images_fixed(self, image_paths):
        frames = []
        for p in image_paths:
            bgr = cv2.imread(p)
            if bgr is None:
                raise IOError(f"Cannot read image: {p}")
            bgr = cv2.resize(bgr, (self.input_w, self.input_h), interpolation=cv2.INTER_AREA)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            frames.append(torch.from_numpy(rgb).permute(2, 0, 1))
        return torch.stack(frames, dim=0)

    def _select_points(self, preds):
        if preds.get("world_points", None) is not None:
            return preds["world_points"]
        if preds.get("points", None) is not None:
            return preds["points"]
        if preds.get("local_points", None) is not None:
            return preds["local_points"]
        raise KeyError("PanoVGGT output does not contain world_points/points/local_points")

    def _select_confidence(self, preds, points):
        for key in ("world_points_conf", "points_conf", "point_conf", "conf", "confidence", "depth_conf"):
            if key not in preds or preds[key] is None:
                continue
            conf = preds[key]
            if conf.ndim == 5 and conf.shape[-1] == 1:
                conf = conf[..., 0]
            if conf.ndim == 5 and conf.shape[2] == 1:
                conf = conf[:, :, 0]
            if conf.shape[-2:] == points.shape[-3:-1]:
                return conf.float()
        return torch.ones(points.shape[:-1], dtype=torch.float32, device=points.device)

    def infer_chunk(self, image_paths):
        if self.model is None:
            raise RuntimeError("Call load() before infer_chunk().")
        images = self.load_images_fixed(image_paths).to(self.device)
        amp_ctx = (
            torch.amp.autocast("cuda", dtype=torch.bfloat16)
            if self.device == "cuda"
            else contextlib.nullcontext()
        )
        with torch.no_grad(), amp_ctx:
            preds = self.model(images.unsqueeze(0))

        points = self._select_points(preds)
        if points.dtype == torch.bfloat16:
            points = points.float()
        conf = self._select_confidence(preds, points)
        extrinsic = preds.get("camera_poses", preds.get("extrinsic", None))
        if isinstance(extrinsic, torch.Tensor) and extrinsic.dtype == torch.bfloat16:
            extrinsic = extrinsic.float()
        intrinsic = preds.get("intrinsic", None)
        if isinstance(intrinsic, torch.Tensor) and intrinsic.dtype == torch.bfloat16:
            intrinsic = intrinsic.float()

        out = {
            "world_points": points,
            "world_points_conf": conf,
            "extrinsic": extrinsic,
            "intrinsic": intrinsic,
            "images": images.unsqueeze(0),
            "mask": None,
        }
        out["mask"] = None
        return out


class PanoVGGT_Long:
    def __init__(self, image_dir, save_dir, config, repo_root):
        self.config = config
        self.repo_root = Path(repo_root)
        self.chunk_size = self.config["Model"]["chunk_size"]
        self.overlap = self.config["Model"]["overlap"]
        self.seed = 42
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = (
            torch.bfloat16
            if self.device == "cuda" and torch.cuda.get_device_capability()[0] >= 8
            else torch.float16
        )
        self.useDBoW = self.config["Model"]["useDBoW"]

        self.img_dir = image_dir
        self.img_list = None
        self.output_dir = save_dir

        self.result_unaligned_dir = os.path.join(save_dir, "_tmp_results_unaligned")
        self.result_aligned_dir = os.path.join(save_dir, "_tmp_results_aligned")
        self.result_loop_dir = os.path.join(save_dir, "_tmp_results_loop")
        self.pcd_dir = os.path.join(save_dir, "pcd")
        os.makedirs(self.result_unaligned_dir, exist_ok=True)
        os.makedirs(self.result_aligned_dir, exist_ok=True)
        os.makedirs(self.result_loop_dir, exist_ok=True)
        os.makedirs(self.pcd_dir, exist_ok=True)

        self.all_camera_poses = []
        self.all_camera_intrinsics = []
        self.delete_temp_files = self.config["Model"]["delete_temp_files"]

        self.model = PanoVGGTAdapter(self.config, self.repo_root)
        self.chunk_indices = None
        self.loop_list = []
        self.loop_optimizer = Sim3LoopOptimizer(self.config)
        self.sim3_list = []
        self.loop_sim3_list = []
        self.loop_predict_list = []
        self.loop_enable = self.config["Model"]["loop_enable"]

        if self.loop_enable:
            if self.useDBoW:
                self.retrieval = RetrievalDBOW(config=self.config)
            else:
                loop_info_save_path = os.path.join(save_dir, "loop_closures.txt")
                self.loop_detector = LoopDetector(
                    image_dir=image_dir,
                    output=loop_info_save_path,
                    config=self.config,
                )
        print("init done.")

    def get_loop_pairs(self):
        if self.useDBoW:
            for frame_id, img_path in tqdm(enumerate(self.img_list)):
                image_ori = np.array(Image.open(img_path))
                if len(image_ori.shape) == 2:
                    image_ori = cv2.cvtColor(image_ori, cv2.COLOR_GRAY2RGB)
                frame = cv2.resize(image_ori, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
                self.retrieval(frame, frame_id)
                cands = self.retrieval.detect_loop(
                    thresh=self.config["Loop"]["DBoW"]["thresh"],
                    num_repeat=self.config["Loop"]["DBoW"]["num_repeat"],
                )
                if cands is not None:
                    self.retrieval.confirm_loop(cands[0], cands[1])
                    self.retrieval.found.clear()
                    self.loop_list.append(cands)
                self.retrieval.save_up_to(frame_id)
        else:
            self.loop_detector.run()
            self.loop_list = self.loop_detector.get_loop_list()

    def _to_numpy_for_save(self, predictions):
        out = {}
        for key, value in predictions.items():
            if isinstance(value, torch.Tensor):
                out[key] = value.cpu().numpy().squeeze(0)
            else:
                out[key] = value
        return out

    def process_single_chunk(self, range_1, chunk_idx=None, range_2=None, is_loop=False):
        start_idx, end_idx = range_1
        chunk_image_paths = self.img_list[start_idx:end_idx]
        if range_2 is not None:
            start_idx, end_idx = range_2
            chunk_image_paths += self.img_list[start_idx:end_idx]

        predictions = self.model.infer_chunk(chunk_image_paths)
        predictions = self._to_numpy_for_save(predictions)

        if is_loop:
            save_dir = self.result_loop_dir
            filename = f"loop_{range_1[0]}_{range_1[1]}_{range_2[0]}_{range_2[1]}.npy"
        else:
            if chunk_idx is None:
                raise ValueError("chunk_idx must be provided when is_loop is False")
            save_dir = self.result_unaligned_dir
            filename = f"chunk_{chunk_idx}.npy"

        save_path = os.path.join(save_dir, filename)

        if not is_loop and range_2 is None:
            chunk_range = self.chunk_indices[chunk_idx]
            extrinsics = predictions.get("extrinsic", predictions.get("camera_poses", None))
            intrinsics = predictions.get("intrinsic", None)
            self.all_camera_poses.append((chunk_range, extrinsics))
            self.all_camera_intrinsics.append((chunk_range, intrinsics))

        if "depth" in predictions and predictions["depth"] is not None:
            predictions["depth"] = np.squeeze(predictions["depth"])

        np.save(save_path, predictions)
        return predictions if is_loop or range_2 is not None else None

    def build_chunk_indices(self):
        if self.overlap >= self.chunk_size:
            raise ValueError(
                f"[SETTING ERROR] Overlap ({self.overlap}) must be less than chunk size ({self.chunk_size})"
            )
        if len(self.img_list) <= self.chunk_size:
            self.chunk_indices = [(0, len(self.img_list))]
        else:
            step = self.chunk_size - self.overlap
            num_chunks = (len(self.img_list) - self.overlap + step - 1) // step
            self.chunk_indices = []
            for i in range(num_chunks):
                start_idx = i * step
                end_idx = min(start_idx + self.chunk_size, len(self.img_list))
                self.chunk_indices.append((start_idx, end_idx))
        return len(self.chunk_indices)

    def _confidence_threshold(self, conf1, conf2):
        if self.config["Model"]["Pointcloud_Save"].get("use_conf_filter", True):
            return min(np.median(conf1), np.median(conf2)) * 0.1
        return -1.0

    def _save_chunk_pcd(self, chunk_data, ply_path):
        points = chunk_data["world_points"].reshape(-1, 3)
        images = chunk_data["images"]
        colors = (images.transpose(0, 2, 3, 1).reshape(-1, 3) * 255).astype(np.uint8)
        confs = chunk_data["world_points_conf"].reshape(-1)
        conf_threshold = (
            np.mean(confs) * self.config["Model"]["Pointcloud_Save"]["conf_threshold_coef"]
            if self.config["Model"]["Pointcloud_Save"].get("use_conf_filter", True)
            else -1.0
        )
        save_confident_pointcloud_batch(
            points=points,
            colors=colors,
            confs=confs,
            output_path=ply_path,
            conf_threshold=conf_threshold,
            sample_ratio=self.config["Model"]["Pointcloud_Save"]["sample_ratio"],
        )

    def process_long_sequence(self):
        num_chunks = self.build_chunk_indices()

        for chunk_idx in range(len(self.chunk_indices)):
            print(f"[Progress]: {chunk_idx}/{len(self.chunk_indices)-1}")
            self.process_single_chunk(self.chunk_indices[chunk_idx], chunk_idx=chunk_idx)
            torch.cuda.empty_cache()

        if self.loop_enable:
            print("Loop SIM(3) estimating...")
            loop_results = process_loop_list(
                self.chunk_indices,
                self.loop_list,
                half_window=int(self.config["Model"]["loop_chunk_size"] / 2),
            )
            loop_results = remove_duplicates(loop_results)
            print(loop_results)
            for item in loop_results:
                single_chunk_predictions = self.process_single_chunk(item[1], range_2=item[3], is_loop=True)
                self.loop_predict_list.append((item, single_chunk_predictions))
                print(item)

        print(
            f"Processing {len(self.img_list)} images in {num_chunks} chunks "
            f"of size {self.chunk_size} with {self.overlap} overlap"
        )

        del self.model
        torch.cuda.empty_cache()

        print("Aligning all the chunks...")
        for chunk_idx in range(len(self.chunk_indices) - 1):
            print(f"Aligning {chunk_idx} and {chunk_idx+1} (Total {len(self.chunk_indices)-1})")
            chunk_data1 = np.load(
                os.path.join(self.result_unaligned_dir, f"chunk_{chunk_idx}.npy"),
                allow_pickle=True,
            ).item()
            chunk_data2 = np.load(
                os.path.join(self.result_unaligned_dir, f"chunk_{chunk_idx+1}.npy"),
                allow_pickle=True,
            ).item()

            point_map1 = chunk_data1["world_points"][-self.overlap:]
            point_map2 = chunk_data2["world_points"][:self.overlap]
            conf1 = chunk_data1["world_points_conf"][-self.overlap:]
            conf2 = chunk_data2["world_points_conf"][:self.overlap]
            mask = None
            if chunk_data1.get("mask", None) is not None:
                mask1 = chunk_data1["mask"][-self.overlap:]
                mask2 = chunk_data2["mask"][:self.overlap]
                mask = mask1.squeeze() & mask2.squeeze()

            conf_threshold = self._confidence_threshold(conf1, conf2)
            s, R, t = weighted_align_point_maps(
                point_map1,
                conf1,
                point_map2,
                conf2,
                mask,
                conf_threshold=conf_threshold,
                config=self.config,
            )
            print("Estimated Scale:", s)
            print("Estimated Rotation:\n", R)
            print("Estimated Translation:", t)
            self.sim3_list.append((s, R, t))

        if self.loop_enable:
            self.process_loop_constraints()
            input_abs_poses = self.loop_optimizer.sequential_to_absolute_poses(self.sim3_list)
            self.sim3_list = self.loop_optimizer.optimize(self.sim3_list, self.loop_sim3_list)
            optimized_abs_poses = self.loop_optimizer.sequential_to_absolute_poses(self.sim3_list)
            self.save_loop_plot(input_abs_poses, optimized_abs_poses)

        print("Apply alignment")
        self.sim3_list = accumulate_sim3_transforms(self.sim3_list)
        for chunk_idx in range(len(self.chunk_indices) - 1):
            print(f"Applying {chunk_idx + 1} -> {chunk_idx} (Total {len(self.chunk_indices) - 1})")
            s, R, t = self.sim3_list[chunk_idx]

            chunk_data = np.load(
                os.path.join(self.result_unaligned_dir, f"chunk_{chunk_idx + 1}.npy"),
                allow_pickle=True,
            ).item()
            chunk_data["world_points"] = apply_sim3_direct(chunk_data["world_points"], s, R, t)
            aligned_path = os.path.join(self.result_aligned_dir, f"chunk_{chunk_idx + 1}.npy")
            np.save(aligned_path, chunk_data)

            if chunk_idx == 0:
                chunk_data_first = np.load(
                    os.path.join(self.result_unaligned_dir, "chunk_0.npy"),
                    allow_pickle=True,
                ).item()
                np.save(os.path.join(self.result_aligned_dir, "chunk_0.npy"), chunk_data_first)
                self._save_chunk_pcd(chunk_data_first, os.path.join(self.pcd_dir, "0_pcd.ply"))

            aligned_chunk_data = np.load(
                os.path.join(self.result_aligned_dir, f"chunk_{chunk_idx+1}.npy"),
                allow_pickle=True,
            ).item()
            self._save_chunk_pcd(aligned_chunk_data, os.path.join(self.pcd_dir, f"{chunk_idx + 1}_pcd.ply"))

        if len(self.chunk_indices) == 1:
            chunk_data_first = np.load(os.path.join(self.result_unaligned_dir, "chunk_0.npy"), allow_pickle=True).item()
            np.save(os.path.join(self.result_aligned_dir, "chunk_0.npy"), chunk_data_first)
            self._save_chunk_pcd(chunk_data_first, os.path.join(self.pcd_dir, "0_pcd.ply"))

        self.save_camera_poses()
        print("Done.")

    def process_loop_constraints(self):
        for item in self.loop_predict_list:
            chunk_idx_a = item[0][0]
            chunk_idx_b = item[0][2]
            chunk_a_range = item[0][1]
            chunk_b_range = item[0][3]

            point_map_loop = item[1]["world_points"][: chunk_a_range[1] - chunk_a_range[0]]
            conf_loop = item[1]["world_points_conf"][: chunk_a_range[1] - chunk_a_range[0]]
            chunk_a_rela_begin = chunk_a_range[0] - self.chunk_indices[chunk_idx_a][0]
            chunk_a_rela_end = chunk_a_rela_begin + chunk_a_range[1] - chunk_a_range[0]
            chunk_data_a = np.load(
                os.path.join(self.result_unaligned_dir, f"chunk_{chunk_idx_a}.npy"),
                allow_pickle=True,
            ).item()
            point_map_a = chunk_data_a["world_points"][chunk_a_rela_begin:chunk_a_rela_end]
            conf_a = chunk_data_a["world_points_conf"][chunk_a_rela_begin:chunk_a_rela_end]
            conf_threshold = self._confidence_threshold(conf_a, conf_loop)
            s_a, R_a, t_a = weighted_align_point_maps(
                point_map_a, conf_a, point_map_loop, conf_loop, None, conf_threshold, self.config
            )

            point_map_loop = item[1]["world_points"][-chunk_b_range[1] + chunk_b_range[0] :]
            conf_loop = item[1]["world_points_conf"][-chunk_b_range[1] + chunk_b_range[0] :]
            chunk_b_rela_begin = chunk_b_range[0] - self.chunk_indices[chunk_idx_b][0]
            chunk_b_rela_end = chunk_b_rela_begin + chunk_b_range[1] - chunk_b_range[0]
            chunk_data_b = np.load(
                os.path.join(self.result_unaligned_dir, f"chunk_{chunk_idx_b}.npy"),
                allow_pickle=True,
            ).item()
            point_map_b = chunk_data_b["world_points"][chunk_b_rela_begin:chunk_b_rela_end]
            conf_b = chunk_data_b["world_points_conf"][chunk_b_rela_begin:chunk_b_rela_end]
            conf_threshold = self._confidence_threshold(conf_b, conf_loop)
            s_b, R_b, t_b = weighted_align_point_maps(
                point_map_b, conf_b, point_map_loop, conf_loop, None, conf_threshold, self.config
            )
            s_ab, R_ab, t_ab = compute_sim3_ab((s_a, R_a, t_a), (s_b, R_b, t_b))
            self.loop_sim3_list.append((chunk_idx_a, chunk_idx_b, (s_ab, R_ab, t_ab)))

    def save_loop_plot(self, input_abs_poses, optimized_abs_poses):
        def extract_xyz(pose_tensor):
            poses = pose_tensor.cpu().numpy()
            return poses[:, 0], poses[:, 1], poses[:, 2]

        x0, _, y0 = extract_xyz(input_abs_poses)
        x1, _, y1 = extract_xyz(optimized_abs_poses)
        plt.figure(figsize=(8, 6))
        plt.plot(x0, y0, "o--", alpha=0.45, label="Before Optimization")
        plt.plot(x1, y1, "o-", label="After Optimization")
        for i, j, _ in self.loop_sim3_list:
            plt.plot([x0[i], x0[j]], [y0[i], y0[j]], "r--", alpha=0.25)
            plt.plot([x1[i], x1[j]], [y1[i], y1[j]], "g-", alpha=0.35)
        plt.gca().set_aspect("equal")
        plt.title("Sim3 Loop Closure Optimization")
        plt.xlabel("x")
        plt.ylabel("z")
        plt.legend()
        plt.grid(True)
        plt.axis("equal")
        save_path = os.path.join(self.output_dir, "sim3_opt_result.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

    def run(self):
        print(f"Loading images from {self.img_dir}...")
        self.img_list = []
        for pattern in _IMG_EXTS:
            self.img_list += glob.glob(os.path.join(self.img_dir, f"*{pattern}"))
        self.img_list = sorted(self.img_list)
        if len(self.img_list) == 0:
            raise ValueError(f"[DIR EMPTY] No images found in {self.img_dir}!")
        print(f"Found {len(self.img_list)} images")

        if self.loop_enable:
            self.get_loop_pairs()
            if self.useDBoW:
                self.retrieval.close()
                gc.collect()
            else:
                del self.loop_detector
        torch.cuda.empty_cache()
        print("Loading model...")
        self.model.load()
        self.process_long_sequence()

    def save_camera_poses(self):
        if not self.all_camera_poses or self.all_camera_poses[0][1] is None:
            print("No camera poses predicted; skipping camera pose export.")
            return

        chunk_colors = [
            [255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0],
            [255, 0, 255], [0, 255, 255], [128, 0, 0], [0, 128, 0],
            [0, 0, 128], [128, 128, 0],
        ]
        print("Saving all camera poses to txt file...")
        all_poses = [None] * len(self.img_list)
        all_intrinsics = [None] * len(self.img_list)

        first_chunk_range, first_chunk_extrinsics = self.all_camera_poses[0]
        _, first_chunk_intrinsics = self.all_camera_intrinsics[0]
        for i, idx in enumerate(range(first_chunk_range[0], first_chunk_range[1])):
            all_poses[idx] = first_chunk_extrinsics[i]
            if first_chunk_intrinsics is not None:
                all_intrinsics[idx] = first_chunk_intrinsics[i]

        for chunk_idx in range(1, len(self.all_camera_poses)):
            chunk_range, chunk_extrinsics = self.all_camera_poses[chunk_idx]
            _, chunk_intrinsics = self.all_camera_intrinsics[chunk_idx]
            s, R, t = self.sim3_list[chunk_idx - 1]
            S = np.eye(4)
            S[:3, :3] = s * R
            S[:3, 3] = t
            for i, idx in enumerate(range(chunk_range[0], chunk_range[1])):
                c2w = chunk_extrinsics[i]
                transformed_c2w = S @ c2w
                transformed_c2w[:3, :3] /= s
                all_poses[idx] = transformed_c2w
                if chunk_intrinsics is not None:
                    all_intrinsics[idx] = chunk_intrinsics[i]

        poses_path = os.path.join(self.output_dir, "camera_poses.txt")
        with open(poses_path, "w") as f:
            for pose in all_poses:
                flat_pose = pose.flatten()
                f.write(" ".join([str(x) for x in flat_pose]) + "\n")
        print(f"Camera poses saved to {poses_path}")

        if all_intrinsics[0] is not None:
            intrinsics_path = os.path.join(self.output_dir, "intrinsic.txt")
            with open(intrinsics_path, "w") as f:
                for intrinsic in all_intrinsics:
                    fx = intrinsic[0, 0]
                    fy = intrinsic[1, 1]
                    cx = intrinsic[0, 2]
                    cy = intrinsic[1, 2]
                    f.write(f"{fx} {fy} {cx} {cy}\n")
            print(f"Camera intrinsics saved to {intrinsics_path}")

        ply_path = os.path.join(self.output_dir, "camera_poses.ply")
        with open(ply_path, "w") as f:
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
            color = chunk_colors[0]
            for pose in all_poses:
                position = pose[:3, 3]
                f.write(f"{position[0]} {position[1]} {position[2]} {color[0]} {color[1]} {color[2]}\n")
        print(f"Camera poses visualization saved to {ply_path}")

    def close(self):
        if not self.delete_temp_files:
            return
        total_space = 0
        for temp_dir in (self.result_unaligned_dir, self.result_aligned_dir, self.result_loop_dir):
            print(f"Deleting the temp files under {temp_dir}")
            for filename in os.listdir(temp_dir):
                file_path = os.path.join(temp_dir, filename)
                if os.path.isfile(file_path):
                    total_space += os.path.getsize(file_path)
                    os.remove(file_path)
        print("Deleting temp files done.")
        print(f"Saved disk space: {total_space/1024/1024/1024:.4f} GiB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PanoVGGT-Long")
    parser.add_argument("--image_dir", type=str, required=True, help="Image path")
    parser.add_argument(
        "--config",
        type=str,
        required=False,
        default="./configs/base_config.yaml",
        help="config path",
    )
    parser.add_argument(
        "--exp_folder_name",
        type=str,
        default="./exps",
        help="experiment root path",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent
    config = load_config(args.config)
    current_datetime = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    save_dir = os.path.join(args.exp_folder_name, args.image_dir.replace("/", "_").replace("\\", "_"), current_datetime)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"The exp will be saved under dir: {save_dir}")
        copy_file(args.config, save_dir)

    if config["Model"]["align_method"] == "numba":
        warmup_numba()

    panovggt_long = PanoVGGT_Long(args.image_dir, save_dir, config, repo_root)
    panovggt_long.run()
    panovggt_long.close()

    del panovggt_long
    torch.cuda.empty_cache()
    gc.collect()

    all_ply_path = os.path.join(save_dir, "pcd/combined_pcd.ply")
    input_dir = os.path.join(save_dir, "pcd")
    print("Saving all the point clouds")
    merge_ply_files(input_dir, all_ply_path)
    print("PanoVGGT-Long done.")
    sys.exit()
