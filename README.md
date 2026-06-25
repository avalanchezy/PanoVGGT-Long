# PanoVGGT-Long

PanoVGGT-Long extends the long-sequence reconstruction recipe of
[VGGT-Long](https://github.com/DengKaiCQ/VGGT-Long) to panoramic videos by using
PanoVGGT as the local geometry model.

The project keeps the same high-level pipeline as VGGT-Long and Pi-Long:

1. split an ordered image sequence into overlapping chunks;
2. reconstruct each chunk independently with a feed-forward 3D model;
3. align adjacent chunks with overlap-based Sim(3) registration;
4. optionally add loop-closure constraints;
5. export aligned chunk point clouds and a merged point cloud.

The only intended model change is the local reconstructor: PanoVGGT-Long uses
PanoVGGT for panoramic/equirectangular frames. It does not require SLAM poses,
camera calibration, 2D maps, road masks, or pose-assisted inputs.

## Highlights

- Long-sequence panoramic reconstruction with a VGGT-Long-style chunk pipeline.
- Overlap-based Sim(3) chunk alignment.
- Optional visual place recognition and loop-closure refinement hooks inherited
  from VGGT-Long.
- PanoVGGT adapter with 2:1 panoramic input resizing.
- Compatible output layout with VGGT-Long-style downstream inspection scripts.

## Environment

PanoVGGT-Long is intended to run on Linux or WSL2 with an NVIDIA GPU. Native
Windows execution is not the recommended path because the long-sequence stack
uses Linux-oriented dependencies and scripts.

Create a conda environment:

```bash
conda create -n panovggt-long python=3.10 -y
conda activate panovggt-long
```

Install PyTorch. Choose the CUDA wheel that matches your driver/runtime. For
CUDA 12.4:

```bash
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
  --index-url https://download.pytorch.org/whl/cu124
```

Install the remaining Python dependencies:

```bash
pip install -r requirements.txt
```

The C++ loop solver is optional. The Python/Numba path works without compiling
it. If you want the C++ solver:

```bash
python setup.py install
```

## Weights

Download the PanoVGGT checkpoint and optional loop-retrieval weights:

```bash
bash scripts/download_weights.sh
```

By default this creates:

```text
weights/
  model.pt
  dino_salad.ckpt
  dinov2_vitb14_pretrain.pth
  ORBvoc.txt
```

Only `weights/model.pt` is required when `Model.loop_enable: False`. The DINO,
SALAD, and ORB vocabulary files are used for loop-closure retrieval.

If you already have the PanoVGGT checkpoint, you can link it instead of
downloading it again:

```bash
PANO_MODEL=/path/to/model.pt bash scripts/download_weights.sh
```

## Prepare Images

The input is an ordered directory of RGB frames. For panoramic video, extract
equirectangular frames at a moderate frame rate:

```bash
mkdir -p ./data/my_pano_sequence
ffmpeg -i input_360_video.mp4 -vf "fps=2,scale=1036:518" \
  ./data/my_pano_sequence/frame_%06d.png
```

Lower FPS reduces redundant frames and memory pressure. Higher FPS can help
when motion is fast, but it also increases chunk count and disk usage.

## Run

Basic run:

```bash
python panovggt_long.py \
  --image_dir ./data/my_pano_sequence \
  --config configs/base_config.yaml \
  --exp_folder_name ./exps
```

For 40-frame chunks with 20-frame overlap:

```bash
python panovggt_long.py \
  --image_dir ./data/my_pano_sequence \
  --config configs/panorama_c40_o20.yaml \
  --exp_folder_name ./exps
```

WSL example from Windows:

```cmd
scripts\run_example.bat /mnt/c/path/to/panoramic_frames
```

Linux/WSL shell example:

```bash
bash scripts/run_example.sh ./data/my_pano_sequence
```

## Configuration

The main options are in `configs/base_config.yaml`.

```yaml
Weights:
  PanoVGGT: './weights/model.pt'
  PanoVGGT_config: './configs/panovggt_model.yaml'

Model:
  chunk_size: 60
  overlap: 30
  loop_enable: False
  input_h: 518
  input_w: 1036
```

Useful settings:

- `chunk_size`: number of frames processed by PanoVGGT at once.
- `overlap`: number of shared frames used to align adjacent chunks.
- `loop_enable`: enables visual loop detection and global Sim(3) refinement.
- `Pointcloud_Save.sample_ratio`: point subsampling ratio for exported PLY files.
- `Pointcloud_Save.conf_threshold_coef`: confidence threshold coefficient.

For PanoVGGT, `input_w` should normally be twice `input_h` for equirectangular
frames.

## Outputs

Each run writes a timestamped folder:

```text
exps/<image_dir_name>/<timestamp>/
  _tmp_results_unaligned/
  _tmp_results_aligned/
  _tmp_results_loop/
  pcd/
    0_pcd.ply
    1_pcd.ply
    ...
    combined_pcd.ply
  camera_poses.txt
  camera_poses.ply
```

`pcd/combined_pcd.ply` is the merged reconstruction. Individual chunk PLY files
are kept for debugging chunk alignment.

## Loop Closure

The default configuration disables loop closure:

```yaml
Model:
  loop_enable: False
```

To enable VGGT-Long-style loop closure, set:

```yaml
Model:
  loop_enable: True
```

and make sure the retrieval weights exist:

```text
weights/dino_salad.ckpt
weights/dinov2_vitb14_pretrain.pth
weights/ORBvoc.txt
```

Loop closure follows the same role as in VGGT-Long: it proposes non-adjacent
chunk constraints and refines the global Sim(3) graph.

## Troubleshooting

`CUDA out of memory`

Reduce `chunk_size`, reduce image resolution, or close other GPU processes.
For long panoramic clips, `chunk_size=40` and `overlap=20` is a practical
starting point.

`CPU memory or system freeze`

Long-sequence reconstruction stores large intermediate NumPy files. Make sure
the output drive has enough free space and avoid placing `exps/` on a slow or
nearly full disk.

`faiss-gpu` cannot be installed

Loop closure depends on FAISS. If your CUDA stack does not have a matching
`faiss-gpu` wheel, either install a platform-specific FAISS package or keep
`loop_enable: False`.

`libGL.so.1` is missing

Install the OpenCV runtime dependency:

```bash
sudo apt-get update
sudo apt-get install -y libgl1
```

## Acknowledgements

This repository builds on PanoVGGT and the long-sequence infrastructure of
VGGT-Long / Pi-Long. Please check the upstream repositories and licenses before
redistributing models or using the code commercially.

## License

This repository includes code adapted from VGGT-Long and PanoVGGT. The source
code and model weights may be subject to their respective upstream licenses.
See `LICENSE_VGGT_LONG.txt` and the upstream PanoVGGT release for details.
