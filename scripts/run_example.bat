@echo off
set IMAGE_DIR=%1
set CONFIG=%2
set OUTPUT_DIR=%3
set REPO_WIN=%~dp0..

if "%IMAGE_DIR%"=="" (
  echo Usage: scripts\run_example.bat /mnt/c/path/to/panoramic_frames [config] [output_dir]
  exit /b 1
)

if "%CONFIG%"=="" set CONFIG=configs/base_config.yaml
if "%OUTPUT_DIR%"=="" set OUTPUT_DIR=./exps

for /f "usebackq delims=" %%I in (`wsl.exe wslpath -a "%REPO_WIN%"`) do set REPO_WSL=%%I

wsl.exe -- bash -lc "source ~/miniforge3/etc/profile.d/conda.sh; conda activate ${CONDA_ENV:-panovggt-long}; cd '%REPO_WSL%'; python panovggt_long.py --image_dir '%IMAGE_DIR%' --config '%CONFIG%' --exp_folder_name '%OUTPUT_DIR%'"
