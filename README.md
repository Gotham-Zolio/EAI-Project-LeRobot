
# EAI Course Project: Diffusion Policy for LeRobot SO-101

This repository implements a Diffusion Policy for the LeRobot SO-101 manipulator, supporting Lift, Stack, and Sort tasks in simulation and training for the Embodied AI 2025 course project (Track 1).

## 📁 Repository Structure

```
EAI-Project-LeRobot/
├── assets/                # Robot models, URDF, SRDF, parts
│   └── SO101/             # SO-101 specific resources
├── configs/               # Configuration files
│   ├── env/               # Environment configs
│   ├── policy/            # Diffusion Policy configs
│   ├── robots/            # Robot calibration/configs
│   └── train.yaml         # Main training config
├── data/                  # Collected datasets
├── docs/                  # Documentation, reports, images
│   ├── images/            # Result images
│   └── midterm_report/    # Midterm report (LaTeX)
├── grasp-cube-sample/     # Related samples and external dependencies
├── logs/                  # Logs and simulation outputs
│   ├── debug/             # Debug logs
│   └── simulation/        # Simulation images/videos
├── scripts/               # Main executable scripts
│   ├── train.py               # Training entry
│   ├── eval.py                # Evaluation entry
│   ├── collect_data.py        # Data collection (motion planning)
│   ├── sim_env_demo.py        # Simulation environment test
│   ├── reward_functions.py    # Reward functions
│   ├── visualize_training.py  # Training visualization
│   └── examples/              # Example scripts
├── src/                     # Source code
│   └── lerobot/             # Main implementation
├── tools/                   # Utility scripts
│   ├── calibration/         # Camera calibration
│   ├── web_viewer/          # Web visualization
│   └── visualize_gripper_pose.py # Gripper pose visualization
├── requirements.txt         # Python dependencies
├── pyproject.toml           # Python project config
└── README.md                # Project documentation
```

## 🚀 Getting Started

### 1. Installation

It is recommended to use **conda** for Python environment management:

```bash
conda create -n lerobot python=3.10
conda activate lerobot
python -m pip install -r requirements.txt
```

### 2. Simulation Environment Test

Run the simulation environment and generate test images:

```bash
python scripts/sim_env_demo.py --task lift
```
Available tasks: `lift`, `sort`, `stack`. Output images are saved in `logs/simulation/<task>/`.

### 3. Data Collection (Motion Planning)

Collect demonstration data for a specific task using motion planning:

```bash
python scripts/collect_data.py --config.task lift --config.num-episodes 100 --config.web-viewer --config.vis
```
Key arguments:
- `--config.task`: Task type (`lift`, `sort`, `stack`)
- `--config.num-episodes`: Number of episodes to collect
- `--config.save-dir`: Output directory (default: `data/raw`)
- `--config.headless`: Run without GUI for faster collection
- `--config.web-viewer`: Enable web visualization
- `--config.vis`: Enable simulation visualization (show axes and debug info)

All data is collected online; there is no need to download datasets or perform video conversion.

### 4. Gripper Pose Visualization for Debugging

To visualize the target gripper pose for debugging motion planning:

```bash
python tools/visualize_gripper_pose.py --task lift
```
This helps verify the correctness of the planned gripper trajectory and target pose.

### 5. Training Diffusion Policy

```bash
python scripts/train.py task=lift
```
Available tasks: `lift`, `sort`, `stack`. You can modify configs in `configs/train.yaml` or override via command line. Training logs and models are saved in `logs/train/<task>/`.

### 6. Training Visualization

```bash
python scripts/visualize_training.py --task lift
```
Or use TensorBoard:
```bash
tensorboard --logdir logs/train/lift
```

### 7. Evaluation and Web Visualization

```bash
python scripts/eval.py --checkpoint logs/train/lift/<date>/<time>/checkpoint_XXX.pth --task lift --web-viewer
```
Open [http://localhost:5000](http://localhost:5000) in your browser to view real-time camera streams.

## 🧩 Main Dependencies

- Python 3.10
- SAPIEN 3.x
- torch 2.7.x
- gymnasium, hydra-core, tyro, opencv-python, diffusers, datasets, h5py, etc.

See `requirements.txt` and `pyproject.toml` for details.

## 👥 Team

- Guanheng Chen
- Zuo Gou
- Zhengyang Fan
