# EAI Course Project: Diffusion Policy for LeRobot SO-101

This repository contains the implementation of **Diffusion Policy** for the Embodied AI 2025 Course Project (Track 1). The goal is to control a simulated LeRobot SO-101 manipulator to perform Lift, Stack, and Sort tasks.

## 📂 Repository Structure

```
EAI-Project-LeRobot/
├── assets/               # Robot assets (URDF, meshes)
├── configs/              # Configuration files
│   ├── env/              # Environment configs
│   ├── policy/           # Policy configs (Diffusion)
│   ├── robots/           # Robot calibration/config files
│   └── train.yaml        # Main training configuration
├── data/                 # Datasets (Lift, Sort, Stack)
├── docs/                 # Documentation, reports, and images
│   ├── images/           # Reference images and results
│   └── midterm_report/   # Midterm report LaTeX source
├── logs/                 # Runtime logs and scene captures
│   ├── simulation/       # Simulation outputs (images, videos)
│   └── train/            # Training logs and checkpoints
├── scripts/              # Main executable scripts
│   ├── train.py          # Training script
│   ├── eval.py           # Evaluation script (Sim)
│   ├── visualize_training.py # Training visualization tool
│   ├── collect_data.py   # Data collection script
│   └── examples/         # Example scripts (e.g., dummy_eval.py)
├── src/                  # Source code package
│   └── lerobot/          # LeRobot library implementation
├── tools/                # Utility tools (calibration, video conversion)
├── requirements.txt      # Python dependencies
└── README.md             # Project documentation
```

## 🚀 Project Roadmap

### Phase 1: Simulation & Environment Setup
- [x] Set up SAPIEN/Gym environment with SO-101 robot.
- [x] Configure camera sensors (Front, Left Wrist, Right Wrist).
- [x] Implement task environments: Lift, Stack, Sort.

### Phase 2: Data Collection
- [x] Collect expert demonstrations for all tasks.
- [x] Preprocess data (normalization, chunking) for Diffusion Policy training.

### Phase 3: Policy Training
- [x] Implement DDPM-based Diffusion Policy.
- [x] Implement Training Loop with Hydra & TensorBoard.
- [ ] Train policies on collected datasets (In Progress).
- [ ] Tune hyperparameters (noise schedule, horizon, etc.).

### Phase 4: Evaluation & Sim-to-Real
- [x] Implement Evaluation Script (`eval.py`).
- [ ] Transfer trained policies to the real SO-101 robot.
- [ ] Address Sim-to-Real gaps.

## 🛠️ Usage


### 1. Installation

We recommend using **conda** to manage your Python environment:

```bash
# Create conda environment
conda create -n lerobot python=3.10
conda activate lerobot

# Install dependencies
python -m pip install -r requirements.txt
```


### 2. Data Preparation

Download the demonstration datasets from:

https://cloud.tsinghua.edu.cn/d/2687cde6d00b46b7a6db/

Extract all contents into the `data/` directory.

**Video Conversion:**
If you encounter video playback issues (e.g., in browsers or some video tools), or need to ensure all videos are in a standard format (H.264, yuv420p), run the following (requires `ffmpeg`):

```bash
bash tools/convert_videos.sh
```
This will re-encode all `.mp4` files in `data/` to a compatible format. Only run this if you experience compatibility problems or need to process videos for web visualization.


### 3. Training

To train the Diffusion Policy for a specific task (e.g., lift, sort, stack):

```bash
python scripts/train.py task=<task>
```

Replace `<task>` with `lift`, `sort`, or `stack` as needed. You can modify configurations in `configs/train.yaml` or override them via command line.


### 4. Visualization

To visualize training loss curves for a specific task:

```bash
python scripts/visualize_training.py --task <task>
```

To monitor with TensorBoard for a specific task:

```bash
tensorboard --logdir logs/train/<task>
```

Replace `<task>` with `lift`, `sort`, or `stack` as needed.



### 5. Evaluation

To evaluate a trained checkpoint in the simulation for a specific task and enable the web viewer:

```bash
python scripts/eval.py --checkpoint logs/train/<task>/<date>/<time>/checkpoint_XXX.pth --task <task> --web-viewer
```

Replace `<task>` with `lift`, `sort`, or `stack` as needed. When `--web-viewer` is enabled, open [http://localhost:5000](http://localhost:5000) in your browser to view live camera streams and record videos/screenshots during evaluation.

## 👥 Team
- Guanheng Chen
- Zuo Gou
- Zhengyang Fan
