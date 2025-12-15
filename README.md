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

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# Install dependencies
python -m pip install -r requirements.txt
```

### 2. Training

To train the Diffusion Policy on the Lift task:

```bash
python scripts/train.py batch_size=8
```

You can modify configurations in `configs/train.yaml` or override them via command line.

### 3. Visualization

To visualize training loss curves:

```bash
python scripts/visualize_training.py
```

To monitor with TensorBoard:

```bash
tensorboard --logdir logs/train
```

### 4. Evaluation

To evaluate a trained checkpoint in the simulation:

```bash
python scripts/eval.py --checkpoint logs/train/2025-12-14/13-13-50/checkpoints/last.pth
```

## 👥 Team
- Guanheng Chen
- Zuo Gou
- Zhengyang Fan
