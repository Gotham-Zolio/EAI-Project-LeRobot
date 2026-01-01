# EAI Course Project: Diffusion Policy for LeRobot SO-101

This repository implements a Diffusion Policy for the SO-101 dual-arm manipulator in SAPIEN simulation, supporting Lift, Stack, and Sort tasks. Developed for Embodied AI 2025 (Track 1).

## 📁 Repository Structure

```
EAI-Project-LeRobot/
├── assets/SO101/              # Robot models (URDF, SRDF, meshes)
├── configs/                   # Hydra configs
│   ├── train.yaml            # Training configuration
│   ├── env/default.yaml
│   ├── policy/diffusion.yaml
│   └── robots/
├── data/datasets/             # Collected demonstrations
│   ├── {task}/
│   │   ├── raw/              # HDF5: {task}_{version}_{date}.h5
│   │   └── meta/             # Metadata: {task}_{version}_info.json
├── logs/                      # Training and evaluation outputs
│   ├── train/{task}/{date}/{time}/
│   │   ├── checkpoints/      # Model weights
│   │   │   ├── checkpoint_epoch_10.pth
│   │   │   ├── checkpoint_epoch_100.pth
│   │   │   └── last.pth      # Latest checkpoint
│   │   ├── logs/             # TensorBoard logs
│   │   ├── eval_results/     # Evaluation metrics
│   │   └── stats.json        # Normalization + dataset metadata
│   └── simulation/           # Debug outputs
├── scripts/                   # Main entry points
│   ├── collect_data.py       # FSM+IK data collection
│   ├── train.py              # Policy training
│   ├── eval.py               # Policy evaluation
│   └── sim_env_demo.py       # Environment sanity check
├── src/lerobot/              # Core implementation
│   ├── envs/
│   ├── policy/
│   └── real/
└── tools/                     # Utilities
    ├── calibration/
    └── web_viewer/
```

## 🚀 Quick Start

### Installation

```bash
conda create -n lerobot python=3.10
conda activate lerobot
pip install -r requirements.txt
```

### Sanity Check

```bash
python scripts/sim_env_demo.py --task lift
```

## 📊 Data Collection

### Collect Demonstrations

Uses FSM + IK to generate expert trajectories. Automatically skips failed IK episodes.

```bash
# First batch (creates lift_v1.0_YYYYMMDD.h5)
python scripts/collect_data.py task=lift num_episodes=50 version=v1.0

# Append to same version
python scripts/collect_data.py task=lift num_episodes=50 version=v1.0

# New version (after strategy change)
python scripts/collect_data.py task=lift num_episodes=50 version=v1.1
```

**Parameters:**
- `task`: `lift`, `stack`, or `sort`
- `num_episodes`: Episodes to collect
- `version`: Dataset version (e.g., `v1.0`, `v1.1`)
- `headless`: Disable GUI (default: `True`)
- `web_viewer`: Enable web visualization (default: `False`)

**Output Structure:**

```
data/datasets/{task}/
├── raw/
│   └── {task}_{version}_{date}.h5          # HDF5 episodes
└── meta/
    └── {task}_{version}_info.json          # Metadata (counts, success rate, etc.)
```

**Best Practices:**
- Collect 50-100+ episodes per version
- Use same version when appending
- Increment version after strategy changes (v1.0 → v1.1 → v2.0)

## 🏋️ Training

### Train Policy

```bash
# Auto-detect latest dataset
python scripts/train.py task=lift batch_size=8 epochs=100

# Train with specific dataset version
python scripts/train.py task=lift dataset_version=v1.0 epochs=100

# Train with explicit path
python scripts/train.py task=lift dataset_path=data/datasets/lift/raw/lift_v1.0_20260101.h5
```

**Parameters:**
- `task`: Task name (required)
- `batch_size`: Batch size (default: 8)
- `epochs`: Training epochs (default: 100)
- `lr`: Learning rate (default: 1e-4)
- `save_freq`: Checkpoint frequency (default: 10)
- `dataset_version`: Specify dataset version (e.g., `v1.0`)
- `dataset_path`: Explicit dataset path

**Output Structure:**

```
logs/train/{task}/{date}/{time}/
├── checkpoints/
│   ├── checkpoint_epoch_10.pth
│   ├── checkpoint_epoch_100.pth
│   └── last.pth                      # Latest checkpoint
├── logs/                             # TensorBoard logs
├── .hydra/                           # Config backup
└── stats.json                        # Normalization + dataset metadata
```

### Monitor Training

```bash
tensorboard --logdir logs/train/lift --port 6006
```

Access at `http://localhost:6006`

**Logged Metrics:**
- Loss/batch, Loss/epoch, GradientNorm/batch, LearningRate, Hyperparameters

## 🎯 Evaluation

### Evaluate Policy

```bash
# Evaluate latest checkpoint
python scripts/eval.py \
  --checkpoint logs/train/lift/{date}/{time}/checkpoints/last.pth \
  --task lift \
  --num-episodes 20

# Evaluate specific epoch
python scripts/eval.py \
  --checkpoint logs/train/lift/{date}/{time}/checkpoints/checkpoint_epoch_100.pth \
  --task lift \
  --num-episodes 20
```

**Parameters:**
- `--checkpoint`: Path to `.pth` file (required)
- `--task`: Task name (required)
- `--num-episodes`: Episodes to evaluate (default: 10)
- `--device`: `cuda` or `cpu` (default: `cuda`)
- `--headless`: Disable SAPIEN GUI
- `--no-web-viewer`: Disable web visualization
- `--port`: Web viewer port (default: 5000)

### Web Viewer

Automatically starts at `http://localhost:5000` showing camera streams.

### Evaluation Output

Results are saved to `{checkpoint_dir}/eval_results/eval_{task}_{timestamp}.json`

## 🔧 Advanced Usage

### Batch Dataset Exploration

```python
import h5py
with h5py.File("data/datasets/lift/raw/lift_v1.0_20260101.h5", "r") as f:
    print(f"Episodes: {f.attrs['num_episodes']}")
    total_steps = sum(f[ep]['action'].shape[0] for ep in f.keys() if ep.startswith('episode_'))
    print(f"Total steps: {total_steps}")
```

### Comparing Multiple Runs

```bash
python scripts/train.py task=lift dataset_version=v1.0 epochs=50
python scripts/train.py task=lift dataset_version=v1.1 epochs=50
tensorboard --logdir logs/train/lift
```

### GPU Selection

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train.py task=lift epochs=100
CUDA_VISIBLE_DEVICES=2 python scripts/eval.py --checkpoint ... --task lift
```

## 🧩 System Details

### Simulator
- **Framework:** SAPIEN 3.x with PhysX
- **Robot:** SO-101 (6 DOF + 2-finger gripper per arm)
- **Physics:** Static friction=2.0, dynamic friction=2.0, restitution=0.0

### Policy
- **Architecture:** DDPM-based Diffusion Policy (100 diffusion steps)
- **Vision:** ResNet18 encoder
- **Observations:** Joint positions (qpos) + RGB images
- **Actions:** Target joint positions

### Tasks
- **Lift:** Grasp cube and raise above threshold
- **Stack:** Stack one cube on another
- **Sort:** Sort cubes by color using both arms

## 📦 Dependencies

Python 3.10, PyTorch 2.7+, SAPIEN 3.x, diffusers, hydra-core, h5py, gymnasium

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| GPU OOM | Reduce `batch_size` or use `CUDA_VISIBLE_DEVICES` |
| IK failures | Check gripper targets; visualize with `tools/visualize_gripper_pose.py` |
| stats.json missing | Ensure training completed successfully |
| Web viewer won't load | Check port 5000 availability; use `--port` to change |
| SAPIEN/Vulkan errors | Install NVIDIA drivers (warnings on headless systems are normal) |

## 👥 Team

Guanheng Chen, Zuo Gou, Zhengyang Fan

## 📄 License

Developed for educational purposes as part of Embodied AI 2025.