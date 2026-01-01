EAI Course Project: Diffusion Policy for LeRobot SO-101
=======================================================

This repository trains and evaluates a Diffusion Policy for the SO-101 manipulator in SAPIEN simulation across Lift, Stack, and Sort tasks (Embodied AI 2025 Track 1).

Repository Layout
-----------------
- assets/SO101/ – robot models (URDF, SRDF, meshes)
- configs/ – Hydra configs (train.yaml, env/, policy/, robots/)
- data/datasets/{task}/{raw,meta}/ – collected demos (HDF5) and metadata
- logs/train/{task}/ – training checkpoints and stats.json
- scripts/ – main entry points (collect_data.py, train.py, eval.py, sim_env_demo.py)
- src/lerobot/ – environments and DiffusionPolicy implementation
- tools/ – calibration, web viewer, debugging utilities

Setup
-----
```bash
conda create -n lerobot python=3.10
conda activate lerobot
pip install -r requirements.txt
```

Quick Workflow
--------------
1) Sanity check simulation
```bash
python scripts/sim_env_demo.py --task lift
```

2) Collect demonstrations (FSM + IK, skips failed IK episodes)
```bash
# First batch (creates data/datasets/lift/raw/lift_v1.0_YYYYMMDD.h5)
python scripts/collect_data.py task=lift num_episodes=50 version=v1.0

# Append to same version (continues the same H5)
python scripts/collect_data.py task=lift num_episodes=50 version=v1.0

# New version after strategy changes
python scripts/collect_data.py task=lift num_episodes=50 version=v1.1
```
Key flags: task {lift|stack|sort}, num_episodes, version (default v1.0), save_dir (default data/datasets), headless (default True), verbose.

3) Dataset format (HDF5)
- File attrs: task, version, num_episodes, last_updated, cameras, collection_method="fsm_ik"
- Episodes: episode_k/qpos (T,6), action (T,6), reward (T,), done (T,), images/front and right_wrist (T,H,W,3)
- Metadata JSON: data/datasets/{task}/meta/{task}_{version}_info.json tracks counts and cameras.

4) Train diffusion policy
```bash
# Auto-detect latest dataset for the task
python scripts/train.py task=lift batch_size=8 epochs=100

# Or specify a file explicitly
python scripts/train.py task=lift dataset_path=data/datasets/lift/raw/lift_v1.0_20260101.h5 batch_size=8
```
Outputs: logs/train/{task}/{date}/{time}/ with checkpoints, stats.json (normalization), and TensorBoard logs.

5) Evaluate
```bash
python scripts/eval.py \
  --checkpoint logs/train/lift/2026-01-01/12-32-31/checkpoint_epoch_100.pth \
  --task lift \
  --num-episodes 10
```
Flags: --device {cuda|cpu}, --headless, --no-web-viewer, --port <web_port>. Requires stats.json next to the checkpoint. Results are printed and saved to {checkpoint_dir}/eval_results/, and the web viewer streams to http://localhost:5000 by default.

Best Practices
--------------
- Collect 50–100+ episodes per version; keep the same version when appending; bump version when the policy/collection logic changes (v1.0 → v1.1 → v2.0).
- Monitor IK success; failed IK episodes are already skipped, but frequent failures usually indicate target or limit issues.
- Training OOM: lower batch_size or set CUDA_VISIBLE_DEVICES.
- Web viewer port conflicts: use --port to change.
- Missing stats.json during eval: rerun training or copy the file alongside the checkpoint.

Environment and Physics
-----------------------
- Simulator: SAPIEN 3.x with PhysX
- Robot: SO-101 dual-arm manipulator (6 DOF per arm)
- Tasks: lift, stack, sort
- Materials aligned with grasp-cube-sample: static_friction=2.0, dynamic_friction=2.0, restitution=0.0

Key Dependencies
----------------
- Python 3.10, PyTorch 2.7.x, diffusers, hydra-core, gymnasium, h5py, SAPIEN 3.x
See requirements.txt for the full list.

Project Summary
---------------
1) sim_env_demo.py (sanity)
2) collect_data.py (demos → HDF5 + metadata)
3) train.py (auto stats saving)
4) eval.py (success rate + web viewer)

Troubleshooting
---------------
- Vulkan/ICD warnings on headless servers are common; ensure NVIDIA drivers are installed if rendering fails.
- To inspect data counts:
```python
import h5py
f = h5py.File("data/datasets/lift/raw/lift_v1.0_20260101.h5")
print(f"Episodes: {f.attrs['num_episodes']}")
print(sum(f[k]['action'].shape[0] for k in f.keys() if k.startswith('episode_')))
```

Contributors
------------
- Guanheng Chen, Zuo Gou, Zhengyang Fan
# EAI Course Project: Diffusion Policy for LeRobot SO-101

This repository implements a Diffusion Policy for the LeRobot SO-101 manipulator, supporting Lift, Stack, and Sort tasks in simulation. Developed for the Embodied AI 2025 course project (Track 1).

## 📁 Repository Structure

```
EAI-Project-LeRobot/
├── assets/SO101/          # Robot models (URDF, SRDF, meshes)
├── configs/               # Hydra configs (train.yaml, env/, policy/, robots/)
├── data/datasets/         # Collected demonstration data
│   ├── {task}/
│   │   ├── raw/           # HDF5 files: {task}_{version}_{date}.h5
│   │   └── meta/          # Metadata JSON files
├── logs/                  # Training and evaluation outputs
│   ├── train/{task}/      # Checkpoints and stats.json
│   └── simulation/        # Debug visualizations
├── scripts/               # Main executables
│   ├── collect_data.py    # Data collection with FSM+IK
│   ├── train.py           # Train Diffusion Policy
│   ├── eval.py            # Evaluate policy
│   └── sim_env_demo.py    # Test simulation
├── src/lerobot/           # Core implementation
│   ├── envs/              # SAPIEN environment wrappers
│   ├── policy/            # DiffusionPolicy (DDPM)
│   └── real/              # (Reserved for real robot)
└── tools/                 # Utilities (calibration, web_viewer)
```

## 🚀 Quick Start

### 1. Installation

Create a conda environment and install dependencies:

```bash
conda create -n lerobot python=3.10
conda activate lerobot
pip install -r requirements.txt
```

### 2. Test Simulation Environment

Verify that SAPIEN and the environment are working:

```bash
python scripts/sim_env_demo.py --task lift
```

Available tasks: `lift`, `sort`, `stack`. Output saved to `logs/simulation/<task>/`.

## 📊 Data Collection Workflow

### Collect Expert Demonstrations

The script uses FSM (Finite State Machine) + IK (Inverse Kinematics) to generate expert trajectories:

```bash
python scripts/collect_data.py task=lift num_episodes=50 version=v1.0
```

**Parameters:**
- `task`: Task name (`lift`, `sort`, `stack`)
- `num_episodes`: Number of episodes to collect
- `version`: Version identifier for dataset management (e.g., `v1.0`, `v1.1`)
- `headless`: Run without GUI (default: `True`)
- `web_viewer`: Enable web visualization (default: `False`)

**Output:**
- HDF5 file: `data/datasets/{task}/raw/{task}_{version}_{date}.h5`
- Metadata: `data/datasets/{task}/meta/{task}_{version}_info.json`

### Continue Collecting Same Version

Running the same command appends to existing episodes:

```bash
python scripts/collect_data.py task=lift num_episodes=50 version=v1.0
```

### Start New Version

Increment version when improving collection strategy:

```bash
python scripts/collect_data.py task=lift num_episodes=50 version=v1.1
```

### HDF5 Dataset Structure

Each HDF5 file contains:

```
{task}_{version}_{date}.h5
├── [Attributes]
│   ├── task: "lift"
│   ├── version: "v1.0"
│   ├── num_episodes: 150
│   └── cameras: ["front", "right_wrist"]
├── episode_0/
│   ├── qpos: (T, 6)           # Joint positions
│   ├── action: (T, 6)         # Joint actions
│   ├── reward: (T,)           # Rewards
│   ├── done: (T,)             # Episode done flags
│   └── images/
│       ├── front: (T, H, W, 3)
│       └── right_wrist: (T, H, W, 3)
├── episode_1/
│   └── ...
```

### Best Practices

- **Quality over quantity**: Collect at least 50-100 episodes per version
- **Version control**: Use same `version` for related collections, increment when strategy changes
- **IK failures**: Script automatically skips episodes with IK failures (only saves successful ones)

## 🏋️ Training

### Train Diffusion Policy

The training script automatically detects the latest dataset:

```bash
python scripts/train.py task=lift batch_size=8 epochs=100
```

**Key Parameters:**
- `task`: Task name (`lift`, `sort`, `stack`)
- `batch_size`: Batch size (default: 8)
- `epochs`: Number of training epochs (default: 100)
- `lr`: Learning rate (default: 1e-4)
- `save_freq`: Checkpoint saving frequency (default: 10)

### Specify Dataset Path

```bash
python scripts/train.py task=lift dataset_path=data/datasets/lift/raw/lift_v1.0_20260101.h5
```

### Training Output

```
logs/train/{task}/{date}/{time}/
├── checkpoint_epoch_10.pth
├── checkpoint_epoch_20.pth
├── ...
├── checkpoint_epoch_100.pth
├── stats.json              # Normalization statistics (auto-saved)
└── logs/                   # TensorBoard logs
```

### Monitor Training

```bash
tensorboard --logdir logs/train/lift
```

## 🎯 Evaluation

### Evaluate Trained Policy

```bash
python scripts/eval.py \
    --checkpoint logs/train/lift/2026-01-01/12-32-31/checkpoint_epoch_100.pth \
    --task lift \
    --num-episodes 10
```

**Parameters:**
- `--checkpoint`: Path to checkpoint file (required)
- `--task`: Task name (required)
- `--num-episodes`: Number of episodes to evaluate (default: 10)
- `--device`: Device (`cuda` or `cpu`, default: `cuda`)
- `--headless`: Run without SAPIEN GUI
- `--no-web-viewer`: Disable web viewer
- `--port`: Web viewer port (default: 5000)

### Web Viewer Visualization

By default, evaluation starts a web viewer at [http://localhost:5000](http://localhost:5000). You can view real-time camera streams from:
- Front camera
- Right wrist camera
- Left wrist camera (for `sort` task)

### Evaluation Results

The script prints and saves:
- **Success Rate**: Percentage of successful episodes
- **Average Episode Length**: Mean number of steps per episode
- **Detailed Metrics**: JSON file saved to `{checkpoint_dir}/eval_results/`

Example output:
```
============================================================
📊 Evaluation Results
============================================================
Task: lift
Episodes: 10
Success Rate: 7/10 (70.0%)
Average Episode Length: 234.5 steps
Checkpoint: logs/train/lift/.../checkpoint_epoch_100.pth
============================================================

💾 Results saved to logs/train/lift/.../eval_results/eval_lift_20260101_153045.json
```

## 🛠️ Additional Tools

### Gripper Pose Visualization

Debug motion planning by visualizing target gripper poses:

```bash
python tools/visualize_gripper_pose.py --task lift
```

### Camera Calibration

Calibrate cameras for undistortion (real robot deployment):

```bash
python tools/calibration/calibration.py
python tools/calibration/undistort.py
```

## 🧩 Key Components

### Policy Architecture

- **DiffusionPolicy**: DDPM-based policy with 100 diffusion steps
- **Vision Encoder**: ResNet18 for image observations
- **Observations**: Joint positions (qpos) + RGB images from cameras
- **Actions**: Target joint positions

### Environment

- **Simulator**: SAPIEN 3.x with PhysX
- **Robot**: SO-101 dual-arm manipulator (6 DOF per arm)
- **Tasks**:
  - **Lift**: Pick up a cube and raise it above a threshold
  - **Stack**: Stack one cube on top of another
  - **Sort**: Sort cubes by color using both arms

### Physical Parameters

All objects use realistic friction and restitution values matched to the grasp-cube-sample project:
- Static friction: 2.0
- Dynamic friction: 2.0
- Restitution: 0.0

## 📦 Main Dependencies

- Python 3.10
- SAPIEN 3.x (physics simulation)
- PyTorch 2.7.x
- diffusers (DDPM scheduler)
- h5py (dataset format)
- hydra-core (config management)
- gymnasium (environment interface)

See [requirements.txt](requirements.txt) for complete list.

## 📝 Project Workflow Summary

```
1. Test Environment
   └─> python scripts/sim_env_demo.py --task lift

2. Collect Data
   └─> python scripts/collect_data.py task=lift num_episodes=50 version=v1.0

3. Train Policy
   └─> python scripts/train.py task=lift epochs=100

4. Evaluate
   └─> python scripts/eval.py --checkpoint logs/train/lift/.../checkpoint_epoch_100.pth --task lift
   └─> Open http://localhost:5000 to view results
```

## 🐛 Troubleshooting

### GPU Out of Memory

- Reduce `batch_size` in training config
- Use `CUDA_VISIBLE_DEVICES` to select specific GPU: `CUDA_VISIBLE_DEVICES=0 python scripts/train.py ...`

### IK Failures During Data Collection

The collection script automatically skips episodes with IK failures. If you see many failures:
- Check gripper pose targets in the FSM planner
- Verify robot joint limits
- Use `tools/visualize_gripper_pose.py` to debug

### Missing stats.json

If evaluation fails with "stats.json not found":
- Ensure training completed successfully
- Check that `stats.json` exists alongside the checkpoint
- Re-run training if necessary

### Web Viewer Not Loading

- Check if port 5000 is available
- Use `--port <other_port>` to change port
- Verify firewall settings

## 👥 Team

- Guanheng Chen
- Zuo Gou
- Zhengyang Fan

## 📄 License

This project is developed for educational purposes as part of the Embodied AI 2025 course.
