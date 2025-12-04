# EAI Course Project: Diffusion Policy for LeRobot SO-101

This repository contains the implementation of **Diffusion Policy** for the Embodied AI 2025 Course Project (Track 1). The goal is to control a simulated LeRobot SO-101 manipulator to perform Lift, Stack, and Sort tasks, as well as a custom manipulation task.

## 📂 Repository Structure

```
EAI-CourseProject-LeRobot/
├── configs/              # Configuration files for training and simulation
├── dataset/              # Expert demonstration data (ignored by git)
├── docs/                 # Project documentation and reference PDFs
├── midterm_report/       # LaTeX source for the midterm report
├── scripts/              # Python scripts for simulation, training, and utilities
│   ├── simulation_setup.py   # Main simulation environment setup
│   └── ...
├── .gitignore            # Git ignore rules
└── README.md             # Project overview and instructions
```

## 🚀 Project Roadmap

### Phase 1: Simulation & Environment Setup (Dec 1 - Dec 7)
- [x] Set up SAPIEN/Gym environment with SO-101 robot.
- [x] Configure camera sensors (Front, Left Wrist, Right Wrist).
- [ ] Implement task environments: Lift, Stack, Sort.
- [ ] Design and implement the custom task.

### Phase 2: Data Collection (Dec 8 - Dec 14)
- [ ] Collect expert demonstrations for all tasks.
- [ ] Preprocess data (normalization, chunking) for Diffusion Policy training.

### Phase 3: Policy Training (Dec 15 - Dec 21)
- [ ] Implement DDPM-based Diffusion Policy.
- [ ] Train policies on collected datasets.
- [ ] Tune hyperparameters (noise schedule, horizon, etc.).

### Phase 4: Real Robot Deployment (Dec 22 - Dec 26)
- [ ] Transfer trained policies to the real SO-101 robot.
- [ ] Address Sim-to-Real gaps (visual domain randomization, dynamics tuning).

### Phase 5: Evaluation & Reporting (Dec 27 - Jan 9)
- [ ] Quantitative evaluation (Success Rate).
- [ ] Qualitative analysis (Video demos, failure cases).
- [ ] Final Report and Presentation.

## 🛠️ Usage

### 1. Installation
```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Running Simulation
```bash
python scripts/simulation_setup.py
```

### 3. Training
*(Instructions to be added)*

## 👥 Team
- Guanheng Chen
- Zuo Gou
- Zhengyang Fan
