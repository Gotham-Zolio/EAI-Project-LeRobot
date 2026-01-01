# 数据采集与训练规范指南

## 📁 文件夹结构

```
data/
├── datasets/                           # 所有采集数据
│   ├── lift/
│   │   ├── raw/                        # 原始 HDF5 数据
│   │   │   ├── lift_v1.0_20250101.h5  # {task}_{version}_{date}.h5
│   │   │   └── lift_v1.0_20250102.h5
│   │   └── meta/                       # 元数据（JSON）
│   │       └── lift_v1.0_info.json
│   ├── stack/
│   └── sort/
└── logs/                               # 训练日志（自动生成）
    └── train/lift/...
```

## 📊 采集数据工作流

### 步骤 1：采集第一批数据

```bash
python scripts/collect_data.py task=lift num_episodes=50 version=v1.0
```

- 生成文件：`data/datasets/lift/raw/lift_v1.0_20250101.h5`
- 元数据：`data/datasets/lift/meta/lift_v1.0_info.json`

### 步骤 2：继续采集相同版本的数据（追加）

```bash
python scripts/collect_data.py task=lift num_episodes=50 version=v1.0
```

- 自动追加到同一个 `lift_v1.0_*.h5` 文件
- 元数据中的 `episodes_collected` 会更新（如果当天多次运行）

### 步骤 3：新版本采集

```bash
python scripts/collect_data.py task=lift num_episodes=50 version=v1.1
```

- 生成新文件：`data/datasets/lift/raw/lift_v1.1_20250102.h5`
- 适合用于：改进采集策略、修复 bug 后的重新采集

## 🔧 配置参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `task` | `lift` | 任务类型：lift、stack、sort |
| `num_episodes` | `50` | 本轮采集的 episode 数 |
| `version` | `v1.0` | 版本号（用于版本管理） |
| `save_dir` | `data/datasets` | 保存目录根路径 |
| `headless` | `True` | 无头模式（关闭 UI） |
| `verbose` | `False` | 详细日志输出 |

## 📝 元数据文件示例

**lift_v1.0_info.json**：
```json
{
  "task": "lift",
  "version": "v1.0",
  "created_date": "2025-01-02T14:30:00.123456",
  "episodes_collected": 150,
  "total_steps": 18000,
  "success_rate": 0.92,
  "cameras": ["front", "right_wrist"],
  "fps": 30,
  "h5_file": "lift_v1.0_20250102.h5"
}
```

## 🏋️ 训练工作流

### 自动检测数据

```bash
python scripts/train.py task=lift batch_size=8 epochs=100
```

训练脚本会自动查找：
1. 最新的 `data/datasets/lift/raw/lift_v*.h5` 文件
2. 或使用显式指定的路径

### 显式指定数据路径

```bash
python scripts/train.py task=lift dataset_path=data/datasets/lift/raw/lift_v1.0_20250102.h5 batch_size=8
```

### 训练输出

```
logs/train/lift/2025-01-02/14-30-00/
├── .hydra/                    # 配置备份
├── logs/                       # TensorBoard 日志
├── checkpoint_epoch_10.pth     # 模型检查点
└── stats.json                  # 归一化统计（自动保存）
```

## 💾 HDF5 内部结构

每个 HDF5 文件包含：

**文件属性** (attrs):
- `task`: 任务名称
- `version`: 版本号
- `num_episodes`: 总 episode 数
- `last_updated`: 最后更新时间
- `cameras`: 摄像头列表
- `collection_method`: "fsm_ik"

**Episode 数据** (episode_0, episode_1, ...):
```
episode_{id}/
├── qpos: (T, 6)           # 关节位置
├── action: (T, 6)         # 执行动作
├── reward: (T,)           # 即时奖励
├── done: (T,)             # 完成标志
└── images/
    ├── front: (T, H, W, 3)      # 前视图
    └── right_wrist: (T, H, W, 3) # 腕部视图
```

## 🎯 最佳实践

### 数据质量

- ✅ 单个版本建议至少采集 **50-100 episodes**
- ✅ 分多批采集时，使用相同 `version` 持续追加
- ✅ 当采集策略改进时，递增 `version`（如 v1.0 → v1.1）

### 版本管理

```
v1.0: 初始采集（可能有 bug）
v1.1: 修复已知 bug，重新采集一部分
v2.0: 改进采集策略后的大规模采集
```

### 训练前检查

```python
# 验证数据完整性
import h5py
with h5py.File("data/datasets/lift/raw/lift_v1.0_20250102.h5", "r") as f:
    print(f"Episodes: {f.attrs['num_episodes']}")
    print(f"Total steps: {sum(f[k]['action'].shape[0] for k in f.keys() if k.startswith('episode_'))}")
```

## 🔄 迁移旧数据

如果已有旧格式的 `data/lift_demo.h5`，可以：

```bash
# 指定旧路径运行训练
python scripts/train.py dataset_path=data/lift_demo.h5
```

训练脚本向后兼容旧路径格式。
