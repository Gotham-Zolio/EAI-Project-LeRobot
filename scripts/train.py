import sys
import os

# Add src to path to allow importing local lerobot modules
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, "../src")
sys.path.insert(0, src_path)

import json
from pathlib import Path
from typing import Dict, List, Optional

import h5py
import hydra
import torch
import torch.nn as nn
import torch.optim as optim
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Now these imports should work from src/lerobot
from lerobot.policy.diffusion import DiffusionPolicy


class H5TrajectoryDataset(Dataset):
    """Lightweight dataset for FSM/IK collected HDF5 demos.

    Expected HDF5 layout:
    - attrs: cameras (list[str])
    - groups: episode_0, episode_1, ...
      - qpos: (T, dq)
      - action: (T, da)
      - reward: (T,)
      - done: (T,)
      - images/{camera}: (T or T+1, H, W, C) uint8
    We align each action with the image and qpos at the same timestep t; if images
    have an extra final frame (T+1), we drop the last one.
    """

    def __init__(self, h5_path: str, cameras: Optional[List[str]] = None):
        super().__init__()
        self.h5_path = h5_path
        self.file = h5py.File(self.h5_path, "r")
        self.cameras = cameras or list(self.file.attrs.get("cameras", []))
        self.index: list[tuple[str, int]] = []

        for ep_name in self.file.keys():
            grp = self.file[ep_name]
            T = grp["action"].shape[0]
            self.index.extend([(ep_name, t) for t in range(T)])

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        ep, t = self.index[idx]
        grp = self.file[ep]
        item: dict[str, torch.Tensor] = {}

        action_np = grp["action"][t]
        qpos_np = grp["qpos"][t]
        item["action"] = torch.tensor(action_np, dtype=torch.float32)
        item["observation.state"] = torch.tensor(qpos_np, dtype=torch.float32)

        if "images" in grp and self.cameras:
            img_grp = grp["images"]
            for cam in self.cameras:
                if cam not in img_grp:
                    continue
                imgs = img_grp[cam]
                # images may have length T or T+1; clamp to valid index
                t_idx = min(t, imgs.shape[0] - 1)
                img = torch.from_numpy(imgs[t_idx]).permute(2, 0, 1).float() / 255.0
                # Some captures are RGBA; drop alpha to keep 3 channels
                if img.shape[0] > 3:
                    img = img[:3]
                elif img.shape[0] == 1:
                    img = img.repeat(3, 1, 1)
                item[f"observation.images.{cam}"] = img

        return item

    def close(self) -> None:
        try:
            self.file.close()
        except Exception:
            pass


def compute_stats_from_dataset(ds: Dataset, device: torch.device) -> dict:
    """Compute mean/std for action and observation.state over the dataset.
    Uses small batches to avoid memory spikes on large datasets.
    """
    keys = ["action", "observation.state"]
    sums = {k: 0.0 for k in keys}
    sums_sq = {k: 0.0 for k in keys}
    counts = {k: 0 for k in keys}

    # Use small batch size for memory efficiency
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)
    pbar = tqdm(loader, desc="Computing stats")
    for batch in pbar:
        for k in keys:
            v = batch[k].to(device)
            sums[k] += v.sum(dim=0)
            sums_sq[k] += (v * v).sum(dim=0)
            counts[k] += v.shape[0]

    stats = {}
    for k in keys:
        mean = sums[k] / counts[k]
        var = sums_sq[k] / counts[k] - mean * mean
        std = torch.sqrt(torch.clamp(var, min=1e-8))
        stats[k] = {"mean": mean.cpu(), "std": std.cpu()}
    return stats


def normalize(data, stats, key):
    if key not in stats:
        return data

    mean = stats[key]["mean"].to(data.device, dtype=data.dtype)
    std = stats[key]["std"].to(data.device, dtype=data.dtype)
    return (data - mean) / (std + 1e-8)


def unnormalize(data, stats, key):
    if key not in stats:
        return data

    mean = stats[key]["mean"].to(data.device, dtype=data.dtype)
    std = stats[key]["std"].to(data.device, dtype=data.dtype)
    return data * (std + 1e-8) + mean


import logging

# Configure logging
log = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="../configs", config_name="train")
def train(cfg: DictConfig):
    log.info("Entered train function")
    log.info(f"Training with config:\n{cfg}")
    
    log.info("Checking device...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Using device: {device}")

    # 1. Dataset (FSM/IK HDF5)
    # Support multiple ways to specify dataset:
    # 1. Explicit path via dataset_path
    # 2. Version via dataset_version (finds data/datasets/{task}/raw/{task}_{version}_*.h5)
    # 3. Auto-detect latest
    
    h5_path = None
    
    # Priority 1: Explicit path
    if cfg.dataset_path:
        candidate = Path(cfg.dataset_path)
        if candidate.exists():
            h5_path = candidate
            log.info(f"Using explicit dataset path: {h5_path}")
        else:
            raise FileNotFoundError(f"Specified dataset_path not found: {cfg.dataset_path}")
    
    # Priority 2: Version-based search
    elif cfg.get('dataset_version'):
        version = cfg.dataset_version
        search_pattern = Path("data") / "datasets" / cfg.task / "raw" / f"{cfg.task}_{version}_*.h5"
        matches = list(search_pattern.parent.glob(search_pattern.name))
        if matches:
            h5_path = sorted(matches)[-1]  # Use latest if multiple dates
            log.info(f"Found dataset for version {version}: {h5_path}")
        else:
            raise FileNotFoundError(
                f"No dataset found for version {version}.\n"
                f"Expected pattern: {search_pattern}\n"
                f"Run: python scripts/collect_data.py task={cfg.task} version={version}"
            )
    
    # Priority 3: Auto-detect latest
    else:
        h5_candidates = [
            Path("data") / f"{cfg.task}_demo.h5",  # Old structure
            Path("data") / "datasets" / cfg.task / "raw" / f"{cfg.task}_v*.h5",  # New structure (glob)
        ]
        
        for candidate in h5_candidates:
            if "*" in str(candidate):
                # Handle glob patterns
                matches = list(candidate.parent.glob(candidate.name))
                if matches:
                    h5_path = sorted(matches)[-1]  # Use latest if multiple
                    log.info(f"Auto-detected latest dataset: {h5_path}")
                    break
            elif candidate.exists():
                h5_path = candidate
                log.info(f"Found dataset: {h5_path}")
                break
    
    if h5_path is None:
        raise FileNotFoundError(
            f"HDF5 dataset not found for task '{cfg.task}'.\n"
            f"Options:\n"
            f"  1. Collect data: python scripts/collect_data.py task={cfg.task}\n"
            f"  2. Specify version: python scripts/train.py task={cfg.task} dataset_version=v1.0\n"
            f"  3. Specify path: python scripts/train.py task={cfg.task} dataset_path=path/to/data.h5"
        )

    log.info(f"Loading HDF5 dataset from {h5_path}...")
    dataset = H5TrajectoryDataset(str(h5_path))
    log.info(f"Dataset loaded with {len(dataset)} steps")
    dataloader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=0)

    # Compute stats from dataset
    log.info("Computing normalization stats from dataset...")
    stats = compute_stats_from_dataset(dataset, device)
    log.info("Stats computed")
    
    # Extract version from dataset filename (e.g., lift_v1.0_20260101.h5 -> v1.0)
    dataset_version = "unknown"
    try:
        filename = h5_path.name
        if "_v" in filename:
            version_part = filename.split("_v")[1].split("_")[0]
            dataset_version = f"v{version_part}"
    except Exception:
        pass
    
    # Save stats for reference with dataset metadata
    stats_path = Path(hydra.core.hydra_config.HydraConfig.get().run.dir) / "stats.json"
    stats_dict = {
        "dataset_path": str(h5_path),
        "dataset_version": dataset_version,
        "task": cfg.task,
        "normalization": {k: {"mean": v["mean"].tolist(), "std": v["std"].tolist()} for k, v in stats.items()}
    }
    # Keep backward compatibility - also save stats at root level
    stats_dict.update({k: {"mean": v["mean"].tolist(), "std": v["std"].tolist()} for k, v in stats.items()})
    with open(stats_path, "w") as f:
        json.dump(stats_dict, f, indent=2)
    log.info(f"Stats saved to {stats_path}")
    log.info(f"Dataset version: {dataset_version}")

    # 2. Model
    sample_item = dataset[0]
    action_dim = sample_item["action"].shape[0]
    obs_dim = sample_item["observation.state"].shape[0]
    
    log.info("Initializing policy...")
    policy = DiffusionPolicy(
        action_dim=action_dim,
        obs_dim=obs_dim,
        vision_backbone=cfg.policy.backbone,
        num_diffusion_steps=cfg.policy.num_diffusion_steps
    ).to(device)

    optimizer = optim.AdamW(policy.parameters(), lr=cfg.lr)
    
    # 3. Training Loop
    num_epochs = cfg.epochs
    save_dir = Path(hydra.core.hydra_config.HydraConfig.get().run.dir)
    checkpoints_dir = save_dir / "checkpoints"
    checkpoints_dir.mkdir(exist_ok=True)
    
    # Initialize TensorBoard writer
    writer = SummaryWriter(log_dir=str(save_dir / "logs"))
    global_step = 0
    
    # Log hyperparameters and dataset info
    hparams = {
        "learning_rate": cfg.lr,
        "batch_size": cfg.batch_size,
        "epochs": cfg.epochs,
        "save_freq": cfg.save_freq,
        "action_dim": action_dim,
        "obs_dim": obs_dim,
        "policy_backbone": cfg.policy.backbone,
        "num_diffusion_steps": cfg.policy.num_diffusion_steps,
    }
    
    dataset_info = {
        "task": cfg.task,
        "dataset_version": dataset_version,
        "total_steps": len(dataset),
        "batch_count": len(dataloader),
    }
    
    log.info(f"\n{'='*60}")
    log.info("Training Configuration")
    log.info(f"{'='*60}")
    log.info(f"Task: {cfg.task}")
    log.info(f"Dataset Version: {dataset_version}")
    log.info(f"Dataset Path: {h5_path}")
    log.info(f"Total Steps: {len(dataset)} | Batches: {len(dataloader)}")
    log.info(f"Model: {cfg.policy.backbone} | Action Dim: {action_dim} | Obs Dim: {obs_dim}")
    log.info(f"Hyperparameters: LR={cfg.lr} | BS={cfg.batch_size} | Epochs={cfg.epochs}")
    log.info(f"{'='*60}\n")
    
    # Write hyperparameters to TensorBoard
    writer.add_hparams(hparams, {}, run_name=cfg.task)

    log.info("Starting training loop...")
    for epoch in range(num_epochs):
        policy.train()
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        epoch_loss = 0.0
        batch_count = 0
        
        for batch in pbar:
            # Move to device
            batch = {k: v.to(device) for k, v in batch.items()}

            # Normalize
            batch["action"] = normalize(batch["action"], stats, "action")
            batch["observation.state"] = normalize(batch["observation.state"], stats, "observation.state")
            # Images are already 0-1 from dataset

            optimizer.zero_grad()
            loss = policy(batch)
            loss.backward()
            
            # Compute gradient norm for monitoring
            total_grad_norm = 0.0
            for p in policy.parameters():
                if p.grad is not None:
                    total_grad_norm += p.grad.data.norm(2).item() ** 2
            total_grad_norm = total_grad_norm ** 0.5
            
            optimizer.step()
            
            epoch_loss += loss.item()
            batch_count += 1
            pbar.set_postfix({"loss": loss.item():.4f}, refresh=False)
            
            # Log to TensorBoard
            writer.add_scalar("Loss/batch", loss.item(), global_step)
            writer.add_scalar("GradientNorm/batch", total_grad_norm, global_step)
            
            # Log learning rate
            for param_group in optimizer.param_groups:
                writer.add_scalar("LearningRate", param_group['lr'], global_step)
                break
            
            global_step += 1
            
        avg_loss = epoch_loss / batch_count
        log.info(f"Epoch {epoch+1}/{num_epochs} | Loss: {avg_loss:.4f}")
        writer.add_scalar("Loss/epoch", avg_loss, epoch)
        writer.add_scalar("Loss/epoch_smoothed", avg_loss, epoch)  # For plotting trends
        
        # Prepare checkpoint data
        checkpoint_data = {
            'epoch': epoch,
            'model_state_dict': policy.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
        }
        
        # Save periodic checkpoint
        if (epoch + 1) % cfg.save_freq == 0:
            checkpoint_path = checkpoints_dir / f"checkpoint_epoch_{epoch+1}.pth"
            torch.save(checkpoint_data, checkpoint_path)
            log.info(f"Saved checkpoint to {checkpoint_path}")
        
        # Always save/update last.pth
        last_checkpoint_path = checkpoints_dir / "last.pth"
        torch.save(checkpoint_data, last_checkpoint_path)
        if (epoch + 1) % 10 == 0:  # Log every 10 epochs to avoid spam
            log.info(f"Updated last.pth (epoch {epoch+1})")

    writer.close()
    log.info("Training complete.")

if __name__ == "__main__":
    train()
