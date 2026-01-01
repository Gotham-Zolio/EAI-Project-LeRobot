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
    # Support both old and new path structures
    h5_candidates = [
        Path(cfg.dataset_path) if cfg.dataset_path else None,
        Path("data") / f"{cfg.task}_demo.h5",  # Old structure
        Path("data") / "datasets" / cfg.task / "raw" / f"{cfg.task}_v1.0_*.h5",  # New structure (glob)
    ]
    
    h5_path = None
    for candidate in h5_candidates:
        if candidate is None:
            continue
        if "*" in str(candidate):
            # Handle glob patterns
            matches = list(Path(candidate).parent.glob(candidate.name))
            if matches:
                h5_path = sorted(matches)[-1]  # Use latest if multiple
                break
        elif candidate.exists():
            h5_path = candidate
            break
    
    if h5_path is None:
        raise FileNotFoundError(
            f"HDF5 dataset not found. Tried:\n"
            f"  - {h5_candidates[0]}\n"
            f"  - {h5_candidates[1]}\n"
            f"  - {h5_candidates[2]} (latest match)\n"
            f"Run collect_data.py first or set cfg.dataset_path."
        )

    log.info(f"Loading HDF5 dataset from {h5_path}...")
    dataset = H5TrajectoryDataset(str(h5_path))
    log.info(f"Dataset loaded with {len(dataset)} steps")
    dataloader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=0)

    # Compute stats from dataset
    log.info("Computing normalization stats from dataset...")
    stats = compute_stats_from_dataset(dataset, device)
    log.info("Stats computed")
    
    # Save stats for reference
    stats_path = Path(hydra.core.hydra_config.HydraConfig.get().run.dir) / "stats.json"
    stats_dict = {k: {"mean": v["mean"].tolist(), "std": v["std"].tolist()} for k, v in stats.items()}
    with open(stats_path, "w") as f:
        json.dump(stats_dict, f, indent=2)
    log.info(f"Stats saved to {stats_path}")

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
    
    # Initialize TensorBoard writer
    writer = SummaryWriter(log_dir=str(save_dir / "logs"))
    global_step = 0

    log.info("Starting training loop...")
    for epoch in range(num_epochs):
        policy.train()
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        epoch_loss = 0.0
        
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
            optimizer.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix({"loss": loss.item()})
            
            # Log to TensorBoard
            writer.add_scalar("Loss/train", loss.item(), global_step)
            global_step += 1
            
        avg_loss = epoch_loss / len(dataloader)
        log.info(f"Epoch {epoch+1} Average Loss: {avg_loss:.4f}")
        writer.add_scalar("Loss/epoch", avg_loss, epoch)
        
        # Save checkpoint
        if (epoch + 1) % cfg.save_freq == 0:
            checkpoint_path = save_dir / f"checkpoint_epoch_{epoch+1}.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': policy.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, checkpoint_path)
            log.info(f"Saved checkpoint to {checkpoint_path}")

    writer.close()
    log.info("Training complete.")

if __name__ == "__main__":
    train()
