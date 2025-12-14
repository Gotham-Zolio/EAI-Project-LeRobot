import sys
import os

# Add src to path to allow importing local lerobot modules
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, "../src")
sys.path.insert(0, src_path)

import json
from pathlib import Path

import hydra
import torch
import torch.nn as nn
import torch.optim as optim
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Now these imports should work from src/lerobot
from lerobot.common.dataset import LeRobotDataset
from lerobot.policy.diffusion import DiffusionPolicy


def get_stats(dataset_path, repo_id):
    stats_path = Path(dataset_path) / repo_id / "meta/stats.json"
    with open(stats_path, "r") as f:
        stats = json.load(f)
    return stats

def normalize(data, stats, key):
    if key not in stats:
        return data
    
    mean = torch.tensor(stats[key]["mean"], device=data.device, dtype=data.dtype)
    std = torch.tensor(stats[key]["std"], device=data.device, dtype=data.dtype)
    
    if isinstance(stats[key]["mean"][0], list):
        return data

    return (data - mean) / (std + 1e-8)

def unnormalize(data, stats, key):
    if key not in stats:
        return data
    
    mean = torch.tensor(stats[key]["mean"], device=data.device, dtype=data.dtype)
    std = torch.tensor(stats[key]["std"], device=data.device, dtype=data.dtype)
    
    if isinstance(stats[key]["mean"][0], list):
        return data

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

    # 1. Dataset
    log.info(f"Loading dataset: {cfg.task}...")
    # Use local dataset implementation which handles missing files
    dataset = LeRobotDataset(root="data", repo_id=cfg.task, split="train")
    log.info(f"Dataset loaded with {len(dataset)} items")
    dataloader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=0)
    
    # Load stats for normalization
    stats = get_stats("data", cfg.task)

    # 2. Model
    # Get dimensions from dataset
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
