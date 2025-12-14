import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Union

import cv2
import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class DatasetWriter:
    def __init__(self, save_path):
        self.save_path = save_path
        self.data = {
            "observations": [],
            "actions": [],
            "rewards": [],
            "dones": []
        }
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

    def add_episode(self, obs_list, action_list, reward_list, done_list):
        # Stack lists into arrays
        # This is a simplified structure. 
        # For Diffusion Policy, we usually need:
        # /observations/qpos
        # /observations/images/front
        # /actions
        pass 
        # Implementation depends on exact data format needed by training script

    def save(self):
        # Save to HDF5
        pass


class LeRobotDataset(Dataset):
    def __init__(self, root: str, repo_id: str, split: str = "train"):
        """
        Args:
            root (str): Root directory containing the datasets (e.g. "data").
            repo_id (str): Name of the dataset (e.g. "lift").
            split (str): Split to load (e.g. "train").
        """
        self.root = Path(root)
        self.repo_id = repo_id
        self.split = split
        self.dataset_path = self.root / self.repo_id
        
        if not self.dataset_path.exists():
            raise ValueError(f"Dataset {self.repo_id} not found at {self.dataset_path}")

        # Load metadata
        self.meta_path = self.dataset_path / "meta/info.json"
        with open(self.meta_path, "r") as f:
            self.meta = json.load(f)
            
        self.fps = self.meta.get("fps", 30)
        self.features = self.meta.get("features", {})
        self.video = True 
        
        # Identify video keys
        self.video_keys = [k for k, v in self.features.items() if isinstance(v, dict) and v.get("dtype") == "video"]
        
        # Load data (parquet)
        self.data_dir = self.dataset_path / "data"
        
        # Find all parquet files
        # Pattern: chunk-{chunk_index}/file-{file_index}.parquet
        parquet_files = sorted(list(self.data_dir.glob("chunk-*/file-*.parquet")))
        
        dfs = []
        for p in parquet_files:
            try:
                df = pd.read_parquet(p)
            except Exception as e:
                print(f"Error reading {p}: {e}")
                continue
                
            # Extract chunk and file index from path
            # .../chunk-000/file-000.parquet
            try:
                chunk_idx = int(p.parent.name.split("-")[1])
                file_idx = int(p.stem.split("-")[1])
            except (IndexError, ValueError):
                print(f"Could not parse chunk/file index from {p}")
                continue
            
            df["chunk_index"] = chunk_idx
            df["file_index"] = file_idx
            
            # We assume row 0 in parquet is frame 0 in video for that file
            df["video_frame_index"] = np.arange(len(df))
            
            dfs.append(df)
            
        if dfs:
            self.data = pd.concat(dfs, ignore_index=True)
        else:
            raise RuntimeError(f"No parquet data found in {self.data_dir}")
            
        # Filter by split
        if "splits" in self.meta and self.split in self.meta["splits"]:
            split_range = self.meta["splits"][self.split]
            # Format "start:end" (episodes)
            if ":" in split_range:
                start, end = map(int, split_range.split(":"))
                self.data = self.data[
                    (self.data["episode_index"] >= start) & 
                    (self.data["episode_index"] < end)
                ]
            
        # Video capture cache
        self.cap_cache = {}

    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        # Handle negative indexing
        if idx < 0:
            idx += len(self)
            
        row = self.data.iloc[idx]
        
        item = {
            "action": torch.tensor(row["action"], dtype=torch.float32),
            "observation.state": torch.tensor(row["observation.state"], dtype=torch.float32),
        }
        
        # Add other columns if needed, e.g. episode_index
        item["episode_index"] = torch.tensor(row["episode_index"], dtype=torch.long)
        item["frame_index"] = torch.tensor(row["frame_index"], dtype=torch.long)
        
        if self.video:
            chunk_idx = int(row["chunk_index"])
            file_idx = int(row["file_index"])
            frame_idx = int(row["video_frame_index"])
            
            for key in self.video_keys:
                # Construct video path
                # "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
                # Note: key might be "observation.images.front"
                video_path = self.dataset_path / "videos" / key / f"chunk-{chunk_idx:03d}" / f"file-{file_idx:03d}.mp4"
                
                if not video_path.exists():
                    # Warn once
                    if not hasattr(self, "_warned_missing_video"):
                        print(f"Warning: Video file {video_path} not found. Returning zeros.")
                        self._warned_missing_video = True
                    
                    # Return zeros
                    # We need shape. info.json has it.
                    # "observation.images.front": { "shape": [480, 640, 3], ... }
                    if key in self.features and "shape" in self.features[key]:
                        shape = self.features[key]["shape"]
                        # Shape in json is usually H, W, C. Torch wants C, H, W.
                        if len(shape) == 3:
                            h, w, c = shape
                            item[key] = torch.zeros((c, h, w), dtype=torch.float32)
                        else:
                             # Fallback
                            item[key] = torch.zeros((3, 480, 640), dtype=torch.float32)
                    else:
                        item[key] = torch.zeros((3, 480, 640), dtype=torch.float32)
                    continue

                try:
                    frame = self._load_frame(str(video_path), frame_idx)
                    # Convert to torch (C, H, W) and float 0-1
                    frame = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
                    item[key] = frame
                except RuntimeError as e:
                    # Warn once
                    if not hasattr(self, "_warned_read_error"):
                        print(f"Warning: Failed to read frame from {video_path}: {e}. Returning zeros.")
                        self._warned_read_error = True
                    
                    # Return zeros
                    if key in self.features and "shape" in self.features[key]:
                        shape = self.features[key]["shape"]
                        if len(shape) == 3:
                            h, w, c = shape
                            item[key] = torch.zeros((c, h, w), dtype=torch.float32)
                        else:
                            item[key] = torch.zeros((3, 480, 640), dtype=torch.float32)
                    else:
                        item[key] = torch.zeros((3, 480, 640), dtype=torch.float32)
                
        return item

    def _load_frame(self, video_path, frame_idx):
        if video_path not in self.cap_cache:
            self.cap_cache[video_path] = cv2.VideoCapture(video_path)
            
        cap = self.cap_cache[video_path]
        
        # Check if we need to seek
        # Optimization: if we are at the right frame, just read
        # But we don't know current pos easily without querying, which is also overhead.
        # Safe way: always set.
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        
        if not ret:
            # Try reopening once
            cap.release()
            cap = cv2.VideoCapture(video_path)
            self.cap_cache[video_path] = cap
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                # Return black frame or raise error?
                # Raise error to be safe
                raise RuntimeError(f"Could not read frame {frame_idx} from {video_path}")
                
        # BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame
        
    def __del__(self):
        if hasattr(self, "cap_cache"):
            for cap in self.cap_cache.values():
                cap.release()

if __name__ == "__main__":
    # Test the dataset
    try:
        ds = LeRobotDataset(root="data", repo_id="lift")
        print(f"Loaded dataset with {len(ds)} frames")
        item = ds[0]
        print("Keys:", item.keys())
        print("Action shape:", item["action"].shape)
        if "observation.images.front" in item:
            print("Image shape:", item["observation.images.front"].shape)
    except Exception as e:
        print(f"Failed to load dataset: {e}")
