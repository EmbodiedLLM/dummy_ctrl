import json
import time
import numpy as np
import torch
from pathlib import Path
import logging
from datasets import Dataset
from typing import Optional, Dict, List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RobotDataCollector:
    def __init__(
        self, 
        output_dir: str,
        fps: int = 10, 
        robot_type: str = "custom"
    ):
        """Initialize data collector focused on parquet data storage
        
        Args:
            output_dir: Directory to store robot control data
            fps: Frames per second for data collection
            robot_type: Type of robot being controlled
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create directory structure
        self.meta_dir = self.output_dir / "meta"
        self.meta_dir.mkdir(exist_ok=True)
        self.all_episodes_data = []
        self.train_dir = self.output_dir / "data"
        self.train_dir.mkdir(exist_ok=True)
        self.current_chunk_dir = self.train_dir / "chunk-000"
        self.current_chunk_dir.mkdir(exist_ok=True)
        
        # Basic info
        self.fps = fps
        self.episode_count = 0
        self.total_frames = 0
        self.robot_type = robot_type
        self.episode_data_index = {"from": [], "to": []}
        self.episode_lengths = []
        
        # Current episode data
        self.current_episode_data = {
            "observation.state": [],
            "action": [],
            "episode_index": [],
            "frame_index": [],
            "timestamp": [],
            "next.done": [],
            "index": [],
            "task_index": []
        }
        
        self.frame_count = 0
        self.start_time = None
        
        logger.info(f"RobotDataCollector initialized with output directory: {output_dir}")

    def start_episode(self):
        """Start new episode recording"""
        self.start_time = time.time()
        self.frame_count = 0
        
        # Reset episode buffer
        for key in self.current_episode_data:
            self.current_episode_data[key] = []
        
        logger.info(f"Started episode {self.episode_count}")
        
    def collect_step(self, teach_joints, follow_joints, teach_gripper, follow_gripper, rate):
        """Collect one timestep of data"""
        if self.start_time is None:
            self.start_episode()

        # Use fixed timestamp increment of rate seconds
        timestamp = np.float64(self.frame_count * rate)
        
        # Store state and action data
        self.current_episode_data["observation.state"].append(
            np.concatenate([follow_joints, [float(follow_gripper)]]).tolist()
        )
        self.current_episode_data["action"].append(
            np.concatenate([teach_joints, [float(teach_gripper)]]).tolist()
        )
        self.current_episode_data["episode_index"].append(self.episode_count)
        self.current_episode_data["frame_index"].append(self.frame_count)
        self.current_episode_data["timestamp"].append(timestamp)  # Use fixed interval timestamp
        self.current_episode_data["next.done"].append(False)
        self.current_episode_data["index"].append(self.total_frames + self.frame_count)
        self.current_episode_data["task_index"].append(0)
        
        self.frame_count += 1
        
    def save_episode(self):
        """Save episode data to parquet file"""
        if self.frame_count == 0:
            logger.warning("No frames to save")
            return
            
        # Update last frame done state
        if len(self.current_episode_data["next.done"]) > 0:
            self.current_episode_data["next.done"][-1] = True
        
        # Get next index by checking existing files in the chunk directory
        existing_parquets = list(self.current_chunk_dir.glob("episode_*.parquet"))
        next_index = len(existing_parquets)
                
        # Store episode data in memory
        episode_data = {}
        for key, values in self.current_episode_data.items():
            episode_data[key] = np.array(values)
        self.all_episodes_data.append(episode_data)
        
        # Save parquet file with the index
        parquet_filename = f"episode_{next_index:06d}.parquet"
        parquet_path = self.current_chunk_dir / parquet_filename
        
        dataset = Dataset.from_dict(self.current_episode_data)
        dataset.to_parquet(str(parquet_path))
        
        # Update counters
        self.total_frames += self.frame_count
        self.episode_count += 1
        self.episode_lengths.append(self.frame_count)
        
        logger.info(f"Episode {next_index} saved with {self.frame_count} frames to {parquet_path}")
        
        # Reset frame count and prepare for next episode
        self.frame_count = 0
        return next_index