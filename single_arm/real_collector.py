import json
import time
import numpy as np
import torch
from pathlib import Path
from datetime import datetime
import logging
import cv2
import os
import shutil
from safetensors.torch import save_file
from typing import Optional, Dict, List
from datasets import Dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LeRobotDataCollector:
    def __init__(
        self, 
        output_dir: str = "/Users/jack/Desktop/dummy_ctrl/datasets/robot_data_lerobot", 
        fps: int = 10, 
        camera_url: Optional[str] = None,
        robot_type: str = "custom", 
        use_video: bool = True
    ):
        """Initialize data collector with camera support that follows LeRobot format"""
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
        
        self.videos_dir = self.output_dir / "videos"
        if use_video:
            self.videos_dir.mkdir(exist_ok=True)
            self.current_videos_chunk_dir = self.videos_dir / "chunk-000"
            self.current_videos_chunk_dir.mkdir(exist_ok=True)
            self.camera_dir = self.current_videos_chunk_dir / "observation.images.cam_wrist"
            self.camera_dir.mkdir(exist_ok=True)
            
        # Basic info
        self.fps = fps
        self.episode_count = 0
        self.total_frames = 0
        self.robot_type = robot_type
        self.use_video = use_video
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

        # Temp directory for video encoding
        self.tmp_img_dir = self.output_dir / "tmp_images"

        # Camera setup
        self.camera_url = camera_url
        self.cap = None
        if camera_url:
            self.setup_camera()

    def setup_camera(self) -> bool:
        """Setup camera capture"""
        logger.info(f"Connecting to camera: {self.camera_url}")
        self.cap = cv2.VideoCapture(self.camera_url)
        
        if not self.cap.isOpened():
            logger.error("Failed to connect to camera")
            return False
            
        logger.info("Camera connected successfully")
        return True
        
    def start_episode(self):
        """Start new episode recording"""
        self.start_time = time.time()
        self.frame_count = 0
        
        # Reset episode buffer
        for key in self.current_episode_data:
            self.current_episode_data[key] = []
            
        # Create clean temp image directory
        if self.use_video:
            if self.tmp_img_dir.exists():
                shutil.rmtree(self.tmp_img_dir)
            self.tmp_img_dir.mkdir(parents=True, exist_ok=True)
        
        # Reset and clear camera buffer completely
        if self.camera_url:
            # Release old camera instance
            if self.cap:
                self.cap.release()
                self.cap = None
                time.sleep(0.1)  # Give camera time to properly close
            
            # Setup new camera
            self.setup_camera()
            if self.cap and self.cap.isOpened():
                # Set camera properties
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffer size
                self.cap.set(cv2.CAP_PROP_FPS, self.fps)
                
                # Clear buffer thoroughly
                t_start = time.time()
                while time.time() - t_start < 1.0:  # Clear buffer for 1 second
                    self.cap.grab()  # Just grab frames without decoding
                
                # Read a few more frames to ensure clean start
                for _ in range(10):
                    self.cap.read()
        
        logger.info(f"Started episode {self.episode_count}")
        
    def collect_step(self, teach_joints, follow_joints, teach_gripper, follow_gripper, rate):
        """Collect one timestep of data"""
        if self.start_time is None:
            self.start_episode()

        # Use fixed timestamp increment of rate seconds
        timestamp = np.float64(self.frame_count * rate)
        
        # Capture frame if using camera
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret and self.use_video:
                img_path = self.tmp_img_dir / f"frame_{self.frame_count:05d}.png"
                cv2.imwrite(str(img_path), frame)
                
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
        """Save episode data"""
        if self.frame_count == 0:
            logger.warning("No frames to save")
            return
            
        # Update last frame done state
        if len(self.current_episode_data["next.done"]) > 0:
            self.current_episode_data["next.done"][-1] = True
        
        # Get next index by checking existing files in both directories
        existing_parquets = list(self.current_chunk_dir.glob("episode_*.parquet"))
        next_index = len(existing_parquets)
                
        # Save video if enabled
        if self.use_video and self.cap:
            # Check for existing videos to maintain consistent indexing
            video_filename = f"episode_{next_index:06d}.mp4"
            video_path = self.camera_dir / video_filename
            
            cmd = f"ffmpeg -y -framerate {self.fps} -i {self.tmp_img_dir}/frame_%05d.png "\
                f"-c:v libx264 -pix_fmt yuv420p -crf 23 {str(video_path)}"
            os.system(cmd)
            logger.info(f"Video saved to {video_path}")
        
        # Store episode data in memory
        episode_data = {}
        for key, values in self.current_episode_data.items():
            episode_data[key] = np.array(values)
        self.all_episodes_data.append(episode_data)
        
        # Save parquet file with same index
        parquet_filename = f"episode_{next_index:06d}.parquet"
        parquet_path = self.current_chunk_dir / parquet_filename
        
        dataset = Dataset.from_dict(self.current_episode_data)
        dataset.to_parquet(str(parquet_path))
        logger.info(f"Parquet saved to {parquet_path}")
        
        # Update counters
        self.total_frames += self.frame_count
        self.episode_count += 1
        
        # Cleanup
        if self.tmp_img_dir.exists():
            shutil.rmtree(self.tmp_img_dir)
        self.episode_lengths.append(self.frame_count)        
        logger.info(f"Episode {next_index} saved with {self.frame_count} frames")
        self.frame_count = 0

    def finalize_dataset(self):
        """Finalize dataset and save metadata files according to ALOHA format"""
        info_path = self.meta_dir / "info.json"
        if info_path.exists():
            with open(info_path, 'r') as f:
                info = json.load(f)
            # Update counts
            info["total_episodes"] = self.episode_count
            info["total_frames"] = self.total_frames
            info["total_videos"] = self.episode_count if self.use_video else 0
            info["splits"]["train"] =  f"{round(self.episode_count*0.9,0)}:{self.episode_count-round(self.episode_count*0.9,0)}"
        else:
            # 1. Create info.json
            info = {
                "codebase_version": "v2.1",
                "robot_type": self.robot_type,
                "total_episodes": self.episode_count,
                "total_frames": self.total_frames,
                "total_tasks": 1,  # Single task for now
                "total_videos": self.episode_count if self.use_video else 0,
                "total_chunks": 1,
                "chunks_size": 1000,
                "fps": self.fps,
                "splits": {
                    "train": f"{self.episode_count}:{round(self.episode_count*0.1,0)}"
                },
                "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
                "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
                "features": {
                    "observation.images.cam_wrist": {
                        "dtype": "video",
                        "shape": [720, 1280, 3],
                        "names": ["height", "width", "channel"],
                        "video_info": {
                            "video.fps": float(self.fps),
                            "video.codec": "av1",
                            "video.pix_fmt": "yuv420p",
                            "video.is_depth_map": False,
                            "has_audio": False
                        }
                    },
                    "observation.state": {
                        "dtype": "float32",
                        "shape": [7],  # 6-DOF robot + gripper
                        "names": {
                            "motors": [f"joint_{i+1}" for i in range(6)] + ["gripper"]
                        }
                    },
                    "action": {
                        "dtype": "float32",
                        "shape": [7],  # 6 joints + gripper
                        "names": {
                            "motors": [f"joint_{i+1}" for i in range(6)] + ["gripper"]
                        }
                    },
                    "episode_index": {
                        "dtype": "int64",
                        "shape": [1],
                        "names": None
                    },
                    "frame_index": {
                        "dtype": "int64",
                        "shape": [1],
                        "names": None
                    },
                    "timestamp": {
                        "dtype": "float64",
                        "shape": [1],
                        "names": None
                    },
                    "next.done": {
                        "dtype": "bool",
                        "shape": [1],
                        "names": None
                    },
                    "index": {
                        "dtype": "int64",
                        "shape": [1],
                        "names": None
                    },
                    "task_index": {
                        "dtype": "int64",
                        "shape": [1],
                        "names": None
                    }
                }
            }
            
        with open(info_path, 'w') as f:
            json.dump(info, f, indent=4)
            
        # 2. Create episodes.jsonl
        episodes_path = self.meta_dir / "episodes.jsonl"
        with open(episodes_path, 'w') as f:
            for i in range(self.episode_count):
                episode_data = {
                    "episode_index": i,
                    "tasks": ["Pick the cube into the box"],  # Replace with actual task description
                    "length": self.episode_lengths[i]  # Fixed length episodes
                }
                f.write(json.dumps(episode_data) + "\n")
        # 3. Create episodes_stats.jsonl
        stats_path = self.meta_dir / "episodes_stats.jsonl"
        with open(stats_path, 'w') as f:
            for i, episode_data in enumerate(self.all_episodes_data):
                # 计算基础统计信息
                obs_stats = self._compute_stats(episode_data["observation.state"])
                action_stats = self._compute_stats(episode_data["action"])
                
                stats = {
                    "episode_index": i,
                    "stats": {
                        "observation.state": obs_stats,
                        "action": action_stats
                    }
                }
                # Add video stats if using video
                if self.use_video:
                    video_stats = self._compute_video_stats(i)
                    if video_stats:
                        stats["stats"]["observation.images.cam_wrist"] = video_stats
                f.write(json.dumps(stats) + "\n")
        # 4. Create tasks.jsonl
        task_path = self.meta_dir / "tasks.jsonl"
        if not task_path.exists():  
            with open(task_path, 'w') as f:
                task_data = {
                    "task_index": 0,
                    "task": "Pick the cube into the box"  # Replace with actual task description
                }
                f.write(json.dumps(task_data) + "\n")

        logger.info(f"Dataset finalized with {self.episode_count} episodes and {self.total_frames} frames")

    def _compute_video_stats(self, episode_index: int) -> Dict:
        """计算视频帧的 RGB 统计信息"""
        video_path = self.camera_dir / f"episode_{episode_index:06d}.mp4"
        if not video_path.exists():
            return None
            
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return None
            
        # 初始化累积值
        frame_count = 0
        rgb_sum = np.zeros(3)
        rgb_sq_sum = np.zeros(3)
        rgb_min = np.ones(3) * 255
        rgb_max = np.zeros(3)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # 转换 BGR 到 RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # 计算每个通道的统计值
            frame_float = frame.astype(np.float32)  # 归一化到 [0,1]
            rgb_sum += frame_float.mean(axis=(0,1))
            rgb_sq_sum += (frame_float**2).mean(axis=(0,1))
            rgb_min = np.minimum(rgb_min, frame_float.min(axis=(0,1)))
            rgb_max = np.maximum(rgb_max, frame_float.max(axis=(0,1)))
            frame_count += 1
            
        cap.release()
        
        if frame_count == 0:
            return None
            
        # 计算均值和标准差
        rgb_mean = rgb_sum / frame_count
        rgb_std = np.sqrt(rgb_sq_sum/frame_count - rgb_mean**2)
        
        # 格式化为所需的嵌套列表结构
        return {
            "min": [[[v]] for v in rgb_min],
            "max": [[[v]] for v in rgb_max],
            "mean": [[[v]] for v in rgb_mean],
            "std": [[[v]] for v in rgb_std],
            "count": [frame_count]
        }    
    def _compute_stats(self, data: np.ndarray) -> Dict:
        """计算数据的统计信息"""
        return {
            "min": np.min(data, axis=0).tolist(),
            "max": np.max(data, axis=0).tolist(),
            "mean": np.mean(data, axis=0).tolist(),
            "std": np.std(data, axis=0).tolist(),
            "count": [len(data)]
        }
    def __del__(self):
        """Cleanup resources"""
        if self.cap:
            self.cap.release()