import time
import numpy as np
from pathlib import Path
import logging
import cv2
import os
import shutil
import pandas as pd
from typing import Optional, Dict
import json

from queue import Queue

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LeRobotDataCollector:
    def __init__(
        self, 
        output_dir: str = "/Users/jack/Desktop/dummy_ctrl/datasets/robot_data_lerobot", 
        fps: int = 10, 
        robot_type: str = "custom", 
        use_video: bool = True,
        task: str = "pick the cube into the box"  # Default task for PI0 models
    ):
        """Initialize data collector with camera support that follows LeRobot format
        
        Args:
            output_dir: Directory to save collected data
            fps: Frames per second for video recording
            camera_urls: Dictionary mapping camera names to URLs
            robot_type: Type of robot being used
            use_video: Whether to save video data
            task: Natural language task instruction for PI0 models. This is 
                 required by PI0/PI0FAST policies for inference.
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
        
        self.videos_dir = self.output_dir / "videos"
        self.camera_dirs = {}
        
        if use_video:
            self.videos_dir.mkdir(exist_ok=True)
            self.current_videos_chunk_dir = self.videos_dir / "chunk-000"
            self.current_videos_chunk_dir.mkdir(exist_ok=True)
            
            
        # Basic info
        self.fps = fps
        self.episode_count = 0
        self.total_frames = 0
        self.robot_type = robot_type
        self.use_video = use_video
        self.task = task  # Store the task instruction for PI0 models
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
            "task_index": [],
            "task": []  # Add task field for PI0 models
        }

        self.camera_frames = {
            "head": {
                "frames": [],
                "timestamps": []
            },
            "wrist": {
                "frames": [],
                "timestamps": []
            }
        }
        
        self.frame_count = 0
        self.start_time = None
        
        # High-precision timer
        self.timer_start_time = time.monotonic()
        logger.info(f"Initialized timer at {self.timer_start_time:.6f}")

    def set_task(self, task: str):
        """Set the current task instruction for PI0 models
        
        Args:
            task: Natural language task instruction
        """
        self.task = task
        logger.info(f"Set current task to: {task}")

    def start_episode(self, task: Optional[str] = None, start_time=None):
        """Start new episode recording
        
        Args:
            task: Optional task instruction for this episode. 
                 If provided, it will update the current task.
        """
        if task is not None:
            self.set_task(task)
        if start_time is not None:
            self.start_time = start_time
        else:
            self.start_time = time.time()
        self.camera_frames = {
            "head": {
                "frames": [],
                "timestamps": []
            },
            "wrist": {
                "frames": [],
                "timestamps": []
            }
        }
        self.frame_count = 0
        # Reset episode buffer
        for key in self.current_episode_data:
            self.current_episode_data[key] = []

    def collect_step(self, obs, action, timestamp=None, done=False):
        """Collect one timestep of data"""
        if timestamp is None:
            timestamp = time.time()
        if self.start_time is None:
            self.start_episode()

        if "camera_head" in obs:
            self.camera_frames["head"]["timestamps"].append(timestamp)
            self.camera_frames["head"]["frames"].append(obs["camera_head"])
        if "camera_wrist" in obs:
            self.camera_frames["wrist"]["timestamps"].append(timestamp)
            self.camera_frames["wrist"]["frames"].append(obs["camera_wrist"])

        obs_states_dict = {
            "joint_states": obs.get("joint_states", []),
            "gripper_pos_deg": obs.get("gripper_pos_deg", []),
            "gripper_torque": obs.get("gripper_torque", [])
        }
        self.current_episode_data["observation.state"].append(obs_states_dict)
        self.current_episode_data["action"].append(action)
        self.current_episode_data["timestamp"].append(timestamp)
        self.current_episode_data["next.done"].append(done)
        self.current_episode_data["episode_index"].append(self.episode_count)
        self.current_episode_data["frame_index"].append(self.frame_count)
        self.current_episode_data["index"].append(self.total_frames + self.frame_count)
        self.current_episode_data["task_index"].append(0)
        self.current_episode_data["task"].append(self.task)
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
                
        if self.use_video and len(self.camera_frames["head"]["frames"]) > 0:
            assert len(self.camera_frames["head"]["frames"]) == len(self.camera_frames["wrist"]["frames"]), "Head and wrist video frames count mismatch"
            self._save_simple_videos(next_index)
          
        # Store episode data in memory
        episode_data = {}
        for key, values in self.current_episode_data.items():
            episode_data[key] = np.array(values)
        self.all_episodes_data.append(episode_data)
        
        # Save parquet file with same index
        parquet_filename = f"episode_{next_index:06d}.parquet"
        parquet_path = self.current_chunk_dir / parquet_filename
        
        # Convert dictionary to pandas DataFrame and save as parquet
        processed_data = self._prepare_dataframe()
        df = pd.DataFrame(processed_data)
        df.to_parquet(str(parquet_path))
        logger.info(f"Parquet saved to {parquet_path}")
        
        # Update counters
        self.total_frames += self.frame_count
        self.episode_count += 1
        
                
        self.episode_lengths.append(self.frame_count)        
        logger.info(f"Episode {next_index} saved with {self.frame_count} frames")
        self.frame_count = 0
        # Update LeRobot v2.1 metadata
        self._update_episode_metadata(next_index)
        self._update_info_json()
    
    def _prepare_dataframe(self):
        """Prepare data for DataFrame conversion"""
        processed_data = {}

        for key, values in self.current_episode_data.items():
            if key == "observation.state":
                # 展开嵌套字典结构
                processed_data["observation.joint_states"] = [v["joint_states"] for v
    in values]
                processed_data["observation.gripper_pos"] = [np.array([v["gripper_pos_deg"]]) for v in values]
                processed_data["observation.gripper_torque"] = [np.array([v["gripper_torque"]]) for v in values]
            elif key == "action":
                # 修复：确保action是正确的格式，避免多余嵌套
                processed_data[key] = []
                for v in values:
                    if isinstance(v, dict):
                        # 如果action是字典格式
                        action_array = np.concatenate([v["joint_states"], [v["gripper_pos_deg"]]])
                        processed_data[key].append(action_array.tolist())
                    elif isinstance(v, (list, np.ndarray)):
                        # 如果action已经是列表或数组格式
                        processed_data[key].append(np.array(v).flatten().tolist())
                    else:
                        # 标量值转为单元素列表
                        processed_data[key].append([v])
            else:
                processed_data[key] = values

        return processed_data

    def _save_simple_videos(self, episode_index):
        """Simple video saving using OpenCV"""
        for cam_name, cam_data in self.camera_frames.items():
            if len(cam_data["frames"]) == 0:
                continue

            # 创建视频目录
            if cam_name == "head":
                video_dir = self.current_videos_chunk_dir / "observation.images.cam_head"
            elif cam_name == "wrist":
                video_dir = self.current_videos_chunk_dir / "observation.images.cam_wrist"
            else:
                continue

            video_dir.mkdir(parents=True, exist_ok=True)
            video_path = video_dir / f"episode_{episode_index:06d}.mp4"

            # 使用OpenCV保存视频
            first_frame = cam_data["frames"][0]
            height, width = first_frame.shape[:2]

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(str(video_path), fourcc, self.fps, (width,
    height))

            for frame in cam_data["frames"]:
                video_writer.write(frame)

            video_writer.release()
            logger.info(f"Saved {cam_name} video: {video_path}")

    def __del__(self):
        """Cleanup resources"""
        try:
            # 1. 生成最终的 info.json
            if hasattr(self, 'episode_count') and self.episode_count > 0:
                try:
                    self._update_info_json()
                    logger.info("Final info.json updated successfully")
                except Exception as e:
                    logger.error(f"Error updating final info.json: {e}")
            
            # 2. 清空内存中的帧数据 (释放大量内存)
            if hasattr(self, 'camera_frames'):
                for cam_data in self.camera_frames.values():
                    cam_data["frames"].clear()
                    cam_data["timestamps"].clear()
            
            if hasattr(self, 'all_episodes_data'):
                self.all_episodes_data.clear()
                
            if hasattr(self, 'current_episode_data'):
                for values in self.current_episode_data.values():
                    if isinstance(values, list):
                        values.clear()
            
            logger.info("LeRobotDataCollector cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during LeRobotDataCollector cleanup: {e}")
    
    def _update_episode_metadata(self, episode_idx):
        """Update episodes.jsonl and tasks.jsonl with episode info"""
        # Update episodes.jsonl
        episodes_file = self.meta_dir / "episodes.jsonl"
        episode_info = {
            "episode_index": episode_idx,
            "tasks": [self.task],
            "length": self.episode_lengths[-1]
        }
        
        with open(episodes_file, "a") as f:
            f.write(json.dumps(episode_info) + "\n")
        
        # Update tasks.jsonl
        tasks_file = self.meta_dir / "tasks.jsonl"
        task_info = {
            "task_index": 0,
            "task": self.task
        }
        
        # Only write if tasks.jsonl doesn't exist or is empty
        if not tasks_file.exists():
            with open(tasks_file, "w") as f:
                f.write(json.dumps(task_info) + "\n")
        
        # Update episodes_stats.jsonl
        self._update_episode_stats(episode_idx)
    
    def _update_episode_stats(self, episode_idx):
        """Calculate and update statistics for the episode"""
        episode_stats_file = self.meta_dir / "episodes_stats.jsonl"
        
        # Get episode data
        processed_data = self._prepare_dataframe()

        
        
        stats = {}
        stat_keys = ["observation.joint_states", "observation.gripper_pos", "observation.gripper_torque", "action"]
        for key in stat_keys:
            if key in processed_data and len(processed_data[key]) > 0:
                try:
                    # 转换为numpy数组进行统计
                    if key == "action":
                        # 修复：action数据处理，确保正确格式
                        arr = np.array(processed_data[key])
                        
                        # 计算统计信息
                        stats[key] = {
                            "mean": arr.mean(axis=0).tolist(),
                            "std": arr.std(axis=0).tolist(),
                            "min": arr.min(axis=0).tolist(),
                            "max": arr.max(axis=0).tolist(),
                            "count": [self.frame_count]
                        }
                        
                    elif key in ["observation.gripper_pos", "observation.gripper_torque"]:
                        # 修复：gripper数据处理，确保是1维数组格式
                        data_list = []
                        for item in processed_data[key]:
                            if hasattr(item, '__len__') and not isinstance(item, str):
                                data_list.append(item)
                            else:
                                data_list.append([item])
                        arr = np.array(data_list)
                        
                        stats[key] = {
                            "mean": arr.mean(axis=0).tolist(),
                            "std": arr.std(axis=0).tolist(), 
                            "min": arr.min(axis=0).tolist(),
                            "max": arr.max(axis=0).tolist(),
                            "count": [self.frame_count]
                        }
                        
                    else:
                        # joint_states等其他数据
                        arr = np.array(processed_data[key])
                        
                        stats[key] = {
                            "mean": arr.mean(axis=0).tolist(),
                            "std": arr.std(axis=0).tolist(),
                            "min": arr.min(axis=0).tolist(),
                            "max": arr.max(axis=0).tolist(),
                            "count": [self.frame_count]
                        }
                except Exception as e:
                    logger.warning(f"Error computing stats for {key}: {e}")
                    continue

        episode_stats = {
            "episode_index": episode_idx,
            "stats": stats
        }

        with open(episode_stats_file, "a") as f:
            f.write(json.dumps(episode_stats) + "\n")
    
    def _update_info_json(self):
        """Update info.json with current dataset information"""
        info_file = self.meta_dir / "info.json"
        
        # 动态确定维度
        joint_dim = 6  # 默认6轴机械臂
        action_dim = 7  # 6关节 + 1夹爪
        
        if self.all_episodes_data and len(self.all_episodes_data) > 0:
            processed_data = self._prepare_dataframe()
            
            # 从实际数据推断维度
            if "observation.joint_states" in processed_data and len(processed_data["observation.joint_states"]) > 0:
                joint_dim = len(processed_data["observation.joint_states"][0])
                
            if "action" in processed_data and len(processed_data["action"]) > 0:
                action_sample = processed_data["action"][0]
                action_dim = len(action_sample) if isinstance(action_sample, list) else 1
        
        # 摄像头特征 - 基于实际摄像头数据
        camera_features = {}
        if self.use_video:
            for cam_name in ["head", "wrist"]:
                if cam_name in self.camera_frames and len(self.camera_frames[cam_name]["frames"]) > 0:
                    # 获取实际图像尺寸
                    sample_frame = self.camera_frames[cam_name]["frames"][0]
                    height, width = sample_frame.shape[:2]
                    
                    camera_features[f"observation.images.cam_{cam_name}"] = {
                        "dtype": "video",
                        "shape": [height, width, 3],
                        "names": ["height", "width", "channel"],
                        "info": {
                            "video.fps": self.fps,
                            "video.codec": "mp4v"
                        }
                    }
        
        # 计算视频数量
        video_count = len([k for k in camera_features.keys() if "images" in k]) * self.episode_count
        
        info = {
            "codebase_version": "v2.1",
            "robot_type": self.robot_type,
            "fps": self.fps,
            "total_episodes": self.episode_count,
            "total_frames": self.total_frames,
            "total_tasks": 1,
            "total_videos": video_count,
            "chunks_size": 1000,  # 修复：添加必需的chunks_size字段
            "splits": {"train": f"0:{self.episode_count}"},
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
            "features": {
                # 分离的观测特征
                "observation.joint_states": {
                    "dtype": "float32",
                    "shape": [joint_dim],
                    "names": [f"joint_{i}" for i in range(joint_dim)]
                },
                "observation.gripper_pos": {
                    "dtype": "float32",
                    "shape": [1],  # 修复：确保是[1]而不是[]
                    "names": ["gripper_position_deg"]
                },
                "observation.gripper_torque": {
                    "dtype": "float32", 
                    "shape": [1],  # 修复：确保是[1]而不是[]
                    "names": ["gripper_torque"]
                },
                "action": {
                    "dtype": "float32",
                    "shape": [action_dim],
                    "names": [f"joint_{i}" for i in range(action_dim-1)] + ["gripper"]
                },
                "episode_index": {"dtype": "int64", "shape": []},
                "frame_index": {"dtype": "int64", "shape": []},
                "timestamp": {"dtype": "float64", "shape": []},
                "next.done": {"dtype": "bool", "shape": []},
                "index": {"dtype": "int64", "shape": []},
                "task_index": {"dtype": "int64", "shape": []},
                "task": {"dtype": "string", "shape": []},
                **camera_features
            }
        }
        
        with open(info_file, "w") as f:
            json.dump(info, f, indent=2)