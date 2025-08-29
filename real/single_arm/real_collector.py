import time
import numpy as np
from pathlib import Path
import logging
import cv2
import os
import shutil
import pandas as pd
from typing import Optional, Dict, List, Tuple
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
        
        # 修改：使用LeRobot兼容的数据结构
        self.current_episode_data = {
            "observation.state": [],  # 合并joint_states + gripper_pos
            "gripper_torque": [],     # 重命名，不再是observation.gripper_torque
            "action": [],
            "episode_index": [],
            "frame_index": [],
            "timestamp": [],
            "next.done": [],
            "index": [],
            "task_index": [],
            "task": [],  # Add task field for PI0 models
            "clock_time": []  # Add clock_time field for original timestamps
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
        self.episode_start_time = None  # Track episode start time for relative timestamps
        
        # High-precision timer
        self.timer_start_time = time.monotonic()
        logger.info(f"Initialized timer at {self.timer_start_time:.6f}")

    # ============================================================================
    # LeRobot相机统计算法 - 从lerobot_camera_stats.py复制
    # ============================================================================
    
    def estimate_num_samples(self, dataset_len: int, min_num_samples: int = 100, max_num_samples: int = 10_000, power: float = 0.75) -> int:
        """启发式估计基于数据集大小的样本数量"""
        if dataset_len < min_num_samples:
            min_num_samples = dataset_len
        return max(min_num_samples, min(int(dataset_len**power), max_num_samples))

    def sample_indices(self, data_len: int) -> List[int]:
        """生成均匀分布的采样索引"""
        num_samples = self.estimate_num_samples(data_len)
        return np.round(np.linspace(0, data_len - 1, num_samples)).astype(int).tolist()

    def auto_downsample_height_width(self, img: np.ndarray, target_size: int = 150, max_size_threshold: int = 300):
        """自动降采样图像的高度和宽度"""
        if len(img.shape) == 3:
            _, height, width = img.shape
        else:
            height, width = img.shape[:2]

        if max(width, height) < max_size_threshold:
            return img

        downsample_factor = int(width / target_size) if width > height else int(height / target_size)
        if len(img.shape) == 3:
            return img[:, ::downsample_factor, ::downsample_factor]
        else:
            return img[::downsample_factor, ::downsample_factor]

    def sample_camera_frames(self, frames: List[np.ndarray]) -> np.ndarray:
        """使用LeRobot采样策略处理相机帧"""
        if not frames:
            return np.array([])
        
        # 使用LeRobot的采样策略
        sampled_indices = self.sample_indices(len(frames))
        
        images = None
        for i, frame_idx in enumerate(sampled_indices):
            if frame_idx >= len(frames):
                continue
                
            frame = frames[frame_idx]
            
            # OpenCV格式BGR转RGB，然后转为channel_first格式
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                rgb_frame = frame
            
            # 转为uint8格式并channel_first
            if len(rgb_frame.shape) == 3:
                img = np.transpose(rgb_frame, (2, 0, 1)).astype(np.uint8)  # (H,W,C) -> (C,H,W)
            else:
                img = rgb_frame.astype(np.uint8)
            
            img = self.auto_downsample_height_width(img)
            
            if images is None:
                images = np.empty((len(sampled_indices), *img.shape), dtype=np.uint8)
            
            if i < len(sampled_indices):
                images[i] = img
        
        return images

    def get_feature_stats(self, array: np.ndarray, axis: tuple, keepdims: bool) -> Dict[str, np.ndarray]:
        """计算特征统计信息，与LeRobot完全一致"""
        stats = {
            "min": array.min(axis=axis, keepdims=keepdims),
            "max": array.max(axis=axis, keepdims=keepdims),
            "mean": array.mean(axis=axis, keepdims=keepdims),
            "std": array.std(axis=axis, keepdims=keepdims),
            "count": array.shape[0] if len(axis) == 1 and axis[0] == 0 else np.prod([array.shape[i] for i in axis])
        }
        return stats

    def compute_camera_stats_lerobot(self, frames: List[np.ndarray]) -> Optional[Dict]:
        """使用LeRobot算法计算相机统计信息"""
        try:
            # 使用LeRobot的采样策略获取帧
            images = self.sample_camera_frames(frames)
            
            if images.size == 0:
                return None
            
            # 计算统计信息 - 对图像数据，统计轴为(0, 2, 3)，保留通道维度
            # images shape: [num_samples, C, H, W]
            stats = self.get_feature_stats(images, axis=(0, 2, 3), keepdims=True)
            
            # LeRobot要求图像统计信息shape为(3,1,1)，确保正确reshape
            min_val = stats["min"] / 255.0
            max_val = stats["max"] / 255.0  
            mean_val = stats["mean"] / 255.0
            std_val = stats["std"] / 255.0
            
            # 确保shape为(3,1,1) - 无论原始shape如何，都强制reshape为正确格式
            min_val = min_val.squeeze().reshape(3, 1, 1)
            max_val = max_val.squeeze().reshape(3, 1, 1)
            mean_val = mean_val.squeeze().reshape(3, 1, 1)
            std_val = std_val.squeeze().reshape(3, 1, 1)
            
            # count也需要是正确的格式
            count_val = stats["count"]
            if np.isscalar(count_val):
                count_val = np.array([count_val])
            
            # 转换为Python标准类型并归一化到[0,1]范围（LeRobot的标准做法）
            normalized_stats = {
                "min": min_val.tolist(),
                "max": max_val.tolist(), 
                "mean": mean_val.tolist(),
                "std": std_val.tolist(),
                "count": count_val.tolist() if hasattr(count_val, 'tolist') else [int(count_val)]
            }
            
            return normalized_stats
            
        except Exception as e:
            logger.warning(f"Error computing camera stats with LeRobot algorithm: {e}")
            return None

    # ============================================================================
    # 修改的核心方法
    # ============================================================================

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
            self.episode_start_time = start_time
        else:
            self.start_time = time.time()
            self.episode_start_time = self.start_time
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
        # Reset episode start time
        self.episode_start_time = None

    def collect_step(self, obs, action, timestamp=None, clock_time=None, done=False):
        """Collect one timestep of data
        
        Args:
            obs: Observation data
            action: Action data  
            timestamp: Relative timestamp (from episode start). If None, will be calculated from clock_time
            clock_time: Absolute clock time. If None, will use current time.time()
            done: Whether episode is done
        """
        if clock_time is None:
            clock_time = time.time()
        if self.start_time is None:
            self.start_episode()
        if self.episode_start_time is None:
            self.episode_start_time = clock_time
        
        # Calculate relative timestamp if not provided
        if timestamp is None:
            timestamp = clock_time - self.episode_start_time

        if "cam_head" in obs:
            self.camera_frames["head"]["timestamps"].append(timestamp)
            self.camera_frames["head"]["frames"].append(obs["cam_head"])
        if "cam_wrist" in obs:
            self.camera_frames["wrist"]["timestamps"].append(timestamp)
            self.camera_frames["wrist"]["frames"].append(obs["cam_wrist"])

        # 修改：合并joint_states和gripper_pos为observation.state
        joint_states = obs.get("joint_states", [])
        gripper_pos_deg = obs.get("gripper_pos_deg", 0.0)
        
        # 合并为7维的observation.state (6关节 + 1夹爪位置)
        if isinstance(joint_states, (list, np.ndarray)) and len(joint_states) >= 6:
            observation_state = np.concatenate([joint_states[:6], [gripper_pos_deg]])
        else:
            # 默认6个关节都为0
            observation_state = np.concatenate([np.zeros(6), [gripper_pos_deg]])
            
        self.current_episode_data["observation.state"].append(observation_state.tolist())
        
        # 修改：gripper_torque作为单独字段，不再是observation.gripper_torque
        gripper_torque = obs.get("gripper_torque", 0.0)
        self.current_episode_data["gripper_torque"].append([gripper_torque])  # 保持为列表格式
        self.current_episode_data["action"].append(action)
        self.current_episode_data["timestamp"].append(timestamp)  # Use relative timestamp
        self.current_episode_data["clock_time"].append(clock_time)  # Store original clock time
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
            
        # 修改：确保最后一帧的next.done为True
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
            # 1. 最终化数据集（生成全局stats.json和更新info.json）
            if hasattr(self, 'episode_count') and self.episode_count > 0:
                try:
                    self.finalize_dataset()
                    logger.info("Dataset finalized successfully in cleanup")
                except Exception as e:
                    logger.error(f"Error finalizing dataset in cleanup: {e}")
            
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
    
    # ============================================================================
    # 新增：数据集后处理方法
    # ============================================================================
    
    def generate_global_stats(self):
        """生成全局stats.json文件，类似于fix_dataset_features.py中的逻辑"""
        try:
            from pathlib import Path
            stats_path = self.meta_dir / "stats.json"
            episodes_stats_path = self.meta_dir / "episodes_stats.jsonl"
            
            if not episodes_stats_path.exists():
                logger.warning("episodes_stats.jsonl not found, cannot generate global stats")
                return
            
            # 读取所有episode统计信息
            episodes_stats = []
            with open(episodes_stats_path, 'r') as f:
                for line in f:
                    episodes_stats.append(json.loads(line))
            
            # 提取所有episode的stats
            episode_stats_list = []
            for episode_stat in episodes_stats:
                if "stats" in episode_stat and episode_stat["stats"]:
                    episode_stats_list.append(episode_stat["stats"])
            
            if not episode_stats_list:
                logger.warning("No episode stats found for aggregation")
                return
            
            # 简化的聚合逻辑
            global_stats = {}
            for key in episode_stats_list[0].keys():
                try:
                    # 收集所有episode中该key的统计信息
                    key_stats = [ep_stats[key] for ep_stats in episode_stats_list if key in ep_stats]
                    
                    if not key_stats:
                        continue
                    
                    # 计算聚合统计信息
                    all_means = np.array([s["mean"] for s in key_stats])
                    all_mins = np.array([s["min"] for s in key_stats]) 
                    all_maxs = np.array([s["max"] for s in key_stats])
                    all_counts = np.array([s["count"][0] if isinstance(s["count"], list) else s["count"] for s in key_stats])
                    
                    # 计算加权平均
                    total_count = all_counts.sum()
                    if total_count > 0:
                        weighted_mean = np.average(all_means, weights=all_counts, axis=0)
                    else:
                        weighted_mean = np.mean(all_means, axis=0)
                    
                    global_stats[key] = {
                        "min": np.min(all_mins, axis=0).tolist(),
                        "max": np.max(all_maxs, axis=0).tolist(),
                        "mean": weighted_mean.tolist(),
                        "std": np.std(all_means, axis=0).tolist(),  # 简化的std计算
                        "count": [int(total_count)]
                    }
                    
                except Exception as e:
                    logger.warning(f"Error aggregating stats for {key}: {e}")
                    continue
            
            # 写入全局stats文件
            with open(stats_path, 'w') as f:
                json.dump(global_stats, f, indent=2)
            
            logger.info(f"Generated global stats.json with {len(global_stats)} features")
            
        except Exception as e:
            logger.error(f"Error generating global stats: {e}")

    def finalize_dataset(self):
        """完成数据集收集后的最终处理"""
        logger.info("Finalizing dataset...")
        
        # 生成全局stats.json
        self.generate_global_stats()
        
        # 最终更新info.json
        self._update_info_json()
        
        logger.info("Dataset finalization complete!")
        logger.info(f"Dataset ready for LeRobot at: {self.output_dir}")
        logger.info("Features:")
        logger.info("- observation.state (7D): 6 joints + gripper position")
        logger.info("- gripper_torque (1D): gripper torque sensor")  
        logger.info("- action (7D): 6 joints + gripper action")
        logger.info("- Camera stats: LeRobot-compatible format with sampling")
    
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
        """Calculate and update statistics for the episode - 修改为LeRobot兼容格式"""
        episode_stats_file = self.meta_dir / "episodes_stats.jsonl"
        
        # Get episode data
        processed_data = self._prepare_dataframe()
        
        stats = {}
        stat_keys = ["observation.state", "gripper_torque", "action"]
        
        # 计算状态和动作数据的统计信息
        for key in stat_keys:
            if key in processed_data and len(processed_data[key]) > 0:
                try:
                    # 转换为numpy数组进行统计
                    arr = np.array(processed_data[key])
                    
                    # 计算统计信息
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

        # 计算相机数据的统计信息 - 使用LeRobot算法
        if self.use_video:
            self._compute_camera_stats_lerobot(stats, episode_idx)

        episode_stats = {
            "episode_index": episode_idx,
            "stats": stats
        }

        with open(episode_stats_file, "a") as f:
            f.write(json.dumps(episode_stats) + "\n")
    
    def _compute_camera_stats_lerobot(self, stats, episode_idx):
        """计算相机数据的统计信息 - 使用LeRobot算法"""
        camera_keys = ["observation.images.cam_head", "observation.images.cam_wrist"]
        camera_mapping = {
            "observation.images.cam_head": "head",
            "observation.images.cam_wrist": "wrist"  
        }
        
        for cam_key in camera_keys:
            cam_name = camera_mapping[cam_key]
            if cam_name in self.camera_frames and len(self.camera_frames[cam_name]["frames"]) > 0:
                try:
                    frames = self.camera_frames[cam_name]["frames"]
                    
                    # 使用LeRobot算法计算统计信息
                    camera_stats = self.compute_camera_stats_lerobot(frames)
                    
                    if camera_stats is not None:
                        stats[cam_key] = camera_stats
                        logger.info(f"Computed LeRobot-compatible stats for {cam_key}: mean={camera_stats['mean'][0][0][:3]}...")
                    
                except Exception as e:
                    logger.warning(f"Error computing camera stats for {cam_key}: {e}")
    
    def _update_info_json(self):
        """Update info.json with current dataset information - 修改为LeRobot格式"""
        info_file = self.meta_dir / "info.json"
        
        # 动态确定维度
        joint_dim = 6  # 默认6轴机械臂
        action_dim = 7  # 6关节 + 1夹爪
        
        if self.all_episodes_data and len(self.all_episodes_data) > 0:
            processed_data = self._prepare_dataframe()
            
            # 从实际数据推断维度
            if "observation.state" in processed_data and len(processed_data["observation.state"]) > 0:
                state_dim = len(processed_data["observation.state"][0])
                
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
            "chunks_size": 1000,
            "splits": {"train": f"0:{self.episode_count}"},
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
            "features": {
                # 修改：使用LeRobot兼容的特征定义
                "observation.state": {
                    "dtype": "float32",
                    "shape": [7],  # 6关节 + 1夹爪位置
                    "names": [
                        "joint_0", "joint_1", "joint_2", "joint_3", "joint_4", "joint_5", 
                        "gripper_position_deg"
                    ]
                },
                "gripper_torque": {
                    "dtype": "float32", 
                    "shape": [1],
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
                "clock_time": {"dtype": "float64", "shape": []},
                "next.done": {"dtype": "bool", "shape": []},
                "index": {"dtype": "int64", "shape": []},
                "task_index": {"dtype": "int64", "shape": []},
                "task": {"dtype": "string", "shape": []},
                **camera_features
            }
        }
        
        with open(info_file, "w") as f:
            json.dump(info, f, indent=2)