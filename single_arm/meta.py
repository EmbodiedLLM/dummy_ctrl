import json
import numpy as np
import cv2
from pathlib import Path
import logging
from datasets import load_dataset
import pandas as pd
from typing import Dict, List, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# python meta.py \
#     --data-dir /path/to/robot_data_lerobot \
#     --robot-type custom \
#     --fps 10
def compute_video_stats(video_path: Path) -> Dict:
    """Compute RGB statistics from video file"""
    if not video_path.exists():
        return None
        
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
        
    frame_count = 0
    rgb_sum = np.zeros(3)
    rgb_sq_sum = np.zeros(3)
    rgb_min = np.ones(3) * 255
    rgb_max = np.zeros(3)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_sum += frame.mean(axis=(0,1))
        rgb_sq_sum += (frame.astype(np.float32)**2).mean(axis=(0,1))
        rgb_min = np.minimum(rgb_min, frame.min(axis=(0,1)))
        rgb_max = np.maximum(rgb_max, frame.max(axis=(0,1)))
        frame_count += 1
        
    cap.release()
    
    if frame_count == 0:
        return None
        
    rgb_mean = rgb_sum / frame_count
    rgb_std = np.sqrt(rgb_sq_sum/frame_count - rgb_mean**2)
    
    return {
        "min": [[[float(v)]] for v in rgb_min],
        "max": [[[float(v)]] for v in rgb_max],
        "mean": [[[float(v)]] for v in rgb_mean],
        "std": [[[float(v)]] for v in rgb_std],
        "count": [frame_count]
    }

def compute_array_stats(data: np.ndarray) -> Dict:
    """Compute statistics for numpy array data"""
    return {
        "min": np.min(data, axis=0).tolist(),
        "max": np.max(data, axis=0).tolist(),
        "mean": np.mean(data, axis=0).tolist(),
        "std": np.std(data, axis=0).tolist(),
        "count": [len(data)]
    }

def generate_metadata(
    data_dir: Path,
    robot_type: str = "custom",
    fps: int = 50,
    use_video: bool = True
):
    """Generate metadata files from existing dataset files
    
    This function scans through robot data (parquet files and videos),
    extracts metadata including task information, and generates the necessary
    metadata files for LeRobot format.
    
    Task data is read from the "task" column in each parquet file. Unique tasks
    are identified across all episodes, and task_index values are assigned.
    
    Args:
        data_dir: Path to the dataset directory
        robot_type: Type of robot used for data collection
        fps: Frames per second of the videos
        use_video: Whether the dataset includes video files
    """
    data_dir = Path(data_dir)
    meta_dir = data_dir / "meta"
    meta_dir.mkdir(exist_ok=True)
    
    # Find all parquet files
    chunk_dir = data_dir / "data/chunk-000"
    parquet_files = sorted(chunk_dir.glob("episode_*.parquet"))
    total_episodes = len(parquet_files)
    
    # Calculate total frames and collect episode lengths
    total_frames = 0
    episode_lengths = []
    all_episodes_data = []
    all_episodes_tasks = []  # List to store tasks for each episode
    unique_tasks = set()     # Set to track unique tasks across all episodes
    
    for parquet_file in parquet_files:
        dataset = load_dataset("parquet", data_files=str(parquet_file))["train"]
        length = len(dataset)
        episode_lengths.append(length)
        total_frames += length
        all_episodes_data.append(dataset)
        
        # Extract task from this episode's data using pandas
        try:
            # Read the parquet file directly with pandas to check for task column
            df = pd.read_parquet(parquet_file)
            if "task" in df.columns:
                # Use the first task in the episode (assuming task doesn't change within episode)
                episode_task = df["task"].iloc[0]
                all_episodes_tasks.append(episode_task)
                unique_tasks.add(episode_task)
                logger.info(f"Found task '{episode_task}' in {parquet_file}")
            else:
                # Default task if not found
                default_task = "pick the cube into the box"
                all_episodes_tasks.append(default_task)
                unique_tasks.add(default_task)
                logger.warning(f"No task column found in {parquet_file}, using default: '{default_task}'")
        except Exception as e:
            # Default task if there was an error
            default_task = "pick the cube into the box"
            all_episodes_tasks.append(default_task)
            unique_tasks.add(default_task)
            logger.error(f"Error reading task from {parquet_file}: {e}, using default: '{default_task}'")
    
    # Convert unique tasks to a list and create mapping
    unique_tasks_list = list(unique_tasks)
    task_to_index = {task: idx for idx, task in enumerate(unique_tasks_list)}
    
    # 1. Generate info.json
    info = {
        "codebase_version": "v2.1",
        "robot_type": robot_type,
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "total_tasks": len(unique_tasks_list),  # Update total tasks count
        "total_videos": total_episodes if use_video else 0,
        "total_chunks": 1,
        "chunks_size": 1000,
        "fps": fps,
        "splits": {
            "train": f"{round(total_episodes*0.9,0)}:{round(total_episodes*0.1,0)}"
        },
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": {
            "observation.images.cam_wrist": {
                "dtype": "video",
                "shape": [720, 1280, 3],
                "names": ["height", "width", "channel"],
                "video_info": {
                    "video.fps": float(fps),
                    "video.codec": "av1",
                    "video.pix_fmt": "yuv420p",
                    "video.is_depth_map": False,
                    "has_audio": False
                }
            },
            "observation.images.cam_head": {
                "dtype": "video",
                "shape": [720, 1280, 3],
                "names": ["height", "width", "channel"],
                "video_info": {
                    "video.fps": float(fps),
                    "video.codec": "av1",
                    "video.pix_fmt": "yuv420p",
                    "video.is_depth_map": False,
                    "has_audio": False
                }
            },
            "observation.state": {
                "dtype": "float32",
                "shape": [7],
                "names": {"motors": [f"joint_{i+1}" for i in range(6)] + ["gripper"]}
            },
            "action": {
                "dtype": "float32",
                "shape": [7],
                "names": {"motors": [f"joint_{i+1}" for i in range(6)] + ["gripper"]}
            },
            "episode_index": {"dtype": "int64", "shape": [1], "names": None},
            "frame_index": {"dtype": "int64", "shape": [1], "names": None},
            "timestamp": {"dtype": "float32", "shape": [1], "names": None},
            "next.done": {"dtype": "bool", "shape": [1], "names": None},
            "index": {"dtype": "int64", "shape": [1], "names": None},
            "task_index": {"dtype": "int64", "shape": [1], "names": None},
            "task": {"dtype": "string", "shape": [1], "names": None}  # Add task feature
        },
        "task_information": {
            "task_count": len(unique_tasks_list),
            "tasks": list(unique_tasks)
        }
    }
    
    with open(meta_dir / "info.json", 'w') as f:
        json.dump(info, f, indent=4)
    
    # 2. Generate episodes.jsonl
    with open(meta_dir / "episodes.jsonl", 'w') as f:
        for i in range(total_episodes):
            episode_data = {
                "episode_index": i,
                "tasks": [all_episodes_tasks[i]],  # Use the actual task for this episode
                "length": episode_lengths[i],
                "task_index": task_to_index[all_episodes_tasks[i]]  # Add task index
            }
            f.write(json.dumps(episode_data) + "\n")
            
    # Validate task_index consistency between parquet files and task_to_index mapping
    logger.info("Validating task_index consistency...")
    for i, parquet_file in enumerate(parquet_files):
        try:
            df = pd.read_parquet(parquet_file)
            if "task_index" in df.columns and "task" in df.columns:
                # Check the first row's task_index value
                file_task_index = df["task_index"].iloc[0]
                file_task = df["task"].iloc[0]
                expected_task_index = task_to_index[file_task]
                
                if file_task_index != expected_task_index:
                    logger.warning(
                        f"Task index mismatch in {parquet_file.name}: "
                        f"File has task_index={file_task_index} for task '{file_task}', "
                        f"but mapping gives task_index={expected_task_index}"
                    )
        except Exception as e:
            logger.error(f"Error validating task_index in {parquet_file}: {e}")
    
    # 3. Generate episodes_stats.jsonl
    with open(meta_dir / "episodes_stats.jsonl", 'w') as f:
        for i, dataset in enumerate(all_episodes_data):
            obs_stats = compute_array_stats(np.array(dataset["observation.state"]))
            action_stats = compute_array_stats(np.array(dataset["action"]))
            
            stats = {
                "episode_index": i,
                "stats": {
                    "observation.state": obs_stats,
                    "action": action_stats
                }
            }
            
            if use_video:
                # Add wrist camera statistics
                wrist_video_path = data_dir / f"videos/chunk-000/observation.images.cam_wrist/episode_{i:06d}.mp4"
                wrist_video_stats = compute_video_stats(wrist_video_path)
                if wrist_video_stats:
                    stats["stats"]["observation.images.cam_wrist"] = wrist_video_stats
                
                # Add head camera statistics
                head_video_path = data_dir / f"videos/chunk-000/observation.images.cam_head/episode_{i:06d}.mp4"
                head_video_stats = compute_video_stats(head_video_path)
                if head_video_stats:
                    stats["stats"]["observation.images.cam_head"] = head_video_stats
            
            f.write(json.dumps(stats) + "\n")
    
    # 4. Generate tasks.jsonl
    with open(meta_dir / "tasks.jsonl", 'w') as f:
        for task_index, task in enumerate(unique_tasks_list):
            task_data = {
                "task_index": task_index,
                "task": task
            }
            f.write(json.dumps(task_data) + "\n")
    
    logger.info(f"Generated metadata for {total_episodes} episodes with {len(unique_tasks_list)} unique tasks and {total_frames} frames")
    logger.info(f"Unique tasks found: {', '.join(unique_tasks_list)}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True, help="Path to dataset directory")
    parser.add_argument("--robot-type", type=str, default="custom", help="Robot type")
    parser.add_argument("--fps", type=int, default=10, help="Video FPS")
    parser.add_argument("--no-video", action="store_false", dest="use_video", help="Dataset has no videos")
    
    args = parser.parse_args()
    generate_metadata(
        data_dir=args.data_dir,
        robot_type=args.robot_type,
        fps=args.fps,
        use_video=args.use_video
    )