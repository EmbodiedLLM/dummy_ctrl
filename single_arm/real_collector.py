import json
import time
import numpy as np
from pathlib import Path
from datetime import datetime
import logging
import cv2
import os
import shutil
import pandas as pd
# from safetensors.torch import save_file
from typing import Optional, Dict, List, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LeRobotDataCollector:
    def __init__(
        self, 
        output_dir: str = "/Users/jack/Desktop/dummy_ctrl/datasets/robot_data_lerobot", 
        fps: int = 10, 
        camera_urls: Optional[Dict[str, str]] = None,
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
            
            # Create directories for both cameras
            self.camera_dirs["cam_wrist"] = self.current_videos_chunk_dir / "observation.images.cam_wrist"
            self.camera_dirs["cam_wrist"].mkdir(exist_ok=True)
            self.camera_dirs["cam_head"] = self.current_videos_chunk_dir / "observation.images.cam_head"
            self.camera_dirs["cam_head"].mkdir(exist_ok=True)
            
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
        
        self.frame_count = 0
        self.start_time = None

        # Temp directories for video encoding
        self.tmp_img_dirs = {
            "cam_wrist": self.output_dir / "tmp_images_wrist",
            "cam_head": self.output_dir / "tmp_images_head"
        }

        # Camera setup
        self.camera_urls = camera_urls or {}
        self.caps = {"cam_wrist": None, "cam_head": None}
        
        if camera_urls:
            self.setup_cameras()

    def setup_cameras(self) -> bool:
        """Setup all camera captures"""
        success = True
        
        for camera_name, camera_url in self.camera_urls.items():
            if camera_name not in ["cam_wrist", "cam_head"]:
                logger.warning(f"Unrecognized camera name: {camera_name}, skipping")
                continue
                
            logger.info(f"Connecting to camera {camera_name}: {camera_url}")
            self.caps[camera_name] = cv2.VideoCapture(camera_url)
            
            if not self.caps[camera_name].isOpened():
                logger.error(f"Failed to connect to camera {camera_name}")
                success = False
            else:
                # Set camera properties for better performance
                self.caps[camera_name].set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffer size
                self.caps[camera_name].set(cv2.CAP_PROP_FPS, self.fps)
                logger.info(f"Camera {camera_name} connected successfully")
                
        return success
        
    def set_task(self, task: str):
        """Set the current task instruction for PI0 models
        
        Args:
            task: Natural language task instruction
        """
        self.task = task
        logger.info(f"Set current task to: {task}")
        
    def start_episode(self, task: Optional[str] = None):
        """Start new episode recording
        
        Args:
            task: Optional task instruction for this episode. 
                 If provided, it will update the current task.
        """
        if task is not None:
            self.set_task(task)
            
        self.start_time = time.time()
        self.frame_count = 0
        
        # Reset episode buffer
        for key in self.current_episode_data:
            self.current_episode_data[key] = []
            
        # Create clean temp image directories
        if self.use_video:
            for cam_name, tmp_dir in self.tmp_img_dirs.items():
                if tmp_dir.exists():
                    shutil.rmtree(tmp_dir)
                tmp_dir.mkdir(parents=True, exist_ok=True)
        
        # Reset and clear camera buffers completely
        if self.camera_urls:
            # Release old camera instances
            for cam_name, cap in self.caps.items():
                if cap:
                    cap.release()
                    self.caps[cam_name] = None
            
            time.sleep(0.1)  # Give cameras time to properly close
            
            # Setup new cameras
            self.setup_cameras()
            
            for cam_name, cap in self.caps.items():
                if cap and cap.isOpened():
                    # Clear buffer thoroughly
                    t_start = time.time()
                    while time.time() - t_start < 1.0:  # Clear buffer for 1 second
                        cap.grab()  # Just grab frames without decoding
                    
                    # Read a few more frames to ensure clean start
                    for _ in range(10):
                        cap.read()
        
        logger.info(f"Started episode {self.episode_count}")
        
    def collect_step(self, teach, follow, teach_gripper, follow_gripper):
        """Collect one timestep of data"""
        if self.start_time is None:
            self.start_episode()
        rate = 1 / self.fps
        # Use fixed timestamp increment of rate seconds
        timestamp = np.float64(self.frame_count * rate)
        
        # Capture frames from both cameras with improved reliability
        frames = {}
        for cam_name, cap in self.caps.items():
            if cap and cap.isOpened():
                try:
                    # First try to release any buffered frames
                    for _ in range(3):  # Try to clear some cache
                        cap.grab()
                    
                    # Then capture new frame
                    ret, frame = cap.read()
                    if ret and frame is not None and self.use_video:
                        img_path = self.tmp_img_dirs[cam_name] / f"frame_{self.frame_count:05d}.png"
                        
                        # Ensure save directory exists
                        if not img_path.parent.exists():
                            img_path.parent.mkdir(parents=True, exist_ok=True)
                            
                        # Save image with highest quality
                        success = cv2.imwrite(str(img_path), frame, [cv2.IMWRITE_PNG_COMPRESSION, 0])
                        
                        if success:
                            frames[cam_name] = frame
                            
                            # Force file system sync periodically
                            if self.frame_count % 30 == 0:
                                try:
                                    os.fsync(os.open(str(img_path), os.O_RDONLY))
                                except Exception as e:
                                    logger.debug(f"Failed to sync file system: {e}")
                        else:
                            logger.error(f"Failed to save image for {cam_name}: {img_path}")
                    else:
                        logger.warning(f"Camera '{cam_name}' failed to capture valid frame")
                        
                        # Try to reinitialize camera
                        if not ret:
                            attempts = 0
                            while attempts < 3 and not ret:
                                logger.info(f"Attempting to reinitialize camera {cam_name} (attempt {attempts+1})")
                                cap.release()
                                time.sleep(0.5)
                                self.setup_cameras()
                                if self.caps[cam_name] and self.caps[cam_name].isOpened():
                                    ret, frame = self.caps[cam_name].read()
                                    if ret and frame is not None:
                                        logger.info(f"Camera {cam_name} reinitialization successful")
                                        frames[cam_name] = frame
                                        break
                                attempts += 1
                except Exception as e:
                    logger.error(f"Error capturing frame from {cam_name}: {e}")
                    # Try to recover camera connection
                    try:
                        cap.release()
                        time.sleep(1)
                        self.setup_cameras()
                        logger.info(f"Attempted to recover camera {cam_name} connection")
                    except Exception as recovery_e:
                        logger.error(f"Failed to recover camera connection: {recovery_e}")
        
        # Store state and action data
        self.current_episode_data["observation.state"].append(
            np.concatenate([follow, [float(follow_gripper)]]).tolist()
        )
        self.current_episode_data["action"].append(
            np.concatenate([teach, [float(teach_gripper)]]).tolist()
        )
        self.current_episode_data["episode_index"].append(self.episode_count)
        self.current_episode_data["frame_index"].append(self.frame_count)
        self.current_episode_data["timestamp"].append(timestamp)
        self.current_episode_data["next.done"].append(False)
        self.current_episode_data["index"].append(self.total_frames + self.frame_count)
        self.current_episode_data["task_index"].append(0)
        # Add task instruction for PI0 compatibility
        self.current_episode_data["task"].append(self.task)
        
        self.frame_count += 1
        
        # Force periodic frame saving
        if self.frame_count % 100 == 0:  # Save every 100 frames
            try:
                os.sync()  # Lightweight sync
                logger.debug(f"Synced {self.frame_count} frames to disk")
            except Exception as e:
                logger.error(f"Failed to sync frames to disk: {e}")

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
                
        # Save videos if enabled
        if self.use_video:
            for cam_name, cam_dir in self.camera_dirs.items():
                if cam_name in self.caps and self.caps[cam_name]:
                    # Check for existing videos to maintain consistent indexing
                    video_filename = f"episode_{next_index:06d}.mp4"
                    video_path = cam_dir / video_filename
                    
                    tmp_dir = self.tmp_img_dirs[cam_name]
                    if tmp_dir.exists() and list(tmp_dir.glob("*.png")):
                        # First try using OpenCV directly
                        success = False
                        try:
                            frame_files = sorted(tmp_dir.glob("frame_*.png"))
                            if frame_files:
                                # Read first frame to get dimensions
                                test_img = cv2.imread(str(frame_files[0]))
                                if test_img is not None:
                                    height, width = test_img.shape[:2]
                                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                                    video_writer = cv2.VideoWriter(str(video_path), fourcc, self.fps, (width, height))
                                    
                                    if video_writer.isOpened():
                                        for frame_file in frame_files:
                                            img = cv2.imread(str(frame_file))
                                            if img is not None:
                                                video_writer.write(img)
                                        
                                        video_writer.release()
                                        logger.info(f"Successfully saved video using OpenCV for {cam_name}: {video_path}")
                                        success = True
                                    else:
                                        logger.error(f"OpenCV video writer creation failed for {cam_name}: {video_path}")
                        except Exception as e:
                            logger.error(f"Failed to save video using OpenCV for {cam_name}: {e}")
                        
                        # If OpenCV fails, try ffmpeg
                        if not success:
                            cmd = f"ffmpeg -y -framerate {self.fps} -i {tmp_dir}/frame_%05d.png "\
                                f"-c:v libx264 -pix_fmt yuv420p -crf 23 {str(video_path)}"
                            result = os.system(cmd)
                            if result != 0:
                                logger.error(f"ffmpeg command failed for {cam_name}, return code: {result}")
                                # Backup frames as individual files
                                backup_dir = cam_dir / f"episode_{next_index:06d}_frames"
                                try:
                                    backup_dir.mkdir(parents=True, exist_ok=True)
                                    for frame_file in frame_files:
                                        dst_file = backup_dir / frame_file.name
                                        shutil.copy2(frame_file, dst_file)
                                    logger.info(f"Successfully backed up frames for {cam_name} to {backup_dir}")
                                except Exception as e:
                                    logger.error(f"Failed to backup frames for {cam_name}: {e}")
        
        # Store episode data in memory
        episode_data = {}
        for key, values in self.current_episode_data.items():
            episode_data[key] = np.array(values)
        self.all_episodes_data.append(episode_data)
        
        # Save parquet file with same index
        parquet_filename = f"episode_{next_index:06d}.parquet"
        parquet_path = self.current_chunk_dir / parquet_filename
        
        # Convert dictionary to pandas DataFrame and save as parquet
        df = pd.DataFrame(self.current_episode_data)
        df.to_parquet(str(parquet_path))
        logger.info(f"Parquet saved to {parquet_path}")
        
        # Update counters
        self.total_frames += self.frame_count
        self.episode_count += 1
        
        # Cleanup
        for tmp_dir in self.tmp_img_dirs.values():
            if tmp_dir.exists():
                try:
                    time.sleep(0.5)  # Wait to ensure all file operations are complete
                    shutil.rmtree(tmp_dir)
                except Exception as e:
                    logger.error(f"Failed to clean up temporary directory: {e}")
                
        self.episode_lengths.append(self.frame_count)        
        logger.info(f"Episode {next_index} saved with {self.frame_count} frames")
        self.frame_count = 0

    def __del__(self):
        """Cleanup resources"""
        for cap in self.caps.values():
            if cap:
                cap.release()