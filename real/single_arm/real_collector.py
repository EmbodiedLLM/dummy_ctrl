import time
import numpy as np
from pathlib import Path
import logging
import cv2
import os
import shutil
import pandas as pd
from typing import Optional, Dict
import av

from .timing_utils import get_precise_timestamp, TimestampObsAccumulator, TimestampActionAccumulator
from .camera_utils import start_camera_thread, get_latest_frame
from queue import Queue

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
        
        # PyAV video writers for direct streaming
        self.video_writers = {} if use_video and camera_urls else None
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
        
        # High-precision timer
        self.timer_start_time = time.monotonic()
        logger.info(f"Initialized timer at {self.timer_start_time:.6f}")
        
        # Timestamp accumulators for synchronized data
        self.obs_accumulator = TimestampObsAccumulator(
            start_time=0.0,  # Relative to timer_start_time
            dt=1.0/fps
        )
        self.action_accumulator = TimestampActionAccumulator(
            start_time=0.0,  # Relative to timer_start_time
            dt=1.0/fps
        )

        # Temp directories for video encoding
        self.tmp_img_dirs = {
            "cam_wrist": self.output_dir / "tmp_images_wrist",
            "cam_head": self.output_dir / "tmp_images_head"
        }

        # Camera setup with threads
        self.camera_urls = camera_urls or {}
        self.camera_queues = {}
        self.camera_threads = {}
        
        # Keep old caps for compatibility
        self.caps = {"cam_wrist": None, "cam_head": None}
        
        if camera_urls:
            self.setup_cameras()
            self._setup_camera_threads()

    def _setup_camera_threads(self):
        """Setup camera capture threads"""
        for camera_name, camera_url in self.camera_urls.items():
            if camera_name not in ["cam_wrist", "cam_head"]:
                logger.warning(f"Unrecognized camera name: {camera_name}, skipping")
                continue
            
            logger.info(f"Setting up camera thread {camera_name}: {camera_url}")
            try:
                # Create queue and start thread
                frame_queue = Queue(maxsize=10)
                thread = start_camera_thread(camera_url, frame_queue, self.timer_start_time, target_fps=30)
                
                self.camera_queues[camera_name] = frame_queue
                self.camera_threads[camera_name] = thread
                
                logger.info(f"Camera thread {camera_name} started successfully")
                
            except Exception as e:
                logger.error(f"Error setting up camera thread {camera_name}: {e}")

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
            
        # Reset video writers for new episode
        if self.use_video and self.video_writers is not None:
            # Close any existing video writers
            for cam_name, writer in self.video_writers.items():
                if writer is not None:
                    container, stream = writer
                    # Flush remaining frames
                    for packet in stream.encode():
                        container.mux(packet)
                    container.close()
            # Reset writers dict
            self.video_writers = {cam_name: None for cam_name in self.camera_urls.keys()}
        
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
        # Use high-precision timestamp
        timestamp = get_precise_timestamp(self.timer_start_time)
        logger.debug(f"Frame {self.frame_count}: timestamp={timestamp:.6f}s")
        
        # Get latest frames from camera threads (non-blocking)
        frames = {}
        
        # Try camera threads first
        for cam_name, queue in self.camera_queues.items():
            frame_info = get_latest_frame(queue, max_age=0.1, timer_start_time=self.timer_start_time)
            if frame_info:
                frame_timestamp, frame = frame_info
                frames[cam_name] = frame
                logger.debug(f"Got frame from {cam_name} thread at {frame_timestamp:.6f}s")
            else:
                logger.warning(f"No recent frame from camera thread {cam_name}")
        
        # Fallback to old synchronous method if no threads
        if not frames:
            for cam_name, cap in self.caps.items():
                if cap and cap.isOpened():
                    try:
                        # First try to release any buffered frames
                        for _ in range(3):  # Try to clear some cache
                            cap.grab()
                        
                        # Then capture new frame
                        ret, frame = cap.read()
                        
                        if ret and frame is not None:
                            frames[cam_name] = frame
                            logger.debug(f"Got fallback frame from {cam_name}")
                    except Exception as e:
                        logger.error(f"Error in fallback capture for {cam_name}: {e}")
        
        # Write frames directly to video stream if using video
        if self.use_video and self.video_writers is not None:
            for cam_name, frame in frames.items():
                if frame is not None:
                    # Lazy initialization of video writers
                    if cam_name not in self.video_writers or self.video_writers[cam_name] is None:
                        video_path = self.camera_dirs[cam_name] / f"episode_{self.episode_count:06d}.mp4"
                        container = av.open(str(video_path), mode='w')
                        stream = container.add_stream('h264', rate=self.fps)
                        h, w = frame.shape[:2]
                        stream.width = w
                        stream.height = h
                        stream.pix_fmt = 'yuv420p'
                        stream.codec_context.options = {'crf': '18', 'profile': 'high'}
                        self.video_writers[cam_name] = (container, stream)
                        logger.info(f"Created video writer for {cam_name}: {video_path}")
                    
                    # Write frame directly to video stream
                    container, stream = self.video_writers[cam_name]
                    av_frame = av.VideoFrame.from_ndarray(frame, format='bgr24')
                    for packet in stream.encode(av_frame):
                        container.mux(packet)
        
        # Prepare observation data (robot state only for accumulator)
        obs_data = {
            'robot_state': np.array([follow + [float(follow_gripper)]])
        }
        
        # Prepare action data (teach commands)
        action_data = np.array([teach + [float(teach_gripper)]])
        
        # Push to separate accumulators
        self.obs_accumulator.put(obs_data, np.array([timestamp]))
        self.action_accumulator.put(action_data[None, :], np.array([timestamp]))
        
        # Store in old format for compatibility
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
                
        # Close video writers if enabled
        if self.use_video and self.video_writers is not None:
            for cam_name, writer in self.video_writers.items():
                if writer is not None:
                    container, stream = writer
                    try:
                        # Flush remaining frames
                        for packet in stream.encode():
                            container.mux(packet)
                        container.close()
                        logger.info(f"Successfully saved video for {cam_name}")
                    except Exception as e:
                        logger.error(f"Failed to close video writer for {cam_name}: {e}")
            # Reset writers
            self.video_writers = {cam_name: None for cam_name in self.camera_urls.keys()}
        
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
        
        # No temporary directories to cleanup with PyAV direct streaming
                
        self.episode_lengths.append(self.frame_count)        
        logger.info(f"Episode {next_index} saved with {self.frame_count} frames")
        self.frame_count = 0

    def __del__(self):
        """Cleanup resources"""
        # Stop camera threads
        for thread in self.camera_threads.values():
            if hasattr(thread, 'stop_flag'):
                thread.stop_flag['running'] = False
            thread.join(timeout=2.0)
        
        # Cleanup old caps
        for cap in self.caps.values():
            if cap:
                cap.release()
    
    def get_synchronized_data(self):
        """Get time-aligned data from both accumulators"""
        return {
            'observations': {
                'data': self.obs_accumulator.data,
                'timestamps': self.obs_accumulator.timestamps,
                'actual_timestamps': self.obs_accumulator.actual_timestamps,
                'length': len(self.obs_accumulator)
            },
            'actions': {
                'data': self.action_accumulator.actions,
                'timestamps': self.action_accumulator.timestamps,
                'actual_timestamps': self.action_accumulator.actual_timestamps,
                'length': len(self.action_accumulator)
            }
        }
    
    def get_latest_frames(self):
        """Get latest frames from all cameras for visualization"""
        frames = {}
        
        # Try camera threads first
        for cam_name, queue in self.camera_queues.items():
            frame_info = get_latest_frame(queue, max_age=0.1, timer_start_time=self.timer_start_time)
            if frame_info:
                frame_timestamp, frame = frame_info
                frames[cam_name] = frame
        
        # Fallback to synchronous capture if no thread frames
        if not frames:
            for cam_name, cap in self.caps.items():
                if cap and cap.isOpened():
                    try:
                        ret, frame = cap.read()
                        if ret and frame is not None:
                            frames[cam_name] = frame
                    except Exception as e:
                        logger.error(f"Error getting frame from {cam_name}: {e}")
        
        return frames