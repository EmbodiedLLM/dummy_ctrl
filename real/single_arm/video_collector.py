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
from typing import Optional, Dict, List, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VideoDataCollector:
    def __init__(
        self, 
        output_dir: str, 
        cam_url: str,
        camera_name: str = "cam_wrist",
        fps: int = 10,
        resolution: Tuple[int, int] = None,
        robot_type: str = "custom"
    ):
        """Initialize collector for video data only
        
        Args:
            output_dir: Directory to store video data
            cam_url: URL of the camera (e.g. http://ip:port/?action=stream or 0 for webcam)
            camera_name: Name of the camera (used in directory structure)
            fps: Frames per second for video recording
            resolution: Optional (width, height) tuple to resize camera frames (e.g. (320, 240))
            robot_type: Type of robot being controlled
        """
        self.output_dir = Path(output_dir)
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Ensuring output directory exists: {self.output_dir}")
        except Exception as e:
            logger.error(f"Failed to create output directory: {e}")
        
        # Create directory structure
        self.meta_dir = self.output_dir / "meta"
        self.meta_dir.mkdir(parents=True, exist_ok=True)
        self.videos_dir = self.output_dir / "videos"
        self.videos_dir.mkdir(parents=True, exist_ok=True)
        self.current_videos_chunk_dir = self.videos_dir / "chunk-000"
        self.current_videos_chunk_dir.mkdir(parents=True, exist_ok=True)
        self.camera_dir = self.current_videos_chunk_dir / f"observation.images.{camera_name}"
        try:
            self.camera_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Creating camera directory: {self.camera_dir}")
        except Exception as e:
            logger.error(f"Failed to create camera directory: {e}")
            
        # Basic info
        self.fps = fps
        self.episode_count = 0
        self.robot_type = robot_type
        self.episode_lengths = []
        self.camera_name = camera_name
        self.resolution = resolution  # Store target resolution
        
        self.frame_count = 0
        self.start_time = None

        # Temp directory for video encoding - Add camera_name to ensure each camera has independent temp directory
        self.tmp_img_dir = self.output_dir / f"tmp_images_{camera_name}"
        
        # Ensure temp directory exists
        try:
            if self.tmp_img_dir.exists():
                shutil.rmtree(self.tmp_img_dir)
            self.tmp_img_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Temporary image directory created: {self.tmp_img_dir}")
        except Exception as e:
            logger.error(f"Failed to create temporary image directory: {e}")

        # Camera setup
        self.cam_url = cam_url
        self.cap = None
        self.setup_camera()

    def setup_camera(self) -> bool:
        """Setup camera capture"""
        logger.info(f"Connecting to camera: {self.cam_url}")
        self.cap = cv2.VideoCapture(self.cam_url)
        
        if not self.cap.isOpened():
            logger.error(f"Failed to connect to camera: {self.cam_url}")
            return False
        
        # If resolution is specified, set camera resolution
        if self.resolution:
            width, height = self.resolution
            # Set capture frame width and height
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            
            # Check if setting was successful
            actual_width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            actual_height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            
            if abs(actual_width - width) > 1 or abs(actual_height - height) > 1:
                logger.warning(f"Camera resolution request ({width}x{height}) not supported. "
                               f"Got ({actual_width}x{actual_height}). "
                               f"Will resize frames during collection.")
            else:
                logger.info(f"Camera resolution set to {width}x{height}")
            
        logger.info(f"Camera '{self.camera_name}' connected successfully")
        return True
        
    def start_episode(self):
        """Start new episode recording"""
        self.start_time = time.time()
        self.frame_count = 0
        
        # Create clean temp image directory
        if self.tmp_img_dir.exists():
            shutil.rmtree(self.tmp_img_dir)
        self.tmp_img_dir.mkdir(parents=True, exist_ok=True)
        
        # Reset and clear camera buffer completely
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
        
        logger.info(f"Started episode {self.episode_count} for camera '{self.camera_name}'")
        
    def collect_frame(self, frame):
        """Store frame for later processing
        
        Args:
            frame: Frame to store
        """
        if self.frame_count == 0:
            self.start_time = time.time()
            
            # Ensure temp directory exists
            try:
                if not self.tmp_img_dir.exists():
                    logger.warning(f"Temporary directory doesn't exist, recreating: {self.tmp_img_dir}")
                    self.tmp_img_dir.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                logger.error(f"Failed to create temporary directory: {e}")
                return
                
        # Adjust resolution (if needed)
        if self.resolution is not None and frame is not None:
            original_height, original_width = frame.shape[:2]
            target_width, target_height = self.resolution
            
            # Only resize when current resolution differs from target resolution
            if original_width != target_width or original_height != target_height:
                try:
                    frame = cv2.resize(frame, self.resolution)
                    logger.debug(f"Adjusting image resolution from ({original_width}x{original_height}) to ({target_width}x{target_height})")
                except Exception as e:
                    logger.error(f"Failed to adjust image resolution: {e}")
        
        # Save frame as image file
        try:
            if frame is not None:
                frame_path = self.tmp_img_dir / f"frame_{self.frame_count:05d}.png"
                
                # Ensure save directory exists
                if not frame_path.parent.exists():
                    frame_path.parent.mkdir(parents=True, exist_ok=True)
                    
                # Save image - use highest quality
                success = cv2.imwrite(str(frame_path), frame, [cv2.IMWRITE_PNG_COMPRESSION, 0])
                
                if success:
                    self.frame_count += 1
                    
                    # Log successful save
                    if self.frame_count % 30 == 0:  # Log every 30 frames to reduce log volume
                        logger.debug(f"Camera '{self.camera_name}' successfully saved frame to: {frame_path}")
                        
                        # Force file system sync every 30 frames
                        try:
                            os.fsync(os.open(str(frame_path), os.O_RDONLY))
                        except Exception as e:
                            logger.debug(f"Failed to sync file system: {e}")
                else:
                    logger.error(f"Failed to save image: {frame_path}")
            else:
                logger.warning(f"Camera '{self.camera_name}' received empty frame, skipping")
        except Exception as e:
            logger.error(f"Camera '{self.camera_name}' failed to save frame: {e}")
            
        # Force periodic frame saving
        if self.frame_count % 100 == 0:  # Save every 100 frames
            try:
                # Lightweight sync
                os.sync()
                logger.debug(f"Synced {self.frame_count} frames to disk - {self.camera_name}")
            except Exception as e:
                logger.error(f"Failed to sync frames to disk: {e}")
                
        return self.frame_count
        
    def save_video(self):
        """Save collected frames as video"""
        if self.frame_count == 0:
            logger.warning(f"No frames to save for camera '{self.camera_name}'")
            return
            
        # Ensure all frames are written to disk
        self.flush_frames()
            
        # Check if temporary image directory exists
        if not self.tmp_img_dir.exists():
            logger.error(f"Temporary image directory does not exist: {self.tmp_img_dir}")
            return None
            
        # Check if there are frame images
        frame_files = list(sorted(self.tmp_img_dir.glob("frame_*.png")))
        logger.info(f"Found {len(frame_files)} frame images ready to save as video (camera: {self.camera_name})")
        
        if not frame_files:
            logger.error(f"No frame images found: {self.tmp_img_dir}")
            return None
            
        # Print some frame filenames for debugging
        if len(frame_files) > 0:
            logger.info(f"First frame: {frame_files[0].name}, Last frame: {frame_files[-1].name}")
        
        # Ensure output directory exists again
        if not self.camera_dir.exists():
            logger.warning(f"Video output directory does not exist, recreating: {self.camera_dir}")
            try:
                self.camera_dir.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                logger.error(f"Failed to create video output directory: {e}")
                return None
                
        # Get next index by checking existing files
        existing_videos = list(self.camera_dir.glob("episode_*.mp4"))
        next_index = len(existing_videos)
                
        # Create video file
        video_filename = f"episode_{next_index:06d}.mp4"
        video_path = self.camera_dir / video_filename
        
        # Validate video path
        try:
            video_dir = video_path.parent
            if not video_dir.exists():
                logger.warning(f"Video directory does not exist, creating: {video_dir}")
                video_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.error(f"Failed to validate video path: {e}")
        
        logger.info(f"Preparing to save video to: {video_path}")
        
        # First try using more reliable cv2 method to directly save video
        success = False
        try:
            logger.info(f"Attempting to save video using OpenCV directly (frames: {len(frame_files)}, camera: {self.camera_name})...")
            
            if len(frame_files) > 0:
                # Read first frame to get dimensions
                test_img = cv2.imread(str(frame_files[0]))
                if test_img is not None:
                    height, width = test_img.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use MPEG-4 encoding
                    video_writer = cv2.VideoWriter(str(video_path), fourcc, self.fps, (width, height))
                    
                    if video_writer.isOpened():
                        # Write frame by frame
                        for frame_file in frame_files:
                            img = cv2.imread(str(frame_file))
                            if img is not None:
                                video_writer.write(img)
                        
                        video_writer.release()
                        logger.info(f"Successfully saved video using OpenCV: {video_path} (frame count: {len(frame_files)})")
                        success = True
                    else:
                        logger.error(f"OpenCV video writer creation failed: {video_path}")
                else:
                    logger.error(f"Cannot read first frame: {frame_files[0]}")
            else:
                logger.error(f"No frame files found to generate video: {self.tmp_img_dir}")
        except Exception as e:
            logger.error(f"Failed to save video using OpenCV: {e}")
            
        # If OpenCV fails, try using ffmpeg
        if not success:
            # Use absolute path and add more ffmpeg parameters
            input_pattern = str(self.tmp_img_dir / "frame_%05d.png")
            cmd = f"ffmpeg -y -framerate {self.fps} -i \"{input_pattern}\" "\
                f"-c:v libx264 -pix_fmt yuv420p -crf 23 \"{str(video_path)}\""
            
            logger.info(f"Executing command: {cmd}")
            
            result = os.system(cmd)
            if result != 0:
                logger.error(f"ffmpeg command failed, return code: {result}")
                # If ffmpeg also fails, last fallback method: copy all frames as individual files
                logger.info(f"Attempting to backup all frames to output directory...")
                backup_dir = self.camera_dir / f"episode_{next_index:06d}_frames"
                try:
                    backup_dir.mkdir(parents=True, exist_ok=True)
                    for i, frame_file in enumerate(frame_files):
                        dst_file = backup_dir / f"frame_{i:05d}.png"
                        shutil.copy2(frame_file, dst_file)
                    logger.info(f"Successfully backed up {len(frame_files)} frames to {backup_dir}")
                except Exception as e:
                    logger.error(f"Failed to backup frames: {e}")
        
        # Update counters
        self.episode_count += 1
        self.episode_lengths.append(self.frame_count)
        logger.info(f"Video processing complete: {video_path} ({self.frame_count} frames, actually saved {len(frame_files)} frames) - {self.camera_name}")
        
        # Cleanup
        if self.tmp_img_dir.exists():
            try:
                # Wait a moment to ensure all file operations are complete
                time.sleep(0.5)
                shutil.rmtree(self.tmp_img_dir)
                logger.debug(f"Cleaned up temporary directory: {self.tmp_img_dir}")
            except Exception as e:
                logger.error(f"Failed to clean up temporary directory: {e}")
        
        # Reset frame count
        self.frame_count = 0
        return next_index
        
    def flush_frames(self):
        """Ensure all cached frames are written to disk"""
        # Called before video saving to ensure all frames are written to disk
        logger.info(f"Ensuring all frames are written to disk (camera: {self.camera_name})...")
        
        # Force any pending filesystem operations
        try:
            os.sync()  # Try to sync filesystem
            
            # Check files in temporary directory
            if self.tmp_img_dir.exists():
                frame_files = list(sorted(self.tmp_img_dir.glob("frame_*.png")))
                logger.info(f"Temporary directory {self.tmp_img_dir} has {len(frame_files)} frames")
                
                # Check if last frame exists
                if self.frame_count > 0:
                    last_frame_path = self.tmp_img_dir / f"frame_{self.frame_count-1:05d}.png"
                    if not last_frame_path.exists():
                        logger.warning(f"Last frame {last_frame_path.name} does not exist!")
                    else:
                        logger.info(f"Last frame {last_frame_path.name} exists")
        except Exception as e:
            logger.error(f"Error when flushing frames: {e}")

    def _compute_video_stats(self, episode_index: int) -> Dict:
        """Calculate RGB statistics for video frames"""
        video_path = self.camera_dir / f"episode_{episode_index:06d}.mp4"
        if not video_path.exists():
            return None
            
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return None
            
        # Initialize accumulation values
        frame_count = 0
        rgb_sum = np.zeros(3)
        rgb_sq_sum = np.zeros(3)
        rgb_min = np.ones(3) * 255
        rgb_max = np.zeros(3)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Calculate statistics for each channel
            frame_float = frame.astype(np.float32)
            rgb_sum += frame_float.mean(axis=(0,1))
            rgb_sq_sum += (frame_float**2).mean(axis=(0,1))
            rgb_min = np.minimum(rgb_min, frame_float.min(axis=(0,1)))
            rgb_max = np.maximum(rgb_max, frame_float.max(axis=(0,1)))
            frame_count += 1
            
        cap.release()
        
        if frame_count == 0:
            return None
            
        # Calculate mean and standard deviation
        rgb_mean = rgb_sum / frame_count
        rgb_std = np.sqrt(rgb_sq_sum/frame_count - rgb_mean**2)
        
        # Format as required nested list structure
        return {
            "min": [[[v]] for v in rgb_min],
            "max": [[[v]] for v in rgb_max],
            "mean": [[[v]] for v in rgb_mean],
            "std": [[[v]] for v in rgb_std],
            "count": [frame_count]
        }

    def __del__(self):
        """Cleanup resources"""
        if self.cap:
            self.cap.release()

    def capture_and_save_frame(self):
        """Capture and save a frame
        
        Returns:
            bool: Whether the operation was successful
        """
        if self.start_time is None:
            self.start_episode()
        
        # Record some statistics
        frames_before = self.frame_count    
            
        # Capture frame
        if self.cap and self.cap.isOpened():
            try:
                # First try to release any buffered frames
                for _ in range(3):  # Try to clear some cache
                    self.cap.grab()
                
                # Then capture new frame
                ret, frame = self.cap.read()
                if ret and frame is not None:
                    # Add debug output
                    if self.frame_count % 30 == 0:  # Reduce log frequency
                        logger.debug(f"Camera '{self.camera_name}' successfully captured frame #{self.frame_count}")
                    
                    # Pass frame to collect_frame for processing and saving
                    self.collect_frame(frame)
                    
                    # Check if frame was successfully added
                    if self.frame_count > frames_before:
                        return True
                    else:
                        logger.warning(f"Frame count did not increase: {frames_before} -> {self.frame_count}")
                        return False
                else:
                    logger.warning(f"Camera '{self.camera_name}' failed to capture valid frame")
                    
                    # Try to reinitialize camera
                    if not ret:
                        attempts = 0
                        while attempts < 3 and not ret:
                            logger.info(f"Attempting to reinitialize camera {self.camera_name} (attempt {attempts+1})")
                            self.cap.release()
                            time.sleep(0.5)
                            self.setup_camera()
                            if self.cap and self.cap.isOpened():
                                ret, frame = self.cap.read()
                                if ret and frame is not None:
                                    logger.info(f"Camera {self.camera_name} reinitialization successful")
                                    self.collect_frame(frame)
                                    return True
                            attempts += 1
            except Exception as e:
                logger.error(f"Error capturing frame: {e}")
                # Try to recover camera connection
                try:
                    self.cap.release()
                    time.sleep(1)
                    self.setup_camera()
                    logger.info(f"Attempted to recover camera {self.camera_name} connection")
                except Exception as recovery_e:
                    logger.error(f"Failed to recover camera connection: {recovery_e}")
        else:
            logger.warning(f"Camera '{self.camera_name}' not open or unavailable")
            # Try to initialize camera
            try:
                if self.cap:
                    self.cap.release()
                self.cap = None
                time.sleep(1)
                self.setup_camera()
                logger.info(f"Attempted to reinitialize camera {self.camera_name}")
            except Exception as e:
                logger.error(f"Failed to reinitialize camera: {e}")
            
        return False 