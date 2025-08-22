# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This file contains utilities for recording frames from network cameras.
For more info look at `NetworkCamera` docstring.
"""

import argparse
import math
import threading
import time
from pathlib import Path
from threading import Thread
import logging
import concurrent.futures
import shutil

import numpy as np
from PIL import Image

from lerobot.common.robot_devices.cameras.configs import NetworkCameraConfig
from lerobot.common.robot_devices.utils import (
    RobotDeviceAlreadyConnectedError,
    RobotDeviceNotConnectedError,
    busy_wait,
)
from lerobot.common.utils.utils import capture_timestamp_utc

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def validate_network_camera_url(url: str, timeout: float = 5.0, mock: bool = False) -> bool:
    """
    Validates if a URL points to a valid camera stream
    
    Args:
        url: Camera URL to validate
        timeout: Connection timeout in seconds
        mock: Whether to use mock OpenCV
        
    Returns:
        bool: True if valid, False otherwise
    """
    if mock:
        import tests.mock_cv2 as cv2
    else:
        import cv2
    
    cap = cv2.VideoCapture(url)
    start_time = time.time()
    is_valid = False
    
    while time.time() - start_time < timeout:
        if cap.isOpened():
            # Try to read a frame to verify connection
            ret, _ = cap.read()
            if ret:
                is_valid = True
                break
            time.sleep(0.1)
    
    cap.release()
    return is_valid


def find_network_cameras(urls: list[str], timeout: float = 5.0, mock: bool = False) -> list[str]:
    """
    Tests a list of network camera URLs and returns the valid ones
    
    Args:
        urls: List of camera URLs to test
        timeout: Connection timeout in seconds per camera
        mock: Whether to use mock OpenCV
        
    Returns:
        list[str]: List of valid camera URLs
    """
    valid_urls = []
    
    for url in urls:
        logger.info(f"Testing camera URL: {url}")
        if validate_network_camera_url(url, timeout, mock):
            logger.info(f"Valid camera found at URL: {url}")
            valid_urls.append(url)
        else:
            logger.warning(f"Could not connect to camera URL: {url}")
    
    return valid_urls


def save_image(img_array, camera_url, frame_index, images_dir):
    """
    Save an image from a network camera to disk
    
    Args:
        img_array: Image array from camera
        camera_url: Camera URL for naming
        frame_index: Frame index
        images_dir: Directory to save images
    """
    img = Image.fromarray(img_array)
    # Create a safe filename from the URL
    safe_url = "".join(c if c.isalnum() else "_" for c in camera_url)[:30]
    path = images_dir / f"camera_{safe_url}_frame_{frame_index:06d}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(str(path), quality=100)


def save_images_from_network_cameras(
    images_dir: Path,
    camera_urls: list[str] | None = None,
    fps=None,
    width=None,
    height=None,
    record_time_s=2,
    connection_timeout=5.0,
    mock=False,
):
    """
    Initializes all the network cameras and saves images to the directory.
    Useful to visually identify the cameras.
    
    Args:
        images_dir: Directory to save images
        camera_urls: List of camera URLs to use
        fps: Frame rate to use (or None for camera default)
        width: Width to use (or None for camera default)
        height: Height to use (or None for camera default)
        record_time_s: How long to record
        connection_timeout: Connection timeout in seconds
        mock: Whether to use mock OpenCV
    """
    if camera_urls is None or len(camera_urls) == 0:
        logger.error("No camera URLs provided")
        return

    logger.info("Connecting to network cameras")
    cameras = []
    for url in camera_urls:
        config = NetworkCameraConfig(
            url=url, 
            fps=fps, 
            width=width, 
            height=height, 
            connection_timeout=connection_timeout,
            mock=mock
        )
        camera = NetworkCamera(config)
        try:
            camera.connect()
            logger.info(
                f"NetworkCamera({camera.url}, fps={camera.fps}, width={camera.width}, "
                f"height={camera.height}, color_mode={camera.color_mode})"
            )
            cameras.append(camera)
        except Exception as e:
            logger.error(f"Failed to connect to camera {url}: {e}")

    if not cameras:
        logger.error("No cameras could be connected")
        return

    images_dir = Path(images_dir)
    if images_dir.exists():
        shutil.rmtree(images_dir)
    images_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving images to {images_dir}")
    frame_index = 0
    start_time = time.perf_counter()
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(cameras)) as executor:
        while True:
            now = time.perf_counter()

            futures = []
            for camera in cameras:
                try:
                    # If fps is None, use sync read to avoid saving the same image multiple times
                    image = camera.read() if fps is None else camera.async_read()
                    
                    futures.append(
                        executor.submit(
                            save_image,
                            image,
                            camera.url,
                            frame_index,
                            images_dir,
                        )
                    )
                except Exception as e:
                    logger.error(f"Error capturing frame from {camera.url}: {e}")
            
            # Wait for all futures to complete
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"Error saving image: {e}")

            if fps is not None:
                dt_s = time.perf_counter() - now
                busy_wait(1 / fps - dt_s)

            logger.info(f"Frame: {frame_index:04d}\tLatency (ms): {(time.perf_counter() - now) * 1000:.2f}")

            if time.perf_counter() - start_time > record_time_s:
                break

            frame_index += 1

    # Cleanup
    for camera in cameras:
        try:
            camera.disconnect()
        except Exception as e:
            logger.error(f"Error disconnecting camera {camera.url}: {e}")

    logger.info(f"Images have been saved to {images_dir}")


class NetworkCamera:
    """
    The NetworkCamera class allows efficiently recording images from network cameras like IP cameras, 
    RTSP streams, webcams, etc. It relies on OpenCV to handle the connection and frame retrieval.
    
    Features:
    - Connection retry mechanism
    - Async reading support
    - Configurable timeout and retry settings
    - Support for rotation
    
    Example usage:
    ```python
    from lerobot.common.robot_devices.cameras.configs import NetworkCameraConfig
    
    # Basic RTSP camera
    config = NetworkCameraConfig("rtsp://admin:password@192.168.1.100:554/stream1")
    camera = NetworkCamera(config)
    camera.connect()
    color_image = camera.read()
    # When done, disconnect
    camera.disconnect()
    
    # With specific settings
    config = NetworkCameraConfig(
        "http://192.168.1.101:8080/video", 
        fps=30, 
        width=1280, 
        height=720,
        connection_timeout=5.0,
        retry_count=3,
        retry_delay=1.0
    )
    ```
    """

    def __init__(self, config: NetworkCameraConfig):
        self.config = config
        self.url = config.url
        self.fps = config.fps
        self.width = config.width
        self.height = config.height
        self.channels = config.channels
        self.color_mode = config.color_mode
        self.connection_timeout = config.connection_timeout
        self.retry_count = config.retry_count
        self.retry_delay = config.retry_delay
        self.buffer_size = config.buffer_size
        self.mock = config.mock

        self.camera = None
        self.is_connected = False
        self.thread = None
        self.stop_event = None
        self.color_image = None
        self.logs = {}
        self.connect_attempt = 0

        if self.mock:
            import tests.mock_cv2 as cv2
        else:
            import cv2

        # Setup rotation
        self.rotation = None
        if config.rotation == -90:
            self.rotation = cv2.ROTATE_90_COUNTERCLOCKWISE
        elif config.rotation == 90:
            self.rotation = cv2.ROTATE_90_CLOCKWISE
        elif config.rotation == 180:
            self.rotation = cv2.ROTATE_180

    def connect(self):
        """
        Connect to the network camera.
        
        Raises:
            RobotDeviceAlreadyConnectedError: If already connected
            OSError: If connection fails after retries
        """
        if self.is_connected:
            raise RobotDeviceAlreadyConnectedError(f"NetworkCamera({self.url}) is already connected.")

        if self.mock:
            import tests.mock_cv2 as cv2
        else:
            import cv2
            # Limit OpenCV threads to avoid interfering with other operations
            cv2.setNumThreads(1)

        # Try to connect with retry mechanism
        success = False
        error = None
        
        for attempt in range(self.retry_count):
            try:
                self.connect_attempt = attempt + 1
                logger.info(f"Connecting to {self.url} (attempt {self.connect_attempt}/{self.retry_count})")
                
                # Create camera capture
                self.camera = cv2.VideoCapture(self.url)
                
                # Set up buffer size
                self.camera.set(cv2.CAP_PROP_BUFFERSIZE, self.buffer_size)
                
                # Wait for connection to establish
                start_time = time.time()
                while time.time() - start_time < self.connection_timeout:
                    if self.camera.isOpened():
                        # Configure camera if needed
                        self._configure_camera()
                        success = True
                        break
                    time.sleep(0.1)
                
                if success:
                    break
                    
                # If we got here, connection timed out
                self.camera.release()
                logger.warning(f"Connection timed out for {self.url}")
                
            except Exception as e:
                error = e
                logger.warning(f"Connection attempt {self.connect_attempt} failed: {e}")
                if self.camera:
                    self.camera.release()
                    self.camera = None
            
            # Wait before retrying
            if attempt < self.retry_count - 1:
                time.sleep(self.retry_delay)
        
        if not success:
            err_msg = f"Failed to connect to NetworkCamera({self.url}) after {self.retry_count} attempts."
            if error:
                err_msg += f" Last error: {error}"
            raise OSError(err_msg)
        
        self.is_connected = True
        logger.info(f"Successfully connected to {self.url}")

    def _configure_camera(self):
        """
        Configure camera properties like FPS, width and height if specified
        """
        if self.mock:
            import tests.mock_cv2 as cv2
        else:
            import cv2
            
        # Clear initial frames to get clean state
        for _ in range(5):
            self.camera.grab()
            
        # Set properties if specified
        if self.fps is not None:
            self.camera.set(cv2.CAP_PROP_FPS, self.fps)
        if self.width is not None:
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        if self.height is not None:
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            
        # Read actual values
        actual_fps = self.camera.get(cv2.CAP_PROP_FPS)
        actual_width = self.camera.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_height = self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT)
        
        # Check if settings were applied
        if self.fps is not None and not math.isclose(self.fps, actual_fps, rel_tol=1e-3):
            logger.warning(
                f"Couldn't set fps={self.fps} for NetworkCamera({self.url}). "
                f"Using actual value: {actual_fps}"
            )
        if self.width is not None and not math.isclose(self.width, actual_width, rel_tol=1e-3):
            logger.warning(
                f"Couldn't set width={self.width} for NetworkCamera({self.url}). "
                f"Using actual value: {actual_width}"
            )
        if self.height is not None and not math.isclose(self.height, actual_height, rel_tol=1e-3):
            logger.warning(
                f"Couldn't set height={self.height} for NetworkCamera({self.url}). "
                f"Using actual value: {actual_height}"
            )
            
        # Update with actual values
        self.fps = round(actual_fps) if actual_fps > 0 else 30  # Default to 30 fps if not reported correctly
        self.width = round(actual_width)
        self.height = round(actual_height)

    def read(self, temporary_color_mode: str | None = None) -> np.ndarray:
        """
        Read a frame from the camera returned in the format (height, width, channels)
        
        Args:
            temporary_color_mode: Override color mode for this read only
            
        Returns:
            np.ndarray: Image frame in the specified color mode
            
        Raises:
            RobotDeviceNotConnectedError: If not connected
            OSError: If image capture fails
        """
        if not self.is_connected:
            raise RobotDeviceNotConnectedError(
                f"NetworkCamera({self.url}) is not connected. Try running `camera.connect()` first."
            )

        start_time = time.perf_counter()
        max_read_attempts = 3
        
        for attempt in range(max_read_attempts):
            ret, color_image = self.camera.read()
            
            if ret and color_image is not None:
                break
                
            if attempt < max_read_attempts - 1:
                logger.warning(f"Failed to read frame, retrying ({attempt+1}/{max_read_attempts})...")
                # Clear buffer
                for _ in range(3):
                    self.camera.grab()
                time.sleep(0.05)
        
        if not ret or color_image is None:
            raise OSError(f"Can't capture color image from network camera {self.url}.")

        requested_color_mode = self.color_mode if temporary_color_mode is None else temporary_color_mode

        if requested_color_mode not in ["rgb", "bgr"]:
            raise ValueError(
                f"Expected color values are 'rgb' or 'bgr', but {requested_color_mode} is provided."
            )

        if self.mock:
            import tests.mock_cv2 as cv2
        else:
            import cv2

        # Convert BGR to RGB if needed (OpenCV default is BGR)
        if requested_color_mode == "rgb":
            color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

        # Check dimensions
        h, w, _ = color_image.shape
        if (self.height is not None and h != self.height) or (self.width is not None and w != self.width):
            logger.warning(
                f"Image dimensions ({h}x{w}) don't match expected dimensions "
                f"({self.height}x{self.width})"
            )
            # Update internal dimensions to match actual frame
            self.height = h
            self.width = w

        # Apply rotation if needed
        if self.rotation is not None:
            color_image = cv2.rotate(color_image, self.rotation)

        # Store logs
        self.logs["delta_timestamp_s"] = time.perf_counter() - start_time
        self.logs["timestamp_utc"] = capture_timestamp_utc()
        self.color_image = color_image

        return color_image

    def read_loop(self):
        """Background thread for async reading"""
        while not self.stop_event.is_set():
            try:
                self.color_image = self.read()
                # Small sleep to avoid high CPU usage
                time.sleep(0.01)
            except Exception as e:
                logger.error(f"Error in read_loop for {self.url}: {e}")
                time.sleep(0.5)  # Wait a bit on error

    def async_read(self) -> np.ndarray:
        """
        Non-blocking read that uses a background thread
        
        Returns:
            np.ndarray: Latest camera frame
            
        Raises:
            RobotDeviceNotConnectedError: If not connected
            TimeoutError: If frame read times out
        """
        if not self.is_connected:
            raise RobotDeviceNotConnectedError(
                f"NetworkCamera({self.url}) is not connected. Try running `camera.connect()` first."
            )

        # Start background thread if needed
        if self.thread is None:
            self.stop_event = threading.Event()
            self.thread = Thread(target=self.read_loop)
            self.thread.daemon = True
            self.thread.start()

        # Wait for first frame
        num_tries = 0
        timeout_seconds = 5
        while True:
            if self.color_image is not None:
                return self.color_image

            time.sleep(1 / (self.fps or 30))
            num_tries += 1
            
            if num_tries > timeout_seconds * (self.fps or 30):
                raise TimeoutError(f"Timed out waiting for frame from {self.url} after {timeout_seconds} seconds")

    def reconnect(self):
        """Attempt to reconnect if the connection was lost"""
        logger.info(f"Attempting to reconnect to {self.url}")
        if self.is_connected:
            self.disconnect()
        time.sleep(1.0)  # Give some time before reconnecting
        self.connect()

    def disconnect(self):
        """
        Disconnect from the camera
        
        Raises:
            RobotDeviceNotConnectedError: If not connected
        """
        if not self.is_connected:
            raise RobotDeviceNotConnectedError(
                f"NetworkCamera({self.url}) is not connected. Try running `camera.connect()` first."
            )

        # Stop background thread if running
        if self.thread is not None:
            self.stop_event.set()
            self.thread.join(timeout=2.0)  # Wait up to 2 seconds for thread to finish
            if self.thread.is_alive():
                logger.warning(f"Background thread for {self.url} did not terminate gracefully")
            self.thread = None
            self.stop_event = None

        # Release camera
        if self.camera is not None:
            self.camera.release()
            self.camera = None
        
        self.is_connected = False
        logger.info(f"Disconnected from {self.url}")

    def __del__(self):
        """Clean up resources on destruction"""
        if getattr(self, "is_connected", False):
            try:
                self.disconnect()
            except Exception as e:
                logger.error(f"Error during cleanup of NetworkCamera({self.url}): {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Save frames from network cameras."
    )
    parser.add_argument(
        "--camera-urls",
        type=str,
        nargs="+",
        required=True,
        help="List of camera URLs to connect to."
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=None,
        help="Set the number of frames recorded per second. If not provided, use the default fps of each camera."
    )
    parser.add_argument(
        "--width",
        type=int,
        default=None,
        help="Set the width for all cameras. If not provided, use the default width of each camera."
    )
    parser.add_argument(
        "--height",
        type=int,
        default=None,
        help="Set the height for all cameras. If not provided, use the default height of each camera."
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default="outputs/images_from_network_cameras",
        help="Set directory to save frames for each camera."
    )
    parser.add_argument(
        "--record-time-s",
        type=float,
        default=4.0,
        help="Set the number of seconds used to record the frames. By default, 4 seconds."
    )
    parser.add_argument(
        "--connection-timeout",
        type=float,
        default=5.0,
        help="Set the connection timeout in seconds. By default, 5 seconds."
    )
    
    args = parser.parse_args()
    save_images_from_network_cameras(**vars(args))
