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

import abc
from dataclasses import dataclass

import draccus


@dataclass
class CameraConfig(draccus.ChoiceRegistry, abc.ABC):
    @property
    def type(self) -> str:
        return self.get_choice_name(self.__class__)


@CameraConfig.register_subclass("opencv")
@dataclass
class OpenCVCameraConfig(CameraConfig):
    """
    Example of tested options for Intel Real Sense D405:

    ```python
    OpenCVCameraConfig(0, 30, 640, 480)
    OpenCVCameraConfig(0, 60, 640, 480)
    OpenCVCameraConfig(0, 90, 640, 480)
    OpenCVCameraConfig(0, 30, 1280, 720)
    ```

    Network camera examples:
    
    ```python
    OpenCVCameraConfig(url="rtsp://username:password@192.168.1.100:554/stream1")
    OpenCVCameraConfig(url="http://192.168.1.101:8080/video")
    ```
    """

    camera_index: int = -1
    url: str | None = None
    fps: int | None = None
    width: int | None = None
    height: int | None = None
    color_mode: str = "rgb"
    channels: int | None = None
    rotation: int | None = None
    mock: bool = False

    def __post_init__(self):
        if self.camera_index == -1 and self.url is None:
            raise ValueError("Either `camera_index` or `url` must be provided.")
            
        if self.camera_index != -1 and self.url is not None:
            raise ValueError("Only one of `camera_index` or `url` should be provided, not both.")

        if self.color_mode not in ["rgb", "bgr"]:
            raise ValueError(
                f"`color_mode` is expected to be 'rgb' or 'bgr', but {self.color_mode} is provided."
            )

        self.channels = 3

        if self.rotation not in [-90, None, 90, 180]:
            raise ValueError(f"`rotation` must be in [-90, None, 90, 180] (got {self.rotation})")


@CameraConfig.register_subclass("intelrealsense")
@dataclass
class IntelRealSenseCameraConfig(CameraConfig):
    """
    Example of tested options for Intel Real Sense D405:

    ```python
    IntelRealSenseCameraConfig(128422271347, 30, 640, 480)
    IntelRealSenseCameraConfig(128422271347, 60, 640, 480)
    IntelRealSenseCameraConfig(128422271347, 90, 640, 480)
    IntelRealSenseCameraConfig(128422271347, 30, 1280, 720)
    IntelRealSenseCameraConfig(128422271347, 30, 640, 480, use_depth=True)
    IntelRealSenseCameraConfig(128422271347, 30, 640, 480, rotation=90)
    ```
    
    Network camera examples:
    
    ```python
    IntelRealSenseCameraConfig(url="rtsp://username:password@192.168.1.100:554/stream1")
    IntelRealSenseCameraConfig(url="http://192.168.1.101:8080/video", fps=30, width=640, height=480)
    ```
    """

    name: str | None = None
    serial_number: int | None = None
    url: str | None = None
    fps: int | None = None
    width: int | None = None
    height: int | None = None
    color_mode: str = "rgb"
    channels: int | None = None
    use_depth: bool = False
    force_hardware_reset: bool = True
    rotation: int | None = None
    mock: bool = False

    def __post_init__(self):
        # Check whether we have at least one way to identify the camera
        identifying_params = [bool(self.name), bool(self.serial_number), bool(self.url)]
        if sum(identifying_params) == 0:
            raise ValueError("One of `name`, `serial_number`, or `url` must be provided.")
            
        if sum(identifying_params) > 1:
            raise ValueError(
                f"Only one of `name`, `serial_number`, or `url` should be provided, "
                f"but {self.name=}, {self.serial_number=}, and {self.url=} were provided."
            )

        if self.color_mode not in ["rgb", "bgr"]:
            raise ValueError(
                f"`color_mode` is expected to be 'rgb' or 'bgr', but {self.color_mode} is provided."
            )

        self.channels = 3

        at_least_one_is_not_none = self.fps is not None or self.width is not None or self.height is not None
        at_least_one_is_none = self.fps is None or self.width is None or self.height is None
        if at_least_one_is_not_none and at_least_one_is_none:
            raise ValueError(
                "For `fps`, `width` and `height`, either all of them need to be set, or none of them, "
                f"but {self.fps=}, {self.width=}, {self.height=} were provided."
            )

        if self.rotation not in [-90, None, 90, 180]:
            raise ValueError(f"`rotation` must be in [-90, None, 90, 180] (got {self.rotation})")


@CameraConfig.register_subclass("network")
@dataclass
class NetworkCameraConfig(CameraConfig):
    """
    Configuration for network cameras like IP cameras, RTSP streams, etc.
    
    Example usage:
    
    ```python
    # RTSP stream
    NetworkCameraConfig("rtsp://username:password@192.168.1.100:554/stream1")
    
    # HTTP stream
    NetworkCameraConfig("http://192.168.1.101:8080/video")
    
    # With specific dimensions and frame rate
    NetworkCameraConfig("rtsp://192.168.1.100:554/stream1", fps=30, width=1280, height=720)
    
    # With connection timeout and retry settings
    NetworkCameraConfig("rtsp://192.168.1.100:554/stream1", connection_timeout=5.0, retry_count=3)
    ```
    """
    
    url: str
    fps: int | None = None
    width: int | None = None
    height: int | None = None
    color_mode: str = "rgb"
    channels: int | None = None
    rotation: int | None = None
    connection_timeout: float = 10.0  # Timeout in seconds for connection attempts
    retry_count: int = 5  # Number of connection retries before giving up
    retry_delay: float = 1.0  # Delay between retries in seconds
    buffer_size: int = 1  # OpenCV buffer size
    mock: bool = False
    
    def __post_init__(self):
        if not self.url:
            raise ValueError("`url` must be provided for network cameras.")
            
        if self.color_mode not in ["rgb", "bgr"]:
            raise ValueError(
                f"`color_mode` is expected to be 'rgb' or 'bgr', but {self.color_mode} is provided."
            )
            
        self.channels = 3
        
        at_least_one_is_not_none = self.fps is not None or self.width is not None or self.height is not None
        at_least_one_is_none = self.fps is None or self.width is None or self.height is None
        if at_least_one_is_not_none and at_least_one_is_none:
            raise ValueError(
                "For `fps`, `width` and `height`, either all of them need to be set, or none of them, "
                f"but {self.fps=}, {self.width=}, {self.height=} were provided."
            )
            
        if self.rotation not in [-90, None, 90, 180]:
            raise ValueError(f"`rotation` must be in [-90, None, 90, 180] (got {self.rotation})")
