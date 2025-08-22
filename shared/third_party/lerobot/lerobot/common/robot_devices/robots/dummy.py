"""
Dummy机械臂机器人类
"""

import time
from dataclasses import asdict
from typing import Dict, Any, Optional

import numpy as np
from PIL import Image

from lerobot.common.robot_devices.cameras.configs import CameraConfig
from lerobot.common.robot_devices.cameras.utils import make_cameras_from_configs
from lerobot.common.robot_devices.motors.configs import DummyMotorsBusConfig
from lerobot.common.robot_devices.motors.utils import make_motors_buses_from_configs, get_motor_names
from lerobot.common.robot_devices.robots.configs import DummyRobotConfig
from lerobot.common.robot_devices.teleop.dummy_ctrl import TeachFollowArmController
from lerobot.common.robot_devices.utils import RobotDeviceAlreadyConnectedError, RobotDeviceNotConnectedError
import torch

class DummyRobot:
    """
    """

    def __init__(self, config: DummyRobotConfig):
        self.config = config
        self.robot_type = "dummy"
        self.inference_time = self.config.inference_time
        self.control_mode = "joint"  # 控制模式: "joint" 或 "eef"，默认使用末端控制

        if hasattr(self.config, "leader_arms"):
            self.leader_motor = make_motors_buses_from_configs(self.config.leader_arms)
            self.leader_arm = self.leader_motor['main']
        else:
            self.leader_arm = None

        if hasattr(self.config, "follower_arms"):
            self.follower_motor = make_motors_buses_from_configs(self.config.follower_arms)
            self.follower_arm = self.follower_motor['main']
        else:
            self.follower_arm = None

        self.cameras = make_cameras_from_configs(self.config.cameras)
        self.observation_space = {}
        
        if not self.inference_time:
            self.teleop = TeachFollowArmController(self.leader_arm, self.follower_arm)
        else:
            self.teleop = None
        
        self.logs = {}
        self.is_connected = False

    @property
    def camera_features(self)->dict:
        cam_ft = {}
        for name, cam in self.cameras.items():
            key = f"observation.images.{name}"
            cam_ft[key] = {
                "shape": (cam.height, cam.width, cam.channels),
                "names": ["height", "width", "channels"],
                "info": None,
            }
        return cam_ft
    @property
    def motor_features(self) -> dict:
        action_names = get_motor_names(self.leader_motor)
        state_names = get_motor_names(self.follower_motor)  
        return {
            "action": {
                "dtype": "float32",
                "shape": (7,),
                "names": action_names,
            },
            "observation.state": {
                "dtype": "float32",
                "shape": (len(state_names),),
                "names": state_names,
            },
        }

    @property
    def has_camera(self):
        return len(self.cameras) > 0

    @property
    def num_cameras(self):
        return len(self.cameras)   
    
    def connect(self) -> bool:
        """
        连接机械臂和摄像头
        """
        if self.is_connected:
            raise RobotDeviceAlreadyConnectedError(
                "Piper is already connected. Do not run `robot.connect()` twice."
            )

        self.leader_arm.leader_connect(enable=True)
        self.follower_arm.follower_connect(enable=True)

        for name in self.cameras:
            self.cameras[name].connect()
            self.is_connected = self.is_connected and self.cameras[name].is_connected
            print(f"camera {name} conneted")

        self.is_connected = True

    def run_calibration(self) -> Dict:
        """
        """
        return {}

    def teleop_step(self, record_data=False) -> Dict:
        """
        """

        if not self.is_connected:
            raise ConnectionError()
        if self.inference_time:
            return {}
        
        # read target pose state as
        time.sleep(3)  # Wait for 3 seconds
        self.leader_arm.enable_false()
        # read target pose state as
        before_read_t = time.perf_counter()
        state = self.follower_arm.read()
        action = self.teleop.get_action()
        self.logs["read_pos_dt_s"] = time.perf_counter() - before_read_t

        # do action
        before_write_t = time.perf_counter()
        target_joints = list(action.values())
        self.follower_arm.write(target_joints)
        self.logs["write_pos_dt_s"] = time.perf_counter() - before_write_t

        if not record_data:
            return

        state = torch.as_tensor(list(state.values()), dtype=torch.float32)
        action = torch.as_tensor(list(action.values()), dtype=torch.float32)

        images = {}
        for name in self.cameras:
            before_camread_t = time.perf_counter()
            images[name] = self.cameras[name].async_read()
            images[name] = torch.from_numpy(images[name])
            self.logs[f"read_camera_{name}_dt_s"] = self.cameras[name].logs["delta_timestamp_s"]
            self.logs[f"async_read_camera_{name}_dt_s"] = time.perf_counter() - before_camread_t

        obs_dict, action_dict = {}, {}
        obs_dict["observation.state"] = state
        action_dict["action"] = action
        for name in self.cameras:
            obs_dict[f"observation.images.{name}"] = images[name]

        return obs_dict, action_dict    


    def send_action(self, action: Dict[str, Any]) -> Dict:
        """Write the predicted actions from policy to the motors"""
        if not self.is_connected:
            raise RobotDeviceNotConnectedError(
                "Piper is not connected. You need to run `robot.connect()`."
            )
        if self.control_mode == "joint":
            target_joints = action.tolist()
            self.follower_arm.write(target_joints)
        else:
            target_pose = action.tolist()
            self.follower_arm.write_eef(target_pose)
        
        return action

    def capture_observation(self) -> dict:
        """capture current images and joint positions"""
        if not self.is_connected:
            raise RobotDeviceNotConnectedError(
                "Piper is not connected. You need to run `robot.connect()`."
            )
        
        # read current joint positions
        before_read_t = time.perf_counter()
        state = self.follower_arm.read()  # 6 joints + 1 gripper
        self.logs["read_pos_dt_s"] = time.perf_counter() - before_read_t

        state = torch.as_tensor(list(state.values()), dtype=torch.float32)

        # read images from cameras
        images = {}
        for name in self.cameras:
            before_camread_t = time.perf_counter()
            images[name] = self.cameras[name].async_read()
            images[name] = torch.from_numpy(images[name])
            self.logs[f"read_camera_{name}_dt_s"] = self.cameras[name].logs["delta_timestamp_s"]
            self.logs[f"async_read_camera_{name}_dt_s"] = time.perf_counter() - before_camread_t

        # Populate output dictionnaries and format to pytorch
        obs_dict = {}
        obs_dict["observation.state"] = state
        for name in self.cameras:
            obs_dict[f"observation.images.{name}"] = images[name]
        return obs_dict



    def set_control_mode(self, mode: str):
        """
        设置控制模式: 'joint' 或 'eef'
        """
        if mode in ["joint", "eef"]:
            self.control_mode = mode
            print(f"控制模式已切换为: {mode}")
        else:
            print(f"无效的控制模式: {mode}，有效值为 'joint' 或 'eef'")

    def disconnect(self) -> None:
        """
        断开连接
        """
        # # 断开机械臂连接
        # if self.leader_arm:
        #     for arm in self.leader_arm.values():
        #         arm.safe_disconnect()

        # if self.follower_arm:
        #     for arm in self.follower_arm.values():
        #         arm.safe_disconnect()

        # # 断开摄像头连接
        # for camera in self.cameras.values():
        #     camera.disconnect()
            
        # 停止遥控器
        # if self.arm_controller:
        #     self.arm_controller.stop() 