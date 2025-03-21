from dataclasses import dataclass
import numpy as np
from scipy.spatial.transform import Rotation as R
from typing import Dict, Optional, Tuple, Any
import fibre
import ref_tool
import threading
import time
import copy
import gym
from abc import ABC, abstractmethod
from dummy_utils import EEFPose6D, get_dummy_eef_pose6d, convert_eef_to_telemoma_obs, convert_telemoma_action_to_eef, precise_sleep


class DummyEnv(gym.Env):
    """继承标准Gym环境接口"""
    def __init__(self, config, init_reset=True):
        super().__init__()
        self.config = config
        
        if config.simulation:
            self.drive = SimRobotDrive()
        else:
            self.drive = RealRobotDrive(config.arm_serial)
            
        self.controller = MotionController(self.drive, config)
        
        if init_reset:
            self.reset()
    
    def get_all_joints(self):
        return self.controller.joint_angles

    def reset(self) -> Dict[str, Optional[np.ndarray]]:
        """Reset robot and restart movement thread"""
        self.controller.stop_control_loop()
        self.controller.start_control_loop()
        return convert_eef_to_telemoma_obs(self.controller.current_pose)

    def step(self, action: Dict[str, np.ndarray]) -> Tuple[Dict[str, Optional[np.ndarray]], float, bool, Dict]:
        """Update target pose for movement thread"""
        action_np = np.array(action[self.config.arm_type])
        new_target_pose = convert_telemoma_action_to_eef(self.controller.current_pose, action_np)
        self.controller.update_target_pose(new_target_pose)
        obs = convert_eef_to_telemoma_obs(self.controller.current_pose)
        return obs, 0.0, False, {}

    def stop(self):
        """Stop robot movement"""
        self.controller.stop_control_loop()

    def close(self):
        """Cleanup resources"""
        self.stop()

    def __enter__(self):
        """Support for 'with' statement"""
        self.controller.start_control_loop()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Cleanup when exiting 'with' block"""
        self.close()
        return False
