#%%
import sys
sys.path.append("..")
import fibre
import numpy as np
from __future__ import print_function
logger = fibre.utils.Logger(verbose=True)
#%%
left_arm = fibre.find_any(serial_number="208C31875253", logger=logger)
#%%
right_arm = fibre.find_any(serial_number="3950366E3233", logger=logger)
#%%
joint_offset = np.array([0,-73,180,0,0,0])
#%%
from telemoma.configs.base_config import teleop_config
from telemoma.human_interface.teleop_policy import TeleopPolicy

teleop_config.arm_left_controller = 'aloha_real_arm'
teleop_config.arm_right_controller = 'aloha_real_arm'
teleop_config.base_controller = 'keyboard'
teleop_config.interface_kwargs.aloha_real_arm = {
    'left_arm_fibre': left_arm,
    'right_arm_fibre': right_arm,
    'joint_offset': joint_offset,
    'angle_measure': 'radian',
    'gripper_angle_dict': {
        'left_open_angle': 39,
        'right_open_angle': 38,
        'left_close_angle': -35,
        'right_close_angle': -33
    }
}
teleop = TeleopPolicy(teleop_config)
teleop.start()
#%%
from dummy_env_sim import make_dual_arm_dummy_env_sim
env = make_dual_arm_dummy_env_sim()
observation, info = env.reset()
#%%
left_arm.robot.set_enable(False)
right_arm.robot.set_enable(False)
#%%
import time
freq = 10
while True:
    start_time = time.monotonic()
    action = teleop.get_action(observation)
    logger.info(f"action: {action}")
    observation, reward, terminated, truncated, info = env.step(action)
    # logger.info(f"observation: {observation}")
    env.render()
    if terminated or truncated:
        observation, info = env.reset()
    elapsed = time.monotonic() - start_time
    sleep_time = max(0, 1/freq - elapsed)
    time.sleep(sleep_time)
# %%
