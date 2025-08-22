from telemoma.human_interface.teleop_core import BaseTeleopInterface, TeleopAction, TeleopObservation
from telemoma.utils.general_utils import run_threaded_command
from telemoma.utils.transformations import quat_diff, quat_to_euler
import time
import numpy as np

class Aloha_Real_Arm_Interface(BaseTeleopInterface):
    def __init__(self, left_arm_fibre, right_arm_fibre, joint_offset=np.array([0,-73,180,0,0,0]), angle_measure='radian',
                 gripper_angle_dict={'left_open_angle': -89, 'right_open_angle': 38, 'left_close_angle': -158, 'right_close_angle': -33}) -> None:
        self.update_frequency = 100
        self.left_arm = left_arm_fibre
        self.right_arm = right_arm_fibre

        self.left_arm.robot.set_enable(True)
        self.right_arm.robot.set_enable(True)
        self.left_arm.robot.move_j(0, 0, 90, 0, 0, 0)
        self.right_arm.robot.move_j(0, 0, 90, 0, 0, 0)
        # initialize gripper to zero force
        self.left_arm.robot.hand.set_mode(0)
        self.right_arm.robot.hand.set_mode(0)
        self.left_arm.robot.hand.set_torque(0)
        self.right_arm.robot.hand.set_torque(0)

        self.left_open_angle = gripper_angle_dict['left_open_angle']    
        self.right_open_angle = gripper_angle_dict['right_open_angle']
        print(f"left_open_angle: {self.left_open_angle}, right_open_angle: {self.right_open_angle}")  
        self.left_close_angle = gripper_angle_dict['left_close_angle']
        self.right_close_angle = gripper_angle_dict['right_close_angle']
        print(f"left_close_angle: {self.left_close_angle}, right_close_angle: {self.right_close_angle}")  
        # stop fixed update to get zero force drag
        # self.left_arm.robot.set_enable(False)
        # self.right_arm.robot.set_enable(False)
        self.joint_offset = joint_offset
        self._state = {
            'right': {
                "joints": None,
            },

            'left': {
                "joints": None,
            }
        }
        self.angle_measure = angle_measure
    
    def get_gripper_action(self):
        """
        Determine the gripper action based on current hand angle.
        Returns normalized gripper position (0: close, 1: open) for both arms.
        """
        left_hand_angle = self.left_arm.robot.hand.angle
        right_hand_angle = self.right_arm.robot.hand.angle
        
        # Determine if gripper is closer to open or closed position
        left_open_ratio = 1 if abs(left_hand_angle - self.left_open_angle) < abs(left_hand_angle - self.left_close_angle) else 0
        right_open_ratio = 1 if abs(right_hand_angle - self.right_open_angle) < abs(right_hand_angle - self.right_close_angle) else 0
        
        return left_open_ratio, right_open_ratio
    def get_all_joints_angle(self):
        # return joint angle in same system as movej(A_t)
        left_all_joint_angles =  self.left_arm.robot.joint_1.angle, self.left_arm.robot.joint_2.angle, self.left_arm.robot.joint_3.angle, self.left_arm.robot.joint_4.angle, self.left_arm.robot.joint_5.angle, self.left_arm.robot.joint_6.angle
        right_all_joint_angles = self.right_arm.robot.joint_1.angle, self.right_arm.robot.joint_2.angle, self.right_arm.robot.joint_3.angle, self.right_arm.robot.joint_4.angle, self.right_arm.robot.joint_5.angle, self.right_arm.robot.joint_6.angle
        left_all_joint_angles = np.array(left_all_joint_angles) + self.joint_offset
        right_all_joint_angles = np.array(right_all_joint_angles) + self.joint_offset
        # 1, 1: open, 0, 0: close
        left_gripper_action, right_gripper_action = self.get_gripper_action()

        if self.angle_measure == 'radian':
            left_all_joint_angles = np.deg2rad(left_all_joint_angles)
            right_all_joint_angles = np.deg2rad(right_all_joint_angles)

        left_all_joint_angles = np.concatenate([left_all_joint_angles, [left_gripper_action]])
        right_all_joint_angles = np.concatenate([right_all_joint_angles, [right_gripper_action]])
        # delta_angle_to_close_gripper = 30


        return left_all_joint_angles, right_all_joint_angles
    
    def start(self):
        run_threaded_command(self._update_internal_state)

    def stop(self):
        self.running = False

    def _update_internal_state(self):
        self.running = True
        while self.running:
            start_time = time.monotonic()
            left_joints, right_joints = self.get_all_joints_angle()
            self._state['left']['joints'] = left_joints
            self._state['right']['joints'] = right_joints
            elapsed = time.monotonic() - start_time
            sleep_time = max(0, 1/self.update_frequency - elapsed)
            time.sleep(sleep_time)

    
    def get_action(self, obs: TeleopObservation) -> TeleopAction:
        action = self.get_default_action()
        for arm in ['right', 'left']:
            action[arm] = self._state[arm]['joints']
        return action
    
