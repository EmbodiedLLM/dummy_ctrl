import time
from typing import Dict
import numpy as np
import fibre
from single_arm.arm_angle import ArmAngle
# from piper_sdk import *
from lerobot.common.robot_devices.motors.configs import DummyMotorsBusConfig

class DummyMotorsBus:
    """
        对Piper SDK的二次封装
    """
    def __init__(self, 
                 config: DummyMotorsBusConfig):
        self.motors = config.motors
        # self.init_joint_position = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] # [6 joints + 1 gripper] * 0.0
        # self.safe_disable_position = [0.0, 0.0, 0.0, 0.0, 0.52, 0.0, 0.0]
        # self.pose_factor = 1000 # 单位 0.001mm
        # self.joint_factor = 57324.840764 # 1000*180/3.14， rad -> 度（单位0.001度）
        self.joint_offset = np.array([0.0,-73.0,180.0,0.0,0.0,0.0])
        self.logger = fibre.utils.Logger(verbose=True)
        self.teach_arm = fibre.find_any(serial_number="208C31875253", logger=self.logger)
        self.follow_arm = fibre.find_any(serial_number="396636713233", logger=self.logger)
        self.arm_controller = ArmAngle(self.teach_arm, self.follow_arm, self.joint_offset)
        self.teach_hand_init_angle = self.teach_arm.robot.hand.angle
        self.follow_hand_init_angle = self.follow_arm.robot.hand.angle
    @property
    def motor_names(self) -> list[str]:
        return list(self.motors.keys())

    @property
    def motor_models(self) -> list[str]:
        return [model for _, model in self.motors.values()]

    @property
    def motor_indices(self) -> list[int]:
        return [idx for idx, _ in self.motors.values()]


    def connect(self, enable:bool) -> bool:
        '''
            使能机械臂并检测使能状态,尝试5s,如果使能超时则退出程序
        '''
        # enable_flag = False
        # loop_flag = False
        # # 设置超时时间（秒）
        # timeout = 5
        # # 记录进入循环前的时间
        # start_time = time.time()
        # while not (loop_flag):
        #     elapsed_time = time.time() - start_time
        #     print(f"--------------------")
        #     enable_list = []
        #     enable_list.append(self.piper.GetArmLowSpdInfoMsgs().motor_1.foc_status.driver_enable_status)
        #     enable_list.append(self.piper.GetArmLowSpdInfoMsgs().motor_2.foc_status.driver_enable_status)
        #     enable_list.append(self.piper.GetArmLowSpdInfoMsgs().motor_3.foc_status.driver_enable_status)
        #     enable_list.append(self.piper.GetArmLowSpdInfoMsgs().motor_4.foc_status.driver_enable_status)
        #     enable_list.append(self.piper.GetArmLowSpdInfoMsgs().motor_5.foc_status.driver_enable_status)
        #     enable_list.append(self.piper.GetArmLowSpdInfoMsgs().motor_6.foc_status.driver_enable_status)
        #     if(enable):
        #         enable_flag = all(enable_list)
        #         self.piper.EnableArm(7)
        #         self.piper.GripperCtrl(0,1000,0x01, 0)
        #     else:
        #         # move to safe disconnect position
        #         enable_flag = any(enable_list)
        #         self.piper.DisableArm(7)
        #         self.piper.GripperCtrl(0,1000,0x02, 0)
        #     print(f"使能状态: {enable_flag}")
        #     print(f"--------------------")
        #     if(enable_flag == enable):
        #         loop_flag = True
        #         enable_flag = True
        #     else: 
        #         loop_flag = False
        #         enable_flag = False
        #     # 检查是否超过超时时间
        #     if elapsed_time > timeout:
        #         print(f"超时....")
        #         enable_flag = False
        #         loop_flag = True
        #         break
        #     time.sleep(0.5)
        # resp = enable_flag
        # print(f"Returning response: {resp}")
        # return resp
        # logger = fibre.utils.Logger(verbose=True)
        # teach_arm = fibre.find_any(serial_number="208C31875253", logger=logger)
        # follow_arm = fibre.find_any(serial_number="396636713233", logger=logger)
        # joint_offset = np.array([0.0,-73.0,180.0,0.0,0.0,0.0])
        self.teach_arm.robot.set_enable(True)
        self.follow_arm.robot.set_enable(True)
        self.logger.info("Moving Teach Arm to Working Pose")
        self.logger.info("Moving Lead Arm to Working Pose")
        self.teach_arm.robot.move_j(0, -30, 90, 0, 70, 0)
        self.follow_arm.robot.move_j(0, -30, 90, 0, 70, 0)
        self.teach_arm.robot.hand.set_mode(0)
        self.teach_arm.robot.hand.set_torque(0)
        self.follow_arm.robot.hand.set_mode(2)
        self.logger.info(f"Teach Gripper mode: {self.teach_arm.robot.hand.get_mode()}, Follow Gripper mode: {self.follow_arm.robot.hand.get_mode()}")
        self.logger.info(f"Teach Gripper angle: {self.teach_arm.robot.hand.angle}, Follow Gripper angle: {self.follow_arm.robot.hand.angle}")
        
        # self.teach_arm.robot.set_enable(False)

    def motor_names(self):
        return

    def set_calibration(self):
        return
    
    def revert_calibration(self):
        return

    def apply_calibration(self):
        """
            移动到初始位置
        """
        self.teach_arm.robot.resting()
        self.follow_arm.robot.resting()

    def write(self, target_joint:list):
        # """
        #     Joint control
        #     - target joint: in radians
        #         joint_1 (float): 关节1角度 (-92000~92000) / 57324.840764
        #         joint_2 (float): 关节2角度 -1300 ~ 90000 / 57324.840764
        #         joint_3 (float): 关节3角度 2400 ~ -80000 / 57324.840764
        #         joint_4 (float): 关节4角度 -90000~90000 / 57324.840764
        #         joint_5 (float): 关节5角度 19000~-77000 / 57324.840764
        #         joint_6 (float): 关节6角度 -90000~90000 / 57324.840764
        #         gripper_range: 夹爪角度 0~0.08
        # """
        # joint_0 = round(target_joint[0]*self.joint_factor)
        # joint_1 = round(target_joint[1]*self.joint_factor)
        # joint_2 = round(target_joint[2]*self.joint_factor)
        # joint_3 = round(target_joint[3]*self.joint_factor)
        # joint_4 = round(target_joint[4]*self.joint_factor)
        # joint_5 = round(target_joint[5]*self.joint_factor)
        # gripper_range = round(target_joint[6]*1000*1000)

        # self.piper.MotionCtrl_2(0x01, 0x01, 100, 0x00) # joint control
        # self.piper.JointCtrl(joint_0, joint_1, joint_2, joint_3, joint_4, joint_5)
        # self.piper.GripperCtrl(abs(gripper_range), 1000, 0x01, 0) # 单位 0.001°

        joint_0 = round(target_joint[0],3)
        joint_1 = round(target_joint[1],3)
        joint_2 = round(target_joint[2],3)
        joint_3 = round(target_joint[3],3)
        joint_4 = round(target_joint[4],3)
        joint_5 = round(target_joint[5],3)  
        gripper_range = round(target_joint[6],3)

        self.teach_arm.robot.move_j(joint_0, joint_1, joint_2, joint_3, joint_4, joint_5)
        self.follow_arm.robot.move_j(joint_0, joint_1, joint_2, joint_3, joint_4, joint_5)
        self.teach_arm.robot.hand.set_angle(gripper_range)
        self.follow_arm.robot.hand.set_angle(self.teach_arm.robot.hand.angle - self.teach_hand_init_angle + self.follow_hand_init_angle)
    

    def read(self) -> Dict:
        """
            - 机械臂关节消息,单位0.001度
            - 机械臂夹爪消息
        """
        # joint_msg = self.piper.GetArmJointMsgs()
        # joint_state = joint_msg.joint_state

        # gripper_msg = self.piper.GetArmGripperMsgs()
        # gripper_state = gripper_msg.gripper_state
        teach_joints = self.arm_controller.get_teach_joints()
        follow_joints = self.arm_controller.get_follow_joints()
        
        return {
            "joint_1": follow_joints[0],
            "joint_2": follow_joints[1],
            "joint_3": follow_joints[2],
            "joint_4": follow_joints[3],
            "joint_5": follow_joints[4],
            "joint_6": follow_joints[5],
            "gripper": follow_joints[6]
        }
    
    def safe_disconnect(self):
        """ 
            Move to safe disconnect position
        """
        self.write(target_joint=self.safe_disable_position)