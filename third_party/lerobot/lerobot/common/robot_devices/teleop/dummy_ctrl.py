import threading
import time
import numpy as np
from typing import Dict
import sys
import os

# Add the path to single_arm module
sys.path.append("/Users/jack/lab_intern/dummy_ctrl")
from single_arm.arm_angle import ArmAngle
import fibre

class TeachFollowArmController:
    """
    机械臂控制器
    专注于实现示教-跟随模式，让follower arm模仿leader arm的动作
    """
    def __init__(self, teach_arm, follow_arm):
        # 初始化线程参数
        self.running = True
        
        # 存储状态
        self.eef_pos = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # 末端位置和姿态
        self.gripper = 0.0  # 夹爪状态
        self.joints = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # 关节角度
        logger = fibre.utils.Logger(verbose=True)
        # teach-follow模式存储变量
        self.teach_arms = teach_arm
        self.follow_arms = follow_arm
        self.joint_offset = np.array([0.0, -73.0, 180.0, 0.0, 0.0, 0.0])  # 关节偏移

        self.teach_hand_init_angle = self.teach_arms.read()['gripper']
        self.follow_hand_init_angle = self.follow_arms.read()['gripper']
        
        print("初始化示教-跟随控制器")
        
        # 启动更新线程
        self.thread = threading.Thread(target=self.update_controller)
        self.thread.start()
        self.arm_controller = ArmAngle(self.teach_arms, self.follow_arms, self.joint_offset)
    
    def set_arms(self, teach_arms, follow_arms):
        """
        设置示教臂和跟随臂
        """
        self.teach_arms = teach_arms
        self.follow_arms = follow_arms
        
        # 初始化机械臂状态
        if self.teach_arms and self.follow_arms:
            try:
                self.teach_hand_init_angle = self.teach_arm.hand.angle
                self.follow_hand_init_angle = self.follow_arm.hand.angle    
            except Exception as e:
                print(f"初始化机械臂状态失败: {e}")
        else:
            print("示教臂或跟随臂未设置，无法启用示教-跟随模式")
    
    def update_controller(self):
        """
        更新控制器状态
        """
        while self.running:
            # 执行示教-跟随控制
            self.update_teach_follow()
            
            # 控制更新频率
            time.sleep(0.02)
    
    def update_teach_follow(self):
        """
        更新示教-跟随控制
        将示教臂的动作传递给跟随臂
        """
        if not self.teach_arms or not self.follow_arms:
            # 如果没有设置示教臂或跟随臂，不执行操作
            return
        
        try:            
            # 获取示教臂当前末端位置
            teach_eef = list(self.teach_arms.read_eef().values())
            
            # 将示教臂的末端位置传递给跟随臂
            
            
            teach_joints = list(self.teach_arms.read().values())
            self.follow_arms.write(teach_joints)

            teach_hand = teach_joints[-1]
            
            # 更新状态以便记录
            self.eef_pos = teach_eef
            self.gripper = teach_hand
            self.joints = teach_joints
            
        except Exception as e:
            print(f"示教-跟随控制错误: {e}")
    
    def get_action(self) -> Dict:
        """
        """
        try:
            # if self.teach_arms:
            #     print("32894829348928489")
            teach_joints = list(self.teach_arms.read().values())
            # print("32894829348928489")
            # print(teach_joints)
            return {
                'joint0': teach_joints[0],
                'joint1': teach_joints[1],
                'joint2': teach_joints[2],
                'joint3': teach_joints[3],
                'joint4': teach_joints[4],
                'joint5': teach_joints[5],
                'gripper': teach_joints[6]
            }
            # else:
            #     print("++++++++++++++++")
        except Exception as e:
            print(f"获取示教动作失败: {e}")
        
        # 回退到保存的状态
        # return {
        #     'joint0': self.joints[0],
        #     'joint1': self.joints[1],
        #     'joint2': self.joints[2],
        #     'joint3': self.joints[3],
        #     'joint4': self.joints[4],
        #     'joint5': self.joints[5],
        #     'gripper': self.gripper
        # }
    
    def get_eef(self) -> Dict:
        """
        获取末端位姿
        """
        try:
            if self.teach_arms:
                teach_pose = self.arm_controller.get_teach_current_pose()
                return {
                    'x': teach_pose[0],
                    'y': teach_pose[1],
                    'z': teach_pose[2],
                    'rx': teach_pose[3],
                    'ry': teach_pose[4],
                    'rz': teach_pose[5]
                }
        except Exception as e:
            print(f"获取示教末端位姿失败: {e}")
        
        # 回退到保存的状态
        return {
            'x': self.eef_pos[0],
            'y': self.eef_pos[1],
            'z': self.eef_pos[2],
            'rx': self.eef_pos[3],
            'ry': self.eef_pos[4],
            'rz': self.eef_pos[5]
        }
    
    def get_teach_follow_state(self) -> Dict:
        """
        获取示教-跟随的状态数据，用于记录
        """
        if not self.teach_arms or not self.follow_arms:
            return {}
        
        try:
            teach_eef = self.arm_controller.get_teach_current_pose()
            follow_eef = self.arm_controller.get_follow_current_pose()
            teach_hand = self.teach_arms.hand.angle
            follow_hand = self.follow_arms.hand.angle
            
            return {
                'teach_eef': teach_eef,
                'follow_eef': follow_eef,
                'teach_hand': teach_hand,
                'follow_hand': follow_hand
            }
        except Exception as e:
            print(f"获取示教-跟随状态失败: {e}")
            return {}
    
    def stop(self):
        """停止控制器"""
        self.running = False
        if hasattr(self, 'thread'):
            self.thread.join()
        print("控制器已停止")

    def reset(self):
        """重置控制器状态"""
        self.joints = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # 6个关节
        self.gripper = 0.0  # 夹爪状态
        self.eef_pos = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # 末端位置和姿态

# 使用示例
if __name__ == "__main__":
    teleop = TeachFollowArmController
    try:
        while True:
            print(f"关节: {teleop.get_action()}")
            print(f"末端: {teleop.get_eef()}")
            time.sleep(0.5)
    except KeyboardInterrupt:
        teleop.stop()