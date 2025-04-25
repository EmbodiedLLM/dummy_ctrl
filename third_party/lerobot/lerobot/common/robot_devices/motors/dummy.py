import time
from typing import Dict
import numpy as np

from lerobot.common.robot_devices.motors.configs import DummyMotorsBusConfig

class DummyMotorsBus:
    """
    Dummy机械臂电机控制接口
    基于fibre库的接口进行封装，适配LeRobot框架
    """
    def __init__(self, 
                 config: DummyMotorsBusConfig):
        self.config = config
        self.port = self.config.port
        self.motors = self.config.motors
        self.init_joint_position = [0.0, -30.0, 90.0, 0.0, 70.0, 0.0]  # [
        self.safe_disable_position = [0.0, -30.0, 90.0, 0.0, 70.0, 0.0]  # 安全位置
        self.current_position = self.safe_disable_position.copy()
        self.joint_offset = np.array([0.0, -73.0, 180.0, 0.0, 0.0, 0.0])
        
        # 模拟连接状态
        self.is_connected = False
        
        try:
            # 导入fibre库
            import fibre.utils
            import fibre
            # 导入logger
            logger = fibre.utils.Logger(verbose=True)
            # 连接机械臂
            self.arm = fibre.find_any(serial_number=self.port, logger=logger)
            print(f"Connected to arm with serial: {self.port}")
            self.is_connected = True
            
            # 记录初始夹爪位置
            # if hasattr(self.arm.robot, "hand") and hasattr(self.arm.robot.hand, "angle"):
            #     if self.port == "208C31875253":  # 示教臂
            #         print("1111111111")
            self.teach_hand_init_angle = -179.03
                # else:  # 跟随臂
                #     print("222222222")
            self.follow_hand_init_angle = -128.99
            
        except Exception as e:
            print(f"Failed to connect to arm: {e}")
            self.is_connected = False

    @property
    def motor_names(self) -> list[str]:
        return list(self.motors.keys())

    @property
    def motor_models(self) -> list[str]:
        return [model for _, model in self.motors.values()]

    @property
    def motor_indices(self) -> list[int]:
        return [idx for idx, _ in self.motors.values()]

    def connect(self, enable: bool) -> bool:
        """
        使能机械臂并检测使能状态
        """
        
        try:
            self.arm.robot.set_enable(enable)
            # 移动到安全位置
            # self.arm.robot.move_j(*self.safe_disable_position)  # 不包括夹爪
            self.arm.robot.resting()
            print(f"Enabling arm {self.port}")
            
            return True
        except Exception as e:
            print(f"Failed to {'enable' if enable else 'disable'} arm: {e}")
            return False
    
    def enable_false(self):
        self.arm.robot.set_enable(False)
        
    def set_calibration(self):
        # 设置校准参数
        return
    
    def revert_calibration(self):
        # 恢复默认校准参数
        return

    def apply_calibration(self):
        """
        移动到初始位置
        """
        self.write(target_joint=self.init_joint_position)

    def write(self, target_joint: list):
        """
        关节控制
        - target_joint: 弧度制，[joint_1, joint_2, ..., joint_6, gripper]
        """
        # try:
        print("101010101010")
        print(target_joint)
        # 移动机械臂关节
        self.arm.robot.move_j(
            target_joint[0],
            target_joint[1], 
            target_joint[2], 
            target_joint[3], 
            target_joint[4], 
            target_joint[5]
        )
        # 控制夹爪
        if len(target_joint) > 6:
            self.arm.robot.hand.set_angle(target_joint[6] - self.teach_hand_init_angle + self.follow_hand_init_angle)
        
        # 更新当前位置
        self.current_position = target_joint.copy()
        # except Exception as e:
        #     print(f"Failed to move joints: {e}")

    def write_eef(self, target_eef: list):
        """
        末端控制
        - target_eef: [x, y, z, rx, ry, rz]，单位为mm和度
        """

        try:
            # 末端控制
            self.arm.robot.move_l(
                target_eef[0],
                target_eef[1],
                target_eef[2],
                target_eef[3],
                target_eef[4],
                target_eef[5]
            )
            self.arm.robot.hand.set_angle(target_eef[6] - self.teach_hand_init_angle + self.follow_hand_init_angle)
            print(f"Moving to EEF position: {target_eef}")
        except Exception as e:
            print(f"Failed to move to EEF position: {e}")

    def read(self) -> Dict:
        """
        读取当前关节角度和夹爪状态
        返回值包含关节和夹爪的当前角度
        """
        # 读取当前关节角度
        current_angles = self.get_current_joints()
        gripper_angle = self.arm.robot.hand.angle
        
        # 更新当前位置
        self.current_position = [
            current_angles[0],
            current_angles[1],
            current_angles[2],
            current_angles[3],
            current_angles[4],
            current_angles[5],
            gripper_angle
        ]
        
        return {
            "joint_1": current_angles[0],
            "joint_2": current_angles[1],
            "joint_3": current_angles[2],
            "joint_4": current_angles[3],
            "joint_5": current_angles[4],
            "joint_6": current_angles[5],
            "gripper": gripper_angle,
        }
    
    def get_current_joints(self) -> list:
        """
        获取当前关节角度
        """
        try:
            # 读取当前关节角度，根据fibre库API调整
            return np.array([
            self.arm.robot.joint_1.angle,
            self.arm.robot.joint_2.angle,
            self.arm.robot.joint_3.angle,
            self.arm.robot.joint_4.angle,
            self.arm.robot.joint_5.angle,
            self.arm.robot.joint_6.angle,
        ])+ self.joint_offset
        except Exception as e:
            print(f"Failed to get current joints: {e}")
            return self.current_position[:6]
    
    def read_eef(self) -> Dict:
        """
        读取当前末端位置
        返回值包含末端位置和姿态，单位为mm和度
        """
        
        try:
            # 读取当前末端位置
            current_pose = self.get_current_pose()
            
            return {
                "x": current_pose[0],
                "y": current_pose[1],
                "z": current_pose[2],
                "rx": current_pose[3],
                "ry": current_pose[4],
                "rz": current_pose[5],
                "gripper": current_pose[6],
            }
        except Exception as e:
            print(f"Failed to read EEF position: {e}")
            return {
                "x": 200.0,
                "y": 0.0,
                "z": 300.0,
                "rx": 0.0,
                "ry": 0.0,
                "rz": 0.0,
            }
    
    def get_current_pose(self) -> list:
        """
        获取当前末端位置
        """
        
        try:
            # 读取当前末端位置，根据fibre库API调整
            return [
                self.arm.robot.eef_pose.x, 
                self.arm.robot.eef_pose.y, 
                self.arm.robot.eef_pose.z, 
                self.arm.robot.eef_pose.a, 
                self.arm.robot.eef_pose.b, 
                self.arm.robot.eef_pose.c, 
                self.arm.robot.hand.angle,
            ]
        except Exception as e:
            print(f"Failed to get current pose: {e}")
            return [200.0, 0.0, 300.0, 0.0, 0.0, 0.0]
    
    def safe_disconnect(self):
        """
        安全断开连接，先移动到安全位置再断开
        """
        self.write(target_joint=self.safe_disable_position)
        time.sleep(1)  # 等待移动完成
        self.connect(False) 