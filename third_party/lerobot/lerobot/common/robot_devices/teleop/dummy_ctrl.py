
import threading
import time
from typing import Dict
import fibre
from single_arm.arm_angle import ArmAngle
class DummyCtrl:
    def __init__(self):
        
        # # 检查是否有连接的手柄
        # if pygame.joystick.get_count() == 0:
        #     raise Exception("未检测到手柄")
        
        # # 初始化手柄
        # self.joystick = pygame.joystick.Joystick(0)
        # self.joystick.init()
        
        # # 初始化关节和夹爪状态
        # self.joints = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # 6个关节
        # self.gripper = 0.0  # 夹爪状态
        # self.speeds = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # 6个关节的速度
        # self.gripper_speed = 0.0  # 夹爪速度
        
        # # 定义关节弧度限制（计算好的范围）
        # self.joint_limits = [
        #     (-92000 / 57324.840764, 92000 / 57324.840764),  # joint1
        #     (-1300 / 57324.840764, 90000 / 57324.840764),   # joint2
        #     (-80000 / 57324.840764, 0 / 57324.840764),   # joint3
        #     (-90000 / 57324.840764, 90000 / 57324.840764),  # joint4
        #     (-77000 / 57324.840764, 19000 / 57324.840764),  # joint5
        #     (-90000 / 57324.840764, 90000 / 57324.840764)   # joint6
        # ]

        # # 启动更新线程
        # self.running = True
        # self.thread = threading.Thread(target=self.update_joints)
        # self.thread.start()

        self.teach_arm = fibre.find_any(serial_number="208C31875253", logger=self.logger)
        self.follow_arm = fibre.find_any(serial_number="396636713233", logger=self.logger)
        self.arm_controller = ArmAngle(self.teach_arm, self.follow_arm, self.joint_offset)
        self.teach_hand_init_angle = self.teach_arm.robot.hand.angle
        self.follow_hand_init_angle = self.follow_arm.robot.hand.angle
        self.teach_arm.robot.hand.set_mode(0)
        self.teach_arm.robot.hand.set_torque(0)
        self.follow_arm.robot.hand.set_mode(2)
        self.running = True
        self.thread = threading.Thread(target=self.update_joints)
        self.thread.start()

    def update_joints(self):
        while self.running:
            # 处理事件队列

            
            # 控制更新频率
            time.sleep(0.02)
    
    def get_action(self) -> Dict:
        # 返回机械臂的当前状态
        return {
            'joint0': self.teach_arm.robot.joints[0],
            'joint1': self.teach_arm.robot.joints[1],
            'joint2': self.teach_arm.robot.joints[2],
            'joint3': self.teach_arm.robot.joints[3],
            'joint4': self.teach_arm.robot.joints[4],
            'joint5': self.teach_arm.robot.joints[5],
            'gripper': self.teach_arm.robot.hand.angle
        }
    
    def stop(self):
        # 停止更新线程
        self.running = False
        self.thread.join()
        print("Gamepad exits")

    def reset(self):
        self.teach_arm.robot.move_j(0, -30, 90, 0, 70, 0)
        self.follow_arm.robot.move_j(0, -30, 90, 0, 70, 0)

# 使用示例
if __name__ == "__main__":
    arm_controller = DummyCtrl()
    try:
        while True:
            print(arm_controller.get_action())
            time.sleep(0.1)
    except KeyboardInterrupt:
        arm_controller.stop()