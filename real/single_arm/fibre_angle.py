#%%
from __future__ import print_function
import sys
sys.path.append("..")
sys.path.append("single_arm")
#%% 示教夹爪电机(使用UnifiedCANThread)
import serial
from DM_CAN import MotorControl, Motor, DM_Motor_Type, Control_Type
SERIAL_PORT = '/dev/tty.usbmodem00000000050C1' 
ser = serial.Serial(SERIAL_PORT, 921600, timeout=0.5)
teach_hand_motor = Motor(DM_Motor_Type.DMH3510, 0x37, 0x47)
teach_hand_motor_control = MotorControl(ser)
teach_hand_motor_control.addMotor(teach_hand_motor)
if teach_hand_motor_control.switchControlMode(teach_hand_motor, Control_Type.MIT):
    print("switch MIT success for teach_hand_motor")
else:
    print("switch MIT failed for teach_hand_motor")
    raise RuntimeError("Not Detected Teach Hand Motor")
print("Setting Teach Arm Hand to Zero")
#%% 启动UnifiedCANThread (独立100Hz频率)
# 初始化统一CAN通信系统
from single_arm.dm_auto_recv import MotorStateCache, UnifiedCANThread
gripper_cache = MotorStateCache([teach_hand_motor.SlaveID])
gripper_motor_map = {teach_hand_motor.SlaveID: teach_hand_motor}
gripper_zero_positions = {teach_hand_motor.SlaveID: 0.0}  # 将在下一步设置零位

# 启动UnifiedCANThread，独立的100Hz频率
unified_can_thread = UnifiedCANThread(
    motor_control=teach_hand_motor_control,
    motor_map=gripper_motor_map,
    cache=gripper_cache,
    zero_positions=gripper_zero_positions,
    frequency=100,  # 独立的100Hz频率
    max_command_age=0.01  # 10ms命令超时
)
unified_can_thread.start()
#%% 设夹爪零位
# 提示用户输入回车开始校准夹爪零位
input("Press Enter to calibrate gripper zero position...")
teach_hand_motor_control.enable(teach_hand_motor)
teach_hand_motor_control.set_zero_position(teach_hand_motor)
#%%
import sys
sys.path.append('..')
import fibre
import numpy as np
import cv2
import time
from collections import deque

from __future__ import print_function
logger = fibre.utils.Logger(verbose=True)
#%%
teach_arm = fibre.find_any(serial_number="3950366E3233", logger=logger)
#%%
follow_arm = fibre.find_any(serial_number="396636713233", logger=logger)
#%%
def get_all_joint_angles(arm_interface):
    joint_offset = np.array([0,-73,180,0,0,0])
    all_joint_angles = arm_interface.robot.joint_1.angle, arm_interface.robot.joint_2.angle, arm_interface.robot.joint_3.angle, arm_interface.robot.joint_4.angle, arm_interface.robot.joint_5.angle, arm_interface.robot.joint_6.angle
    all_joint_angles = np.round(np.array(all_joint_angles))
    all_joint_angles = all_joint_angles + joint_offset
    return all_joint_angles

class DualArmRealTimePlotter:
    def __init__(self, width=1600, height=600, buffer_size=300):
        self.width = width
        self.height = height
        self.buffer_size = buffer_size
        self.joint_names = ['J1', 'J2', 'J3', 'J4', 'J5', 'J6']
        self.colors = [(255,0,0), (0,255,0), (0,0,255), (255,255,0), (255,0,255), (0,255,255)]
        # Teach arm buffers
        self.teach_data_buffers = [deque(maxlen=buffer_size) for _ in range(6)]
        # Follow arm buffers
        self.follow_data_buffers = [deque(maxlen=buffer_size) for _ in range(6)]
        self.time_buffer = deque(maxlen=buffer_size)
        
    def add_data(self, teach_joint_angles, follow_joint_angles, timestamp):
        for i, angle in enumerate(teach_joint_angles):
            self.teach_data_buffers[i].append(angle)
        for i, angle in enumerate(follow_joint_angles):
            self.follow_data_buffers[i].append(angle)
        self.time_buffer.append(timestamp)
    
    def draw_plot(self):
        img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        if len(self.time_buffer) < 2:
            return img
        
        # Split screen: left for teach arm, right for follow arm
        plot_width = self.width // 2 - 120
        plot_height = self.height - 130
        margin_left = 80
        margin_top = 50
        
        # Get angle range for both arms
        all_angles = []
        for buffer in self.teach_data_buffers + self.follow_data_buffers:
            if buffer:
                all_angles.extend(buffer)
        
        if not all_angles:
            return img
        
        min_angle, max_angle = min(all_angles), max(all_angles)
        angle_range = max_angle - min_angle if max_angle != min_angle else 1
        
        # Draw left plot (Teach Arm)
        self._draw_single_plot(img, self.teach_data_buffers, margin_left, margin_top, 
                              plot_width, plot_height, min_angle, max_angle, angle_range, "Teach Arm")
        
        # Draw right plot (Follow Arm)
        self._draw_single_plot(img, self.follow_data_buffers, margin_left + self.width // 2, margin_top, 
                              plot_width, plot_height, min_angle, max_angle, angle_range, "Follow Arm")
        
        # Draw shared legend at bottom
        legend_start_x = self.width // 2 - 360
        legend_start_y = self.height - 60
        for i, (name, color) in enumerate(zip(self.joint_names, self.colors)):
            x = legend_start_x + i * 120
            cv2.rectangle(img, (x, legend_start_y), (x + 15, legend_start_y + 15), color, -1)
            cv2.putText(img, name, (x + 20, legend_start_y + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return img
    
    def _draw_single_plot(self, img, data_buffers, start_x, start_y, width, height, 
                         min_angle, max_angle, angle_range, title):
        # Draw plot border
        cv2.rectangle(img, (start_x, start_y), (start_x + width, start_y + height), (100, 100, 100), 1)
        
        # Draw grid lines
        for i in range(1, 5):
            y = start_y + int(i * height / 4)
            cv2.line(img, (start_x, y), (start_x + width, y), (50, 50, 50), 1)
            
            angle_val = max_angle - i * angle_range / 4
            cv2.putText(img, f'{angle_val:.1f}°', (start_x - 70, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        for i in range(0, len(self.time_buffer), max(1, len(self.time_buffer) // 10)):
            x = start_x + int(i * width / max(1, len(self.time_buffer) - 1))
            cv2.line(img, (x, start_y), (x, start_y + height), (50, 50, 50), 1)
        
        # Draw joint angle curves
        for joint_idx in range(6):
            if len(data_buffers[joint_idx]) < 2:
                continue
            
            points = []
            for i, angle in enumerate(data_buffers[joint_idx]):
                x = start_x + int(i * width / max(1, len(data_buffers[joint_idx]) - 1))
                y = start_y + int((max_angle - angle) * height / angle_range)
                points.append((x, y))
            
            for i in range(len(points) - 1):
                cv2.line(img, points[i], points[i + 1], self.colors[joint_idx], 2)
        
        # Draw title
        cv2.putText(img, title, (start_x + width // 2 - 60, start_y - 15), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Draw axis labels
        cv2.putText(img, 'Angle (°)', (start_x - 75, start_y + height // 2), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(img, 'Time', (start_x + width // 2 - 20, start_y + height + 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
# %%
# Work Pose
teach_arm.robot.set_enable(True)
teach_arm.robot.move_j(0, 0, 90, 0, 55, 0)
follow_arm.robot.set_enable(True)
follow_arm.robot.move_j(0, 0, 90, 0, 55, 0)
#%%
plotter = DualArmRealTimePlotter()
teach_arm.robot.set_enable(False)
frequency = 10
start_time = time.time()
kp_gripper = 0.8
kd_gripper = 0.05
# 力反馈尺度
force_scale = 0.8

follow_arm.robot.hand.set_enable(True)
while True:
    teach_joint_angles = get_all_joint_angles(teach_arm)
    follow_joint_angles = get_all_joint_angles(follow_arm)
    current_time = time.time() - start_time
    gripper_state = gripper_cache.get_state(teach_hand_motor.SlaveID)
    teach_gripper_pos = gripper_state['pos_abs_rad']
    
    plotter.add_data(teach_joint_angles, follow_joint_angles, current_time)
    
    img = plotter.draw_plot()
    cv2.imshow('Dual Arm Joint Angles Real-time Plot', img)
    
    print(f"Teach Angles: {teach_joint_angles}")
    print(f"Follow Angles: {follow_joint_angles}")
    follow_arm.robot.move_j(*teach_joint_angles)
    follow_arm.robot.hand.control_mit(kp_gripper, kd_gripper, teach_gripper_pos, 0, 0)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
    time.sleep(1/frequency)

cv2.destroyAllWindows()
#%%

#%%
teach_arm.robot.resting()
# %%
follow_arm.robot.resting()

# %%
