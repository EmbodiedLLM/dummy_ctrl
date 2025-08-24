#%%
from __future__ import print_function
import sys
sys.path.append("..")
sys.path.append("single_arm")
import fibre
from pynput import keyboard
import numpy as np
import time
import threading
# Replace with the modified LeRobotDataCollector
from single_arm.real_collector import LeRobotDataCollector
from single_arm.arm_angle import ArmAngle
from single_arm.timing_utils import precise_sleep
from single_arm.dm_auto_recv import MotorStateCache, PositionUpdaterThread, AsyncMotorControl
import cv2
# logger verbose=True
logger = fibre.utils.Logger(verbose=True)
# %%
from datetime import datetime
timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")

# Visualization flag
ENABLE_VISUALIZATION = False

# Use the modified data collector with both cameras
data_collector = LeRobotDataCollector(
    output_dir=f"/Users/yinzi/dummy_ctrl/data/pick_place_{timestamp_str}",
    fps=10,
    camera_urls={
        # "cam_wrist": "http://192.168.237.100:8080/?action=stream",
        "cam_wrist": "0",
        # "cam_head": "http://192.168.237.157:8080/?action=stream"
        "cam_head": "1"
    },
    robot_type="thu_dummy_arm",
    use_video=True
)
#%%
teach_arm_SN = "3950366E3233"
follow_arm_SN = "396636713233"
# %%
teach_arm = fibre.find_any(serial_number=teach_arm_SN, logger=logger)
#%%
follow_arm = fibre.find_any(serial_number=follow_arm_SN, logger=logger)
#%%
teach_arm.robot.set_enable(True)
follow_arm.robot.set_enable(True)
logger.info("Moving Teach Arm to Resting Pose")
logger.info("Moving Follow Arm to Resting Pose")
teach_arm.robot.move_j(0, -90, 90, 0, 70, 0)
joint_offset = np.array([0.0,-73.0,180.0,0.0,0.0,0.0])
# %%
teach_arm.robot.set_enable(True)
follow_arm.robot.set_enable(True)
logger.info("Moving Teach Arm to Working Pose")
logger.info("Moving Follow Arm to Working Pose")
teach_arm.robot.move_j(0, -30, 90, 0, 70, 0)
follow_arm.robot.move_j(0, -30, 90, 0, 70, 0)
#%%
import serial
from DM_CAN import MotorControl, Motor, DM_Motor_Type, Control_Type
logger.info("Init Teach Arm Hand")
SERIAL_PORT = '/dev/tty.usbmodem00000000050C1' 
logger.info(f"Using serial port: {SERIAL_PORT}")
ser = serial.Serial(SERIAL_PORT, 921600, timeout=0.5)
teach_hand_motor =Motor(DM_Motor_Type.DMH3510,0x37, 0x47)
teach_hand_motor_control = MotorControl(ser)
teach_hand_motor_control.addMotor(teach_hand_motor)
if teach_hand_motor_control.switchControlMode(teach_hand_motor,Control_Type.MIT):
    logger.info("switch MIT success for teach_hand_motor")
else:
    logger.error("switch MIT failed for teach_hand_motor")
    raise RuntimeError("Not Detected Teach Hand Motor")
logger.info("Setting Teach Arm Hand to Zero")
#%%
teach_hand_motor_control.enable(teach_hand_motor)
teach_hand_motor_control.set_zero_position(teach_hand_motor)
#%%
# 初始化异步接收系统
gripper_cache = MotorStateCache([teach_hand_motor.MasterID])
gripper_motor_map = {teach_hand_motor.MasterID: teach_hand_motor}
gripper_zero_positions = {teach_hand_motor.MasterID: 0.0}  # 刚刚设置了零位
gripper_updater = PositionUpdaterThread(
    teach_hand_motor_control, 
    gripper_motor_map, 
    gripper_cache, 
    gripper_zero_positions, 
    frequency=25  # 与主控制循环同频
)
gripper_updater.start()
logger.info("Started gripper position updater thread")

# 初始化异步控制器，带时间戳调度（50ms过时阈值适合25Hz控制频率）
async_gripper_control = AsyncMotorControl(teach_hand_motor_control, max_command_age=0.05)
logger.info("Started async gripper control with timestamp scheduling")
#%%
logger.info("Setting Follow Arm Hand to Zero")
follow_arm.robot.hand.set_enable(True)
follow_arm.robot.hand.set_zero()
#%%
# 夹爪MIT控制参数
## 这两个参数等于是位置跟随
kp_gripper = 0.8
kd_gripper = 0.05
# 力反馈尺度
force_scale = 0.8
#%%
arm_controller = ArmAngle(teach_arm, follow_arm, joint_offset)

# Use thread-safe method to listen for keyboard input
stop = False

# Define keyboard listener function
def on_press(key):
    global stop
    try:
        if key == keyboard.Key.shift_r:
            print("\nDetected Shift, stopping loop...")
            stop = True
            return False
    except AttributeError:
        pass

# Start keyboard listener in a separate thread
def start_keyboard_listener():
    with keyboard.Listener(on_press=on_press) as listener:
        listener.join()

# Start keyboard listening thread
keyboard_thread = threading.Thread(target=start_keyboard_listener)
keyboard_thread.daemon = True  # Set as daemon thread, will end automatically when main thread ends
keyboard_thread.start()

# Add another way to stop: press Enter key
def check_input():
    global stop
    print("Press right Shift key or Enter key to stop data collection...")
    while not stop:
        try:
            input()  # Wait for any input
            print("Enter key detected, stopping loop...")
            stop = True
        except:
            pass
        time.sleep(0.1)  # Short sleep to reduce CPU load

# Start input listening thread
input_thread = threading.Thread(target=check_input)
input_thread.daemon = True
input_thread.start()

import time
rate = 0.02  # 50Hz
# rate = 0.1 # 10Hz
# rate = 0.05 # 20 Hz
# rate = 0.04 # 25 Hz

# Start data collection
# data_collector.start_episode(task="pick the purple cube into the box")
data_collector.start_episode(task="pick the green cube into the box")

print("Starting data collection, press right Shift key to stop...")

follow_hand = follow_arm.robot.hand
teach_hand = teach_arm.robot.hand
teach_arm.robot.set_enable(False)
RAD2DEG = 180.0 / np.pi
teach_arm.robot.set_enable(False)
while not stop:
    start_time = time.time()
    
    # Step 1: Capture the robot state (teach and follow joints)
    teach_joints = arm_controller.get_teach_joints()
    print(f"Teach joints: {teach_joints}")
    follow_arm.robot.move_j(*teach_joints)

    # 夹爪跟随位置 - 使用异步缓存
    gripper_state = gripper_cache.get_state(teach_hand_motor.MasterID)
    if gripper_state and gripper_state['status'] == 'connected':
        gripper_pos = gripper_state['pos_rel']  # 相对位置（弧度）
        logger.info(f"Gripper position: {gripper_pos}")
        follow_hand.control_mit(kp_gripper, kd_gripper, gripper_pos, 0, 0)
        
        # 读取夹爪接触力并施加力反馈到剪刀
        gripper_force = follow_hand.torque
        logger.info(f"gripper_force: {gripper_force}")
        feedback_torque = -gripper_force * force_scale
        # 前面参数全0代表纯力矩模式 - 使用异步控制API
        async_gripper_control.control_mit_async(teach_hand_motor, 0, 0, 0, 0, feedback_torque)
    else:
        logger.warning("Gripper disconnected, skipping gripper control")

    follow_joints = arm_controller.get_follow_joints()
    # Step 2: Collect data (robot state + camera frames)
    data_collector.collect_step(
        teach=teach_joints,
        follow=follow_joints,
        teach_gripper=gripper_pos * RAD2DEG,
        follow_gripper=arm_controller.get_follow_hand_position() * RAD2DEG
    )
    
    # Visualization of latest camera frames
    if ENABLE_VISUALIZATION:
        latest_frames = data_collector.get_latest_frames()
        if latest_frames:
            for cam_name, frame in latest_frames.items():
                if frame is not None:
                    cv2.imshow(f"{cam_name}_frame", frame)
            cv2.waitKey(1)  # Non-blocking wait for key press
    
    # # Log progress every 10 frames
    if data_collector.frame_count % 10 == 0:
        print(f"Frames collected: {data_collector.frame_count}")
        
        # 显示异步控制统计信息
        stats = async_gripper_control.get_stats()
        print(f"Async control - Queue size: {stats['queue_size']}")
    
    # Sleep precisely to maintain target frequency
    elapsed = time.time() - start_time
    if elapsed < rate:
        precise_sleep(rate - elapsed)
    
    # Calculate and print actual loop frequency
    loop_time = time.time() - start_time
    actual_freq = 1.0 / loop_time if loop_time > 0 else 0
    if abs(loop_time - 0.1) > 0.01:  # Allow 10ms deviation
        print(f"Timing deviation: {loop_time:.3f}s (target: 0.1s)")
# %
teach_arm.robot.set_enable(True)
follow_arm.robot.set_enable(True)
# teach_arm.robot.resting()
# follow_arm.robot.resting()
teach_arm.robot.move_j(0, -30, 90, 0, 70, 0)
follow_arm.robot.move_j(0, -30, 90, 0, 70, 0)
# follow_hand.control_mit(kp_gripper, kd_gripper, 0, 0, 0)
# teach_hand_motor_control.controlMIT(teach_hand_motor, kp_gripper, kd_gripper, 0, 0, 0)
# %%
print("Program ended, saving data...")

# Close OpenCV windows
if ENABLE_VISUALIZATION:
    cv2.destroyAllWindows()
data_collector.save_episode()
print("Data collection completed")
 #%%
teach_arm.robot.resting()
follow_arm.robot.resting()

# %%
