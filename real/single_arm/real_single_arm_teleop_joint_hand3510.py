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
from single_arm.real_collector import LeRobotDataCollector
from single_arm.arm_angle import ArmAngle
from single_arm.timing_utils import precise_sleep, precise_wait
from single_arm.dm_auto_recv import MotorStateCache, UnifiedCANThread
from single_arm.camera_capture import CameraCapture
import cv2
# logger verbose=True
logger = fibre.utils.Logger(verbose=True)
teach_arm_SN = "3950366E3233"
follow_arm_SN = "396636713233"
ctrl_frequency = 20 # 20 Hz
data_frequency = 10 # 10 Hz
# 夹爪MIT控制参数
## 这两个参数等于是位置跟随
kp_gripper = 0.8
kd_gripper = 0.05
# 力反馈尺度
force_scale = 0.8
# %%
teach_arm = fibre.find_any(serial_number=teach_arm_SN, logger=logger)
#%%
follow_arm = fibre.find_any(serial_number=follow_arm_SN, logger=logger)
#%% 机械臂初始化位置
teach_arm.robot.set_enable(True)
follow_arm.robot.set_enable(True)
logger.info("Moving Teach Arm to Resting Pose")
logger.info("Moving Follow Arm to Resting Pose")
teach_arm.robot.move_j(0, -90, 90, 0, 70, 0)
follow_arm.robot.move_j(0, -90, 90, 0, 70, 0)
joint_offset = np.array([0.0,-73.0,180.0,0.0,0.0,0.0])
# %% 机械臂工作位置
teach_arm.robot.set_enable(True)
follow_arm.robot.set_enable(True)
logger.info("Moving Teach Arm to Working Pose")
logger.info("Moving Follow Arm to Working Pose")
teach_arm.robot.move_j(0, -30, 90, 0, 70, 0)
follow_arm.robot.move_j(0, -30, 90, 0, 70, 0)
#%% 示教夹爪电机(使用UnifiedCANThread)
import serial
from DM_CAN import MotorControl, Motor, DM_Motor_Type, Control_Type
logger.info("Init Teach Arm Hand with UnifiedCANThread")
SERIAL_PORT = '/dev/tty.usbmodem00000000050C1' 
logger.info(f"Using serial port: {SERIAL_PORT}")
ser = serial.Serial(SERIAL_PORT, 921600, timeout=0.5)
teach_hand_motor = Motor(DM_Motor_Type.DMH3510, 0x37, 0x47)
teach_hand_motor_control = MotorControl(ser)
teach_hand_motor_control.addMotor(teach_hand_motor)
if teach_hand_motor_control.switchControlMode(teach_hand_motor, Control_Type.MIT):
    logger.info("switch MIT success for teach_hand_motor")
else:
    logger.error("switch MIT failed for teach_hand_motor")
    raise RuntimeError("Not Detected Teach Hand Motor")
logger.info("Setting Teach Arm Hand to Zero")
#%% 启动UnifiedCANThread (独立100Hz频率)
# 初始化统一CAN通信系统
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
logger.info("Started Teach Gripper UnifiedCANThread at 100Hz")
#%% 设夹爪零位
# 提示用户输入回车开始校准夹爪零位
input("Press Enter to calibrate gripper zero position...")
logger.info("Setting Teach Arm Hand to Zero")
teach_hand_motor_control.enable(teach_hand_motor)
teach_hand_motor_control.set_zero_position(teach_hand_motor)
# teach_arm.robot.hand.set_enable(True)
# teach_arm.robot.hand.set_zero()
logger.info("Setting Follow Arm Hand to Zero")
follow_arm.robot.hand.set_enable(True)
follow_arm.robot.hand.set_zero()

#%%
from datetime import datetime
date_str = datetime.now().strftime("%Y-%m-%d")
datetime_str = datetime.now().strftime("%Y-%m-%d_%H%M%S")
data_collector = LeRobotDataCollector(
    output_dir=f"/Users/yinzi/dummy_ctrl/data/{date_str}/pick_place_greencube_{datetime_str}",
    fps=data_frequency, # 10Hz
    robot_type="dummy_arm_inz",
    use_video=True,
    task="pick the green cube into the box"
)
#%% 开启相机
import cv2
import matplotlib.pyplot as plt

# 使用新的CameraCapture类
camera_head_uri = 0
camera_wrist_uri = 1

camera_head = CameraCapture(camera_head_uri, "head", fps=30.0)
camera_wrist = CameraCapture(camera_wrist_uri, "wrist", fps=30.0)

# 启动摄像头
if not camera_head.start():
    raise RuntimeError("Failed to start camera head")
if not camera_wrist.start():
    raise RuntimeError("Failed to start camera wrist")

# 等待摄像头稳定
time.sleep(2.0)

# 获取测试帧
test_head_result = camera_head.get_latest_frame()
test_wrist_result = camera_wrist.get_latest_frame()

if test_head_result is None or test_wrist_result is None:
    raise RuntimeError("Failed to get initial frames from cameras")

_, camera_head_frame = test_head_result
_, camera_wrist_frame = test_wrist_result

# 显示测试帧
plt.figure(figsize=(8,4))
plt.subplot(1,2,1)
plt.imshow(cv2.cvtColor(camera_head_frame, cv2.COLOR_BGR2RGB))
plt.title("Camera Head")
plt.axis('off')
plt.subplot(1,2,2)
plt.imshow(cv2.cvtColor(camera_wrist_frame, cv2.COLOR_BGR2RGB))
plt.title("Camera Wrist")
plt.axis('off')
plt.show()

logger.info("Cameras initialized successfully")
arm_controller = ArmAngle(teach_arm, follow_arm, joint_offset)
logger.info("Preparing initial joint states")
init_teach_joints = arm_controller.get_teach_joints()
init_follow_joints = arm_controller.get_follow_joints()
assert np.allclose(init_teach_joints, init_follow_joints, atol=0.1), f"Initial joint states do not match: teach={init_teach_joints}, follow={init_follow_joints}"
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

stop = False
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
follow_hand = follow_arm.robot.hand
teach_hand = teach_arm.robot.hand
teach_arm.robot.set_enable(False)
RAD2DEG = 180.0 / np.pi
teach_arm.robot.set_enable(False)
t_start_wall = time.monotonic()
iter_idx = 0
dt = 1/ctrl_frequency
# command_latency  = 0.01 # Latency between receiving command to executing on Robot in Sec.
print("Starting data collection, press right Shift key to stop...")
data_collector.start_episode(task="pick the green cube into the box")
while not stop:
    # calculate timing
    t_cycle_end = t_start_wall + (iter_idx + 1) * dt
    
    # Step 1: Get Obs
    current_time = time.time()
    
    # 使用时间戳获取匹配的帧
    camera_head_frame = camera_head.get_frame_by_timestamp(current_time, tolerance=dt/2)
    camera_wrist_frame = camera_wrist.get_frame_by_timestamp(current_time, tolerance=dt/2)
    
    if camera_head_frame is None or camera_wrist_frame is None:
        ## 如果没找到匹配的帧就直接报错
        ## raise ValueError("No matching timestamp frame found")
        # 如果没找到匹配的帧，使用最新帧
        head_result = camera_head.get_latest_frame()
        wrist_result = camera_wrist.get_latest_frame()
        
        if head_result is None or wrist_result is None:
            logger.error("Camera capture failed - no frames available")
            # 检查摄像头状态
            head_status = camera_head.get_status()
            wrist_status = camera_wrist.get_status()
            logger.error(f"Head camera: {head_status}")
            logger.error(f"Wrist camera: {wrist_status}")
            stop = True
            break
            
        _, camera_head_frame = head_result
        _, camera_wrist_frame = wrist_result

    follow_joints = arm_controller.get_follow_joints()
    arm_controller.refresh_follow_hand()
    obs = {
        "cam_head": camera_head_frame,
        "cam_wrist": camera_wrist_frame,
        "joint_states": follow_joints,
        "gripper_pos_deg": follow_hand.position * RAD2DEG,
        "gripper_torque": follow_hand.torque
    }
    logger.debug(f"Obs collected at {time.monotonic() - t_start_wall:.4f}s")

    # Step 2: Get Action
    # Get Teach Joint States
    teach_joints = arm_controller.get_teach_joints() # 角度
    gripper_state = gripper_cache.get_state(teach_hand_motor.SlaveID)
    if gripper_state is None:
        raise RuntimeError(f"Gripper state not found for SlaveID {teach_hand_motor.SlaveID:#x}")
    if gripper_state.get('status') == 'disconnected':
        raise RuntimeError(f"Gripper motor disconnected, last seen: {gripper_state['last_seen']}")
    teach_gripper_pos = gripper_state['pos_abs_rad'] # 弧度
    # teach_gripper_pos = arm_controller.get_teach_hand_position()
    action = {
        "joint_states": teach_joints,
        "gripper_pos_deg": teach_gripper_pos * RAD2DEG
    }

    # Step3: Exec Action
    follow_arm.robot.move_j(*teach_joints) # 角度
    # 夹爪跟随位置
    # logger.debug(f"Gripper position: {teach_gripper_pos} at {time.monotonic() - t_start_wall:.4f}s")
    follow_hand.control_mit(kp_gripper, kd_gripper, teach_gripper_pos, 0, 0)

    
    # Step4: 力反馈 - 使用UnifiedCANThread异步发送
    # 读取夹爪接触力并施加力反馈到示教夹爪
    gripper_force = follow_hand.torque
    logger.debug(f"gripper_force: {gripper_force}")
    feedback_torque = -gripper_force * force_scale
    # 使用UnifiedCANThread异步发送力反馈命令 (纯力矩模式)
    success = unified_can_thread.add_mit_command(
        motor=teach_hand_motor,
        kp=0,    # 纯力矩模式
        kd=0, 
        pos=0,
        vel=0,
        torque=feedback_torque
    )
    if not success and iter_idx % 100 == 0:
        logger.warn("Force feedback command queue full")
    # 注释掉机械臂内置夹爪的力反馈
    teach_hand.control_mit(0, 0, 0, 0, feedback_torque)

    # Step 5: Collect data (obs + action)
    current_time = time.time()
    # relative_time = time.monotonic() - t_start_wall
    relative_time = iter_idx * dt
    data_collector.collect_step(
        obs=obs,
        action=action,
        timestamp=relative_time,
        clock_time=current_time
    )

    # 计算循环实际耗时, 延迟检测和记录
    if iter_idx % 100 == 0:
        actual_freq = iter_idx / (time.monotonic() - t_start_wall)
        if actual_freq < ctrl_frequency * 0.9:  # 频率低于90%才报警
            logger.warn(f"Low freq: {actual_freq:.1f}Hz")

    precise_wait(t_cycle_end)
    iter_idx += 1

camera_head.stop()
camera_wrist.stop()
print("Cameras stopped")
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
data_collector.save_episode()
print("Data collection completed")
#%%
# 停止UnifiedCANThread
logger.info("Stopping UnifiedCANThread...")
unified_can_thread.stop()
unified_can_thread.join(timeout=1.0)
#%%
data_collector.finalize_dataset()  # 生成全局stats.json
 #%%
teach_arm.robot.resting()
follow_arm.robot.resting()