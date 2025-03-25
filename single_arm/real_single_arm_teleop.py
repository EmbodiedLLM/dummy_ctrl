#%%
import sys
sys.path.append("..")
import fibre
from pynput import keyboard
import numpy as np
from __future__ import print_function
from single_arm.real_collector import LeRobotDataCollector
# logger verbose=True
logger = fibre.utils.Logger(verbose=True)
# %%
teach_arm = fibre.find_any(serial_number="208C31875253", logger=logger)
#%%
follow_arm = fibre.find_any(serial_number="396636713233", logger=logger)
# %%
teach_arm.robot.resting()
follow_arm.robot.resting()
joint_offset = np.array([0.0,-73.0,180.0,0.0,0.0,0.0])
# %%
teach_arm.robot.set_enable(True)
follow_arm.robot.set_enable(True)
logger.info("Moving Teach Arm to Working Pose")
logger.info("Moving Lead Arm to Working Pose")
# teach_arm.robot.move_j(0, 0, 10, 0, 0, 0)
# follow_arm.robot.move_j(0, 0, 10, 0, 0, 0)
#%%
teach_hand_init_angle = teach_arm.robot.hand.angle
follow_hand_init_angle = follow_arm.robot.hand.angle
teach_arm.robot.hand.set_mode(0)
teach_arm.robot.hand.set_torque(0)
follow_arm.robot.hand.set_mode(2)
teach_arm.robot.hand.get_mode()
follow_arm.robot.hand.get_mode()
logger.info(f"Teach Gripper mode: {teach_arm.robot.hand.get_mode()}, Follow Gripper mode: {follow_arm.robot.hand.get_mode()}")
logger.info(f"Teach Gripper angle: {teach_arm.robot.hand.angle}, Follow Gripper angle: {follow_arm.robot.hand.angle}")
# %%

# collector = RobotDataCollector(output_dir="robot_data", fps=1/rate)
# collector = RobotDataCollector(output_dir="robot_data", fps=1/rate, camera_url="http://192.168.65.110:8080/?action=stream")
collector = LeRobotDataCollector(
    output_dir="test",
    fps=10,
    camera_url="http://192.168.65.110:8080/?action=stream",
    robot_type="thu_arm",
    use_video=True,
    # output_dir="test"
    # output_dir="pick_cube_20demos"
)
# %%
teach_arm.robot.set_enable(False)

stop = False
def on_press(key):
    global stop
    try:
        if key == keyboard.Key.shift_r:
            print("\n检测到 Shift, 停止循环...")
            stop = True
            return False
    except AttributeError:
        pass

# 启动键盘监听
listener = keyboard.Listener(on_press=on_press)
listener.start()

import time
# rate = 0.02  # 50Hz
rate = 0.1  # 10Hz
# 开始一个情节的数据收集
collector.start_episode()
while not stop:
    start_time = time.time()
    joint1_angle = teach_arm.robot.joint_1.angle
    joint2_angle = teach_arm.robot.joint_2.angle
    joint3_angle = teach_arm.robot.joint_3.angle
    joint4_angle = teach_arm.robot.joint_4.angle
    joint5_angle = teach_arm.robot.joint_5.angle
    joint6_angle = teach_arm.robot.joint_6.angle
    print(f"Current leader Joints (action): {joint1_angle}, {joint2_angle}, {joint3_angle}, {joint4_angle}, {joint5_angle}, {joint6_angle}")
    all_joint_angles = np.array([joint1_angle, joint2_angle, joint3_angle, joint4_angle, joint5_angle, joint6_angle]) + joint_offset
    all_joint_angles = np.round(all_joint_angles, 5)
    follow_arm.robot.move_j(all_joint_angles[0], all_joint_angles[1], all_joint_angles[2], all_joint_angles[3], all_joint_angles[4], all_joint_angles[5])
    follow_arm.robot.hand.set_angle(teach_arm.robot.hand.angle - teach_hand_init_angle + follow_hand_init_angle)
    print(f"Current follower Joints (state): {all_joint_angles}")
    teach_joints = np.array([teach_arm.robot.joint_1.angle, 
                           teach_arm.robot.joint_2.angle,
                           teach_arm.robot.joint_3.angle, 
                           teach_arm.robot.joint_4.angle,
                           teach_arm.robot.joint_5.angle, 
                           teach_arm.robot.joint_6.angle])
    collector.collect_step(
    teach_joints=teach_joints,
    follow_joints=all_joint_angles,
    teach_gripper=teach_arm.robot.hand.angle,
    follow_gripper=follow_arm.robot.hand.angle,
    rate=rate
    )
    # Sleep precisely to maintain 10Hz
    elapsed = time.time() - start_time
    if elapsed < rate:
        time.sleep(rate - elapsed)
    
    # Calculate and print actual loop frequency
    loop_time = time.time() - start_time
    actual_freq = 1.0 / loop_time if loop_time > 0 else 0
    if abs(loop_time - 0.1) > 0.01:  # Allow 10ms deviation
        print(f"Timing deviation: {loop_time:.3f}s (target: 0.1s)")
# %%
teach_arm.robot.set_enable(True)
follow_arm.robot.set_enable(True)
# %%
collector.save_episode()
# %%
# %%
teach_arm.robot.resting()
follow_arm.robot.resting()
# %%
collector.finalize_dataset()

# %%
