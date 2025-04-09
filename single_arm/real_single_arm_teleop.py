#%%
from __future__ import print_function
import sys
sys.path.append("..")
import fibre
from pynput import keyboard
import numpy as np
import time
import threading
# Replace with the modified LeRobotDataCollector
from single_arm.real_collector import LeRobotDataCollector
from single_arm.arm_angle import ArmAngle
from single_arm.bi_gripper_open import gripper_open
# logger verbose=True
logger = fibre.utils.Logger(verbose=True)
# %%
# Use the modified data collector with both cameras
data_collector = LeRobotDataCollector(
    output_dir="/Users/jack/Desktop/dummy_ctrl/datasets/pick_place_0406",
    fps=10,
    camera_urls={
        "cam_wrist": "http://192.168.237.249:8080/?action=stream",
        "cam_head": "http://192.168.237.157:8080/?action=stream"
    },
    robot_type="thu_arm",
    use_video=True
)
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
teach_arm.robot.move_j(0, 0, 90, 0, 0, 0)
follow_arm.robot.move_j(0, 0, 90, 0, 0, 0)
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
teach_arm.robot.set_enable(False)
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
# rate = 0.02  # 50Hz
rate = 0.1  # 10Hz

# Start data collection
data_collector.start_episode()

print("Starting data collection, press right Shift key to stop...")

while not stop:
    start_time = time.time()
    
    # Step 1: Capture the robot state (teach and follow joints)
    teach_joints = arm_controller.get_teach_joints()
    follow_arm.robot.move_j(*teach_joints)
    follow_arm.robot.hand.set_angle(teach_arm.robot.hand.angle - teach_hand_init_angle + follow_hand_init_angle)
    follow_joints = arm_controller.get_follow_joints()
    follow_hand = follow_arm.robot.hand.angle
    teach_hand = follow_hand
    
    # Step 2: Collect data (robot state + camera frames)
    data_collector.collect_step(
        teach_joints=teach_joints,
        follow_joints=follow_joints,
        teach_gripper=teach_hand,
        follow_gripper=follow_hand,
        rate=rate
    )
    
    # Log progress every 10 frames
    if data_collector.frame_count % 10 == 0:
        print(f"Frames collected: {data_collector.frame_count}")
    
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
# teach_arm.robot.resting()
# follow_arm.robot.resting()
teach_arm.robot.move_j(0, 0, 90, 0, 0, 0)
follow_arm.robot.move_j(0, 0, 90, 0, 0, 0)
# %%
print("Program ended, saving data...")

time.sleep(1)

data_collector.save_episode()
print("Data collection completed")
#%%
teach_arm.robot.resting()
follow_arm.robot.resting()

# %%
