#%%
import sys
sys.path.append("..")
import fibre
from pynput import keyboard
import numpy as np
from __future__ import print_function
from single_arm.real_collector import LeRobotDataCollector
from single_arm.arm_angle import ArmAngle
from single_arm.bi_gripper_open import gripper_open
# logger verbose=True
logger = fibre.utils.Logger(verbose=True)

collector = LeRobotDataCollector(
    # output_dir="/Users/jack/Desktop/dummy_ctrl/datasets/test",
    output_dir="/Users/jack/Desktop/dummy_ctrl/datasets/pick_cube_20demos",
    fps=50,
    camera_url="http://192.168.65.124:8080/?action=stream",
    robot_type="thu_arm",
    use_video=True,
)

follow_arm = fibre.find_any(serial_number="396636713233", logger=logger)
# %%
import re
import numpy as np
import time

def parse_predicted_actions(file_path):
    """解析预测动作数据文件"""
    actions = []
    with open(file_path, 'r') as f:
        content = f.read()
        # 修改正则表达式以更准确地匹配数据格式
        pattern = r"predicted action: tensor\(\[(.*?)\]\)"
        matches = re.findall(pattern, content, re.DOTALL)  # 添加 DOTALL 标志
        
        if not matches:
            print(f"未找到匹配的动作数据，文件内容:\n{content[:200]}...")  # 打印部分文件内容用于调试
            return actions
            
        for match in matches:
            try:
                # 清理数据：移除多余的空格和科学计数法中的额外空格
                cleaned = re.sub(r'e\s*([+-])', r'e\1', match.strip())
                # 分割并转换为浮点数
                values = [float(x.strip()) for x in cleaned.split(',') if x.strip()]
                if len(values) >= 7:  # 确保至少有6个值
                    actions.append(values[:7])
                    print(f"成功解析动作: {values[:7]}")
            except ValueError as e:
                print(f"解析数值出错: {e}\n原始数据: {match}")
                continue
    return actions

def execute_actions(arm, actions, rate=0.02):
    """执行一系列动作"""
    for i, action in enumerate(actions):
        print(f"执行第 {i+1}/{len(actions)} 个动作: {action}")
        
        # 移动机械臂
        arm.robot.move_j(
            action[0],  # joint_1
            action[1],  # joint_2 
            action[2],  # joint_3
            action[3],  # joint_4
            action[4],  # joint_5
            action[5]   # joint_6
        )
        
        # 等待指定时间
        time.sleep(rate)
#%%
# 初始化机械臂
follow_arm.robot.resting()
#%%
follow_arm.robot.set_enable(True)
#%%
try:
    # 读取动作序列
    actions = parse_predicted_actions("/Users/jack/Desktop/dummy_ctrl/eval_results.txt")
    if not actions:
        print("没有读取到动作序列")
        follow_arm.robot.resting()
    else:
        print(f"开始执行 {len(actions)} 个动作")
        
        # 只执行一轮动作
        for i, action in enumerate(actions):
            print(f"执行第 {i+1}/{len(actions)} 个动作: {action}")
            
            # 执行动作
            follow_arm.robot.move_j(
                action[0],  # joint_1
                action[1],  # joint_2 
                action[2],  # joint_3
                action[3],  # joint_4
                action[4],  # joint_5
                action[5]   # joint_6
            )
            
            # 控制夹爪
            if action[6] >= 0.9:
                follow_arm.robot.hand.set_angle(-129.03999)  
            else:
                follow_arm.robot.hand.set_angle(-160.0) 
                
            # 等待固定时间间隔（50Hz）
            time.sleep(0.02)
            
        print("动作序列执行完成")
            
except KeyboardInterrupt:
    print("\n检测到键盘中断，停止执行")
except Exception as e:
    print(f"执行出错: {e}")
finally:
    # 确保机械臂安全
    print("返回休息位置...")
    follow_arm.robot.resting()
    follow_arm.robot.set_enable(False)
# %%
