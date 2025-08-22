import os
import sys
import argparse
import logging
import time
import pandas as pd
import numpy as np
import fibre
from typing import List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ArmController:
    def __init__(self, arm, joint_offset=np.array([0.0, -73.0, 180.0, 0.0, 0.0, 0.0])):
        """初始化机械臂控制器
        
        Args:
            arm: 机械臂对象
            joint_offset: 关节偏移量数组
        """
        self.arm = arm
        self.joint_offset = joint_offset
    
    def get_joints(self):
        """获取机械臂的关节角度"""
        joints = np.array([
            self.arm.robot.joint_1.angle,
            self.arm.robot.joint_2.angle,
            self.arm.robot.joint_3.angle,
            self.arm.robot.joint_4.angle,
            self.arm.robot.joint_5.angle,
            self.arm.robot.joint_6.angle
        ]) + self.joint_offset
        return joints
    
    def move_j(self, joints):
        """移动机械臂关节
        
        Args:
            joints: 关节角度列表 [joint_1, joint_2, joint_3, joint_4, joint_5, joint_6]
        """
        self.arm.robot.move_j(
            joints[0],  # joint_1
            joints[1],  # joint_2
            joints[2],  # joint_3
            joints[3],  # joint_4
            joints[4],  # joint_5
            joints[5]   # joint_6
        )
    
    def set_gripper(self, angle):
        """设置夹爪角度
        
        Args:
            angle: 夹爪角度
        """
        self.arm.robot.hand.set_angle(angle)

def read_parquet_data(parquet_path: str) -> pd.DataFrame:
    """从parquet文件中读取数据
    
    Args:
        parquet_path: parquet文件路径
        
    Returns:
        DataFrame: 包含数据的DataFrame
    """
    try:
        df = pd.read_parquet(parquet_path)
        logger.info(f"成功读取parquet文件，共{len(df)}条数据")
        return df
    except Exception as e:
        logger.error(f"读取parquet文件失败: {e}")
        raise

def get_action_from_parquet(df: pd.DataFrame, frame_index: int) -> List[float]:
    """从DataFrame中获取指定索引的动作数据
    
    Args:
        df: 包含数据的DataFrame
        frame_index: 帧索引
        
    Returns:
        List[float]: 动作数据列表
    """
    try:
        action = df.iloc[frame_index]['action']
        # 确保数组是可写的
        action = action.copy()
        return action.tolist()
    except Exception as e:
        logger.error(f"获取动作数据失败: {e}")
        raise

def main():
    parser = argparse.ArgumentParser(description="从Parquet文件读取数据并执行机械臂MoveJ命令")
    parser.add_argument("--parquet", default="/Users/jack/Desktop/dummy_ctrl/datasets/pick_place_0406/data/chunk-000/episode_000000.parquet", help="包含动作数据的parquet文件路径")
    parser.add_argument("--serial_number", default="396636713233", help="机械臂的序列号")
    parser.add_argument("--start_frame", type=int, default=0, help="起始帧索引")
    parser.add_argument("--end_frame", type=int, default=-1, help="结束帧索引，-1表示所有帧")
    parser.add_argument("--delay", type=float, default=0.02, help="每个动作之间的延迟时间(秒)")
    args = parser.parse_args()
    
    # 初始化fibre日志
    logger_fibre = fibre.utils.Logger(verbose=True)
    
    try:
        # 初始化机械臂
        logger.info(f"连接到序列号为 {args.serial_number} 的机械臂")
        arm = fibre.find_any(serial_number=args.serial_number, logger=logger_fibre)
        
        # 初始化机械臂控制器
        joint_offset = np.array([0.0, -73.0, 180.0, 0.0, 0.0, 0.0])
        arm_controller = ArmController(arm, joint_offset)
        
        # 机械臂归零
        logger.info("机械臂归零中...")
        arm.robot.resting()
        
        # 初始位置
        logger.info("移动到初始位置")
        arm.robot.move_j(0, 0, 90, 0, 0, 0)
        
        # 启用机械臂
        arm.robot.set_enable(True)
        
        # 读取parquet文件
        df = read_parquet_data(args.parquet)
        
        # 确定结束帧
        total_frames = len(df)
        end_frame = args.end_frame
        if end_frame == -1 or end_frame >= total_frames:
            end_frame = total_frames - 1
        
        logger.info(f"执行从 {args.start_frame} 到 {end_frame} 的动作序列")
        
        # 执行动作序列
        for frame_idx in range(args.start_frame, end_frame + 1):
            # 获取动作数据
            action = get_action_from_parquet(df, frame_idx)
            
            logger.info(f"执行第 {frame_idx}/{end_frame} 个动作: {np.round(action, 2)}")
            
            # 执行关节移动
            arm_controller.move_j(action[:6])
            
            # 控制夹爪
            if len(action) > 6:
                gripper_action = action[6]
                # 根据夹爪值设置夹爪角度
                if gripper_action < -155.0:
                    angle = -165.0
                else:
                    angle = gripper_action
                arm_controller.set_gripper(angle)
            
            # 等待延迟时间
            time.sleep(args.delay)
        
        logger.info("动作序列执行完成")
    
    except KeyboardInterrupt:
        logger.info("检测到键盘中断，停止执行")
    except Exception as e:
        logger.error(f"执行出错: {e}")
    finally:
        # 确保机械臂安全
        logger.info("返回休息位置...")
        if 'arm' in locals():
            arm.robot.resting()
            arm.robot.set_enable(False)

if __name__ == "__main__":
    main() 