import os
import sys
from pathlib import Path

def find_project_root(current_dir, marker_files=(".git", "pyproject.toml", "setup.py")):
    current_dir = Path(current_dir).absolute()
    
    while current_dir != current_dir.parent:
        for marker in marker_files:
            if (current_dir / marker).exists():
                return current_dir
        current_dir = current_dir.parent
    
    return Path(os.getcwd())

current_dir = Path(__file__).parent.absolute()

project_root = find_project_root(current_dir)
sys.path.append(str(project_root))
import fibre
import grpc
import torch
import numpy as np
import time
import cv2
import argparse
import logging
from single_arm.arm_angle import ArmAngle
from typing import Dict, Any, List, Tuple
from queue import Queue
from threading import Thread, Event

# Import WandB visualization module - try both relative and absolute import
# Add current directory to path to ensure modules can be found
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from wandb_visualizer import (
    TrajectoryVisualizer, 
    DataLogger, 
    ensure_wandb_login, 
    force_create_wandb_project, 
    disable_wandb_sync,
    forward_kinematics
)

# 添加正确的路径以导入模块
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(script_dir, os.pardir, os.pardir))  # 指向项目根目录
inference_dir = os.path.join(parent_dir, "inference", "grpc")
sys.path.append(parent_dir)  # 添加项目根目录以导入ACT模块
sys.path.append(inference_dir)

# 导入协议文件
from proto import policy_pb2
from proto import policy_pb2_grpc

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 添加从policy_grpc_client.py导入的函数
def precise_sleep(dt: float, slack_time: float=0.001, time_func=time.monotonic):
    """
    Use hybrid of time.sleep and spinning to minimize jitter.
    Sleep dt - slack_time seconds first, then spin for the rest.
    """
    t_start = time_func()
    if dt > slack_time:
        time.sleep(dt - slack_time)
    t_end = t_start + dt
    while time_func() < t_end:
        pass
    return

class VideoStream:
    def __init__(self, url, resolution=None, queue_size=2):
        self.url = url
        self.resolution = resolution  # (width, height) or None for default
        self.queue = Queue(maxsize=queue_size)
        self.stopped = False
        
    def start(self):
        Thread(target=self.update, daemon=True).start()
        return self
        
    def update(self):
        cap = cv2.VideoCapture(self.url)
        
        # Set resolution if specified
        if self.resolution:
            width, height = self.resolution
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            logger.info(f"Setting camera resolution to {width}x{height}")
            
        # Get actual resolution
        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        logger.info(f"Actual camera resolution: {actual_width}x{actual_height}")
        
        while not self.stopped:
            if not cap.isOpened():
                logger.info("Reconnecting to camera...")
                cap = cv2.VideoCapture(self.url)
                if self.resolution:
                    width, height = self.resolution
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                time.sleep(1)
                continue
                
            ret, frame = cap.read()
            if ret:
                if not self.queue.full():
                    self.queue.put(frame)
            else:
                time.sleep(0.01)  # Avoid excessive CPU usage
                
        cap.release()
        
    def read(self):
        return self.queue.get() if not self.queue.empty() else None
        
    def stop(self):
        self.stopped = True

class KeyboardMonitor:
    def __init__(self, follower_arm):
        self.follower_arm = follower_arm
        self.enabled = True
        self.stop_event = Event()
        self.thread = Thread(target=self._monitor_keyboard, daemon=True)
        
    def start(self):
        self.thread.start()
        return self
        
    def _monitor_keyboard(self):
        import sys
        import tty
        import termios
        import select
        
        logger.info("Keyboard monitor started. Press Enter to toggle enable/disable.")
        logger.info(f"Current state: {'Enabled' if self.enabled else 'Disabled'}")
        
        # Save terminal settings
        old_settings = termios.tcgetattr(sys.stdin)
        try:
            tty.setcbreak(sys.stdin.fileno())
            
            while not self.stop_event.is_set():
                # Check if input is available (non-blocking)
                if select.select([sys.stdin], [], [], 0.1)[0]:
                    key = sys.stdin.read(1)
                    # Check for Enter key (both '\n' and '\r' for compatibility)
                    if key in ['\n', '\r']:
                        self.enabled = not self.enabled
                        self.follower_arm.robot.set_enable(self.enabled)
                        status = "Enabled" if self.enabled else "Disabled"
                        logger.info(f"Arm {status}")
        finally:
            # Restore terminal settings
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
    
    def stop(self):
        self.stop_event.set()
        if self.thread.is_alive():
            self.thread.join(timeout=1.0)

class VideoStreamManager:
    def __init__(self, stream_wrist, stream_head):
        """
        Initialize the video stream manager with wrist and head camera streams
        
        Args:
            stream_wrist: VideoStream object for wrist camera
            stream_head: VideoStream object for head camera (can be None)
        """
        self.stream_wrist = stream_wrist
        self.stream_head = stream_head
        self.wrist_frame = None
        self.head_frame = None
        self.running = True
        self.update_thread = None
        
    def start(self):
        """Start the video stream manager"""
        self.running = True
        self.update_thread = Thread(target=self.update_frames, daemon=True)
        self.update_thread.start()
        return self
        
    def update_frames(self):
        """Update the latest frames from both cameras"""
        while self.running:
            # Get wrist frame
            if self.stream_wrist:
                frame = self.stream_wrist.read()
                if frame is not None:
                    # Convert BGR to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    self.wrist_frame = frame_rgb
            
            # Get head frame
            if self.stream_head:
                frame = self.stream_head.read()
                if frame is not None:
                    # Convert BGR to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    self.head_frame = frame_rgb
            
            # Sleep a small amount to avoid excessive CPU usage
            time.sleep(0.01)
    
    def get_latest_wrist_frame(self):
        """Return the latest wrist camera frame"""
        return self.wrist_frame
    
    def get_latest_head_frame(self):
        """Return the latest head camera frame"""
        return self.head_frame
    
    def stop(self):
        """Stop the video stream manager"""
        self.running = False
        if self.update_thread:
            self.update_thread.join(timeout=1.0)
        if self.stream_wrist:
            self.stream_wrist.stop()
        if self.stream_head:
            self.stream_head.stop()

class KeyboardMonitorWrapper:
    def __init__(self, robot):
        self.robot = robot
        self.enabled = True
        self._key_pressed = False
        
    def is_key_pressed(self):
        """Check if a key has been pressed to toggle enable/disable"""
        # This is a simple method to simulate key press detection
        # In a real implementation, you would check for actual keyboard input
        # For now, we'll just return False so the robot stays in its current state
        result = self._key_pressed
        self._key_pressed = False  # Reset flag after reading
        return result
        
    def set_key_pressed(self):
        """Signal that a key has been pressed"""
        self._key_pressed = True

class ACTClient:
    def __init__(self, server_address: str = "localhost:50052", 
                 stream_wrist=None, stream_head=None, robot=None, 
                 warmup_steps=20, task=None, control_rate=10, inference_time=300, args=None):
        """初始化gRPC客户端，连接到服务器"""
        self.server_address = server_address
        logger.info(f"连接到gRPC服务器: {server_address}")
        
        try:
            # 增加消息大小限制
            self.channel = grpc.insecure_channel(
                server_address,
                options=[
                    ('grpc.max_send_message_length', 100 * 1024 * 1024),  # 100MB
                    ('grpc.max_receive_message_length', 100 * 1024 * 1024),  # 100MB
                    ('grpc.keepalive_time_ms', 10000),  # 10 seconds
                    ('grpc.keepalive_timeout_ms', 5000),  # 5 seconds
                    ('grpc.http2.max_pings_without_data', 0),
                    ('grpc.keepalive_permit_without_calls', 1)
                ]
            )
            # 设置连接超时为10秒
            try:
                grpc.channel_ready_future(self.channel).result(timeout=10)
                logger.info(f"成功连接到gRPC服务器: {server_address}")
            except grpc.FutureTimeoutError:
                logger.warning(f"连接到服务器超时: {server_address}，将在后续操作中继续尝试连接")
                
            self.stub = policy_pb2_grpc.PolicyServiceStub(self.channel)
        except Exception as e:
            logger.error(f"连接到gRPC服务器失败: {server_address}: {e}")
            # 创建存根以便后续方法可以优雅地失败
            self.channel = grpc.insecure_channel(
                server_address,
                options=[
                    ('grpc.max_send_message_length', 100 * 1024 * 1024),  # 100MB
                    ('grpc.max_receive_message_length', 100 * 1024 * 1024)  # 100MB
                ]
            )
            self.stub = policy_pb2_grpc.PolicyServiceStub(self.channel)
        
        # 存储视频流和机器人
        self.video_thread = VideoStreamManager(stream_wrist, stream_head)
        self.robot = robot
        
        # 存储控制参数
        self.warmup_steps = warmup_steps
        self.task = task
        self.control_rate = control_rate
        self.inference_time = inference_time
        
        # Store args for potential use (e.g., logging)
        self.args = args
        
        # 添加键盘监控器
        self.keyboard_monitor = KeyboardMonitorWrapper(self.robot)
        
    def health_check(self) -> str:
        """检查服务器健康状态"""
        try:
            response = self.stub.HealthCheck(policy_pb2.HealthCheckRequest())
            return response.status
        except grpc.RpcError as e:
            if e.code() == grpc.StatusCode.UNAVAILABLE:
                logger.error(f"服务器不可用: {self.server_address}")
            elif e.code() == grpc.StatusCode.DEADLINE_EXCEEDED:
                logger.error(f"健康检查超时: {self.server_address}")
            else:
                logger.error(f"健康检查失败: {e}")
            return f"Error: {e.code()}"
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        try:
            response = self.stub.GetModelInfo(policy_pb2.ModelInfoRequest())
            return {
                "status": response.status,
                "model_path": response.model_path,
                "device": response.device,
                "input_features": response.input_features,
                "output_features": response.output_features,
                "message": response.message
            }
        except grpc.RpcError as e:
            logger.error(f"Failed to get model info: {e}")
            return {"error": str(e)}
    
    def predict(self, image_wrist: np.ndarray, image_head: np.ndarray, state: List[float], task: str = None) -> Tuple[List[float], float]:
        """
        Send prediction request to the server
        
        Args:
            image_wrist: Wrist camera RGB image as numpy array (H, W, C)
            image_head: Head camera RGB image as numpy array (H, W, C), can be None
            state: State vector as list of floats
            task: Task description for language-conditioned policies
            
        Returns:
            Tuple of (prediction, inference_time_ms)
        """
        try:
            if image_wrist is None:
                raise ValueError("Wrist camera image cannot be None")
            
            # Ensure images are in the correct format (0-1 range floats or uint8)
            if image_wrist.dtype == np.float32 or image_wrist.dtype == np.float64:
                if image_wrist.max() <= 1.0:
                    # Convert to uint8 for JPEG encoding
                    image_wrist = (image_wrist * 255).astype(np.uint8)
            
            # Convert wrist camera image to JPEG bytes
            success1, encoded_img1 = cv2.imencode('.jpg', image_wrist)
            if not success1:
                raise ValueError("Failed to encode wrist camera image")
            img_bytes1 = encoded_img1.tobytes()
            
            # Get wrist image dimensions
            img1_height, img1_width = image_wrist.shape[0], image_wrist.shape[1]
            
            # Create request with encoded images
            request = policy_pb2.PredictRequest(
                encoded_image=img_bytes1,  # Primary image (wrist)
                image_format="jpeg",
                image_height=img1_height,
                image_width=img1_width,
                state=state
            )
            
            # Add task if provided
            if task:
                request.task = task
                logger.info(f"Added task: '{task}'")
            
            # Add head camera image if provided
            if image_head is not None:
                # Ensure head image is in correct format
                if image_head.dtype == np.float32 or image_head.dtype == np.float64:
                    if image_head.max() <= 1.0:
                        # Convert to uint8 for JPEG encoding
                        image_head = (image_head * 255).astype(np.uint8)
                
                success2, encoded_img2 = cv2.imencode('.jpg', image_head)
                if not success2:
                    raise ValueError("Failed to encode head camera image")
                img_bytes2 = encoded_img2.tobytes()
                
                # Get head image dimensions
                img2_height, img2_width = image_head.shape[0], image_head.shape[1]
                
                # Add to request
                request.encoded_image2 = img_bytes2
                request.image2_height = img2_height
                request.image2_width = img2_width
                
                logger.info(f"Sending both camera images - Wrist: {img1_width}x{img1_height}, Head: {img2_width}x{img2_height}")
            else:
                logger.info(f"Sending only wrist camera image: {img1_width}x{img1_height}")
            
            # Time the request
            start_time = time.perf_counter()
            response = self.stub.Predict(request)
            end_time = time.perf_counter()
            
            # Calculate round-trip time
            rtt_ms = (end_time - start_time) * 1000
            logger.info(f"Round-trip time: {rtt_ms:.2f}ms, Server inference time: {response.inference_time_ms:.2f}ms")
            
            # Log image sizes
            wrist_size = len(img_bytes1) / 1024
            logger.info(f"Sent wrist image size: {wrist_size:.2f}KB")
            if image_head is not None:
                head_size = len(img_bytes2) / 1024
                logger.info(f"Sent head image size: {head_size:.2f}KB")
            
            return response.prediction, response.inference_time_ms
        
        except grpc.RpcError as e:
            if e.code() == grpc.StatusCode.UNAVAILABLE:
                logger.error("Server unavailable, may need to reconnect")
            elif e.code() == grpc.StatusCode.DEADLINE_EXCEEDED:
                logger.error("Prediction request timed out, server response too slow")
            else:
                logger.error(f"Prediction request failed: {e.code()}: {e.details()}")
            return [], 0.0
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            return [], 0.0
    
    def close(self):
        """关闭gRPC通道"""
        self.channel.close()

    def run(self, data_logger=None):
        """
        运行客户端的主循环
        """
        try:
            # 健康检查已在main函数中执行，不需要再次执行
            # 仅确认视频流是否正常
            if not self.video_thread.running:
                logger.error("视频流未运行")
                return False
                
            # 获取模型信息（信息已在main中获取，这里只进行日志记录）
            logger.info("开始运行预热和主循环...")
            
            # 准备获取观测数据 
            if not self.robot or not hasattr(self.robot, 'is_initialized'):
                logger.warning("机器人未初始化，使用虚拟机器人")
            elif not self.robot.is_initialized:
                logger.warning("机器人未初始化，使用虚拟机器人")
            
            # 执行预热
            if self.warmup_steps > 0:
                logger.info(f"执行 {self.warmup_steps} 步预热...")
                warmup_successes = 0
                for i in range(self.warmup_steps):
                    # 获取观测
                    observation = self.get_observation_from_streams()
                    if not observation:
                        logger.error(f"获取预热观测失败 (步骤 {i+1}/{self.warmup_steps})")
                        time.sleep(0.1)
                        continue
                    
                    # 提取图像和状态
                    image_wrist = observation['wrist_rgb']
                    image_head = observation.get('head_rgb', None)
                    state = observation['state']
                    
                    # 发送预测请求
                    prediction, inference_time = self.predict(
                        image_wrist=image_wrist,
                        image_head=image_head,
                        state=state,
                        task=self.task
                    )
                    
                    if prediction:
                        warmup_successes += 1
                        logger.debug(f"预热步骤 {i+1}/{self.warmup_steps} 成功，推理时间: {inference_time:.1f}ms")
                    else:
                        logger.warning(f"预热步骤 {i+1}/{self.warmup_steps} 失败")
                    
                    # 小睡一下以避免服务器过载
                    time.sleep(0.05)
                
                # 检查预热成功率
                warmup_success_rate = warmup_successes / self.warmup_steps
                if warmup_success_rate < 0.5:
                    logger.warning(f"预热成功率低: {warmup_success_rate:.1%}，服务器可能不稳定")
                    
                logger.info(f"预热完成, 成功率: {warmup_success_rate:.1%}")
            
            # 主循环
            logger.info("开始主循环...")
            robot_control_enabled = False
            
            # 计算初始目标位置
            if self.robot and hasattr(self.robot, 'get_joint_angles'):
                try:
                    angles = np.array(self.robot.get_joint_angles())
                    last_action = angles
                except Exception as e:
                    logger.warning(f"获取关节角度失败: {e}")
                    last_action = np.zeros(6)  # 失败时使用零向量
            else:
                last_action = np.zeros(6)  # 虚拟机器人的默认动作
                
            start_time = time.time()
            step = 0
            successful_steps = 0
            consecutive_failures = 0
            inference_times = []
            
            try:
                while time.time() - start_time < self.inference_time:
                    loop_start = time.time()
                    
                    # 检查是否按下了Enter键切换控制状态
                    if self.keyboard_monitor.is_key_pressed():
                        robot_control_enabled = not robot_control_enabled
                        if robot_control_enabled:
                            logger.info("机器人控制已启用")
                        else:
                            logger.info("机器人控制已禁用")
                        time.sleep(0.5)  # 防止多次触发
                    
                    # 获取观测
                    observation = self.get_observation_from_streams()
                    if not observation:
                        logger.warning(f"获取观测失败 (步骤 {step+1})")
                        consecutive_failures += 1
                        
                        # 如果连续失败次数过多，暂停一段时间以恢复
                        if consecutive_failures >= 5:
                            recovery_pause = min(consecutive_failures * 0.2, 2.0)  # 最多暂停2秒
                            logger.warning(f"连续 {consecutive_failures} 次失败，暂停 {recovery_pause:.1f} 秒以恢复")
                            time.sleep(recovery_pause)
                        else:
                            time.sleep(0.1)
                        continue
                        
                    # 提取图像和状态
                    image_wrist = observation['wrist_rgb']
                    image_head = observation.get('head_rgb', None)
                    state = observation['state']
                    
                    # 发送预测请求
                    prediction, inference_time = self.predict(
                        image_wrist=image_wrist,
                        image_head=image_head,
                        state=state,
                        task=self.task
                    )
                    
                    if not prediction:
                        logger.warning(f"未收到预测结果 (步骤 {step+1})")
                        consecutive_failures += 1
                        
                        # 如果连续失败次数过多，暂停一段时间
                        if consecutive_failures >= 5:
                            recovery_pause = min(consecutive_failures * 0.2, 2.0)
                            logger.warning(f"连续 {consecutive_failures} 次失败，暂停 {recovery_pause:.1f} 秒以恢复")
                            time.sleep(recovery_pause)
                        else:
                            time.sleep(0.1)
                        continue
                    
                    # 记录推理时间并重置连续失败计数
                    consecutive_failures = 0
                    if inference_time > 0:
                        inference_times.append(inference_time)
                    
                    # Log data if logger is provided and not in warmup
                    if data_logger and not robot_control_enabled: # Assuming robot_control_enabled=False means warmup or initial phase before user enables
                        try:
                            # Ensure prediction is a list before logging
                            prediction_list = prediction if isinstance(prediction, list) else []
                            # Normalize images back to 0-1 range float for logging if they were uint8
                            log_image_wrist = image_wrist.astype(np.float32) / 255.0 if image_wrist.dtype == np.uint8 else image_wrist
                            log_image_head = image_head.astype(np.float32) / 255.0 if image_head is not None and image_head.dtype == np.uint8 else image_head
                            data_logger.save_data(log_image_wrist, log_image_head, state, prediction_list, inference_time)
                        except Exception as log_e:
                            logger.error(f"Failed to log data: {log_e}")
                    
                    # 执行动作
                    if robot_control_enabled and self.robot:
                        try:
                            # 按照policy_grpc_client.py中的逻辑处理动作
                            action_np = np.array(prediction[:6])  # 只取前6个值作为关节角
                            
                            if step == 0:
                                # 第一步使用较小的动作比例，以使过渡更平滑
                                action_to_take = 0.5 * action_np + 0.5 * np.array(self.robot.get_joint_angles() if hasattr(self.robot, 'get_joint_angles') else [0,0,0,0,0,0])
                            else:
                                # 后续步骤使用完整动作
                                action_to_take = action_np
                                
                            # 添加少量上一步动作以获得平滑过渡
                            action_to_take = 0.9 * action_to_take + 0.1 * last_action
                            last_action = action_to_take
                            
                            # 发送动作到机器人
                            if hasattr(self.robot, 'set_joint_target_positions'):
                                self.robot.set_joint_target_positions(action_to_take.tolist())
                                logger.debug(f"设置关节目标位置: {action_to_take.tolist()}")
                            elif hasattr(self.robot, 'move_j'):
                                # 使用move_j作为备选方案
                                self.robot.move_j(*action_to_take.tolist())
                                logger.debug(f"使用move_j设置关节目标位置: {action_to_take.tolist()}")
                            else:
                                logger.warning("机器人没有可用的关节控制方法")
                            
                            # 如果预测包含抓手控制（第7个元素）
                            if len(prediction) > 6:
                                gripper_val = prediction[6]
                                if gripper_val > 0.5 and hasattr(self.robot, 'open_gripper'):
                                    self.robot.open_gripper()
                                    logger.debug("打开抓手")
                                elif gripper_val <= 0.5 and hasattr(self.robot, 'close_gripper'):
                                    self.robot.close_gripper()
                                    logger.debug("关闭抓手")
                                elif hasattr(self.robot, 'hand') and hasattr(self.robot.hand, 'set_angle'):
                                    # 处理夹爪角度
                                    if gripper_val < -155.0:
                                        angle = -165.0
                                    else:
                                        angle = gripper_val
                                    self.robot.hand.set_angle(angle)
                                    logger.debug(f"设置夹爪角度: {angle}")
                            
                            successful_steps += 1

                            # Log data if logger is provided and robot control is enabled
                            if data_logger:
                                try:
                                    # Ensure prediction is a list before logging
                                    prediction_list = prediction if isinstance(prediction, list) else []
                                    # Normalize images back to 0-1 range float for logging if they were uint8
                                    log_image_wrist = image_wrist.astype(np.float32) / 255.0 if image_wrist.dtype == np.uint8 else image_wrist
                                    log_image_head = image_head.astype(np.float32) / 255.0 if image_head is not None and image_head.dtype == np.uint8 else image_head
                                    data_logger.save_data(log_image_wrist, log_image_head, state, prediction_list, inference_time)
                                except Exception as log_e:
                                    logger.error(f"Failed to log data after action: {log_e}")

                        except Exception as e:
                            logger.error(f"执行机器人动作时出错: {str(e)}")
                    else:
                        # 即使没有机器人控制，推理成功也算一个成功步骤
                        successful_steps += 1
                    
                    # 更新步骤计数
                    step += 1
                    
                    # 每100个步骤记录一次性能统计
                    if step % 100 == 0 and step > 0:
                        elapsed = time.time() - start_time
                        rate = step / elapsed
                        avg_inf_time = np.mean(inference_times) if inference_times else 0
                        success_rate = successful_steps / step if step > 0 else 0
                        logger.info(f"性能统计: 步骤={step}, 成功率={success_rate:.1%}, 控制率={rate:.1f}Hz, 平均推理时间={avg_inf_time:.1f}ms")
                    
                    # 维持稳定的控制率
                    loop_time = time.time() - loop_start
                    sleep_time = max(0, 1.0/self.control_rate - loop_time)
                    if sleep_time > 0:
                        precise_sleep(sleep_time, time_func=time.monotonic)
                    
                    # 如果循环时间超过目标控制率的50%，发出警告
                    actual_rate = 1.0 / (time.time() - loop_start)
                    if actual_rate < self.control_rate * 0.5:
                        logger.warning(f"控制循环延迟: 目标={self.control_rate}Hz, 实际={actual_rate:.1f}Hz")
                    
            except KeyboardInterrupt:
                logger.info("用户中断，停止客户端")
                
            # 计算总体性能统计
            total_time = time.time() - start_time
            avg_rate = step / total_time if total_time > 0 else 0
            success_rate = successful_steps / step if step > 0 else 0
            avg_inf_time = np.mean(inference_times) if inference_times else 0
            
            logger.info(f"运行统计: 总步骤={step}, 成功步骤={successful_steps}, 成功率={success_rate:.1%}")
            logger.info(f"性能统计: 运行时间={total_time:.1f}秒, 平均控制率={avg_rate:.1f}Hz, 平均推理时间={avg_inf_time:.1f}ms")
            
            # 完成后重置服务器状态
            # self.reset()
            logger.info(f"客户端运行完成")
            return True
            
        except Exception as e:
            logger.exception(f"客户端运行时出错: {str(e)}")
            return False

    def get_observation_from_streams(self):
        """
        从视频线程和机器人获取最新的观察数据
        
        返回:
            dict: 包含wrist_rgb, head_rgb(可选)和state的字典
        """
        max_attempts = 5  # Maximum attempts to get valid frames
        attempt = 0
        
        while attempt < max_attempts:
            try:
                observation = {}
                
                # 获取手腕相机图像
                wrist_frame = self.video_thread.get_latest_wrist_frame()
                if wrist_frame is None:
                    logger.warning(f"无法获取手腕相机图像 (尝试 {attempt+1}/{max_attempts})")
                    attempt += 1
                    time.sleep(0.1)  # Short delay before retry
                    continue
                
                # 确保图像是三通道RGB格式
                # 首先检查图像的维度和通道数
                logger.debug(f"原始手腕图像形状: {wrist_frame.shape}")
                
                if len(wrist_frame.shape) == 2:  # 灰度图像，只有高度和宽度
                    # 将灰度图转换为3通道RGB图像
                    wrist_frame_rgb = cv2.cvtColor(wrist_frame, cv2.COLOR_GRAY2RGB)
                    logger.info(f"将灰度手腕图像转换为RGB，形状从 {wrist_frame.shape} 到 {wrist_frame_rgb.shape}")
                elif len(wrist_frame.shape) == 3 and wrist_frame.shape[2] == 1:  # 单通道图像
                    # 将单通道图像转换为3通道
                    wrist_frame_rgb = cv2.cvtColor(wrist_frame, cv2.COLOR_GRAY2RGB)
                    logger.info(f"将单通道手腕图像转换为RGB，形状从 {wrist_frame.shape} 到 {wrist_frame_rgb.shape}")
                elif len(wrist_frame.shape) == 3 and wrist_frame.shape[2] == 3:  # BGR图像（OpenCV默认）
                    # 将BGR转换为RGB
                    wrist_frame_rgb = cv2.cvtColor(wrist_frame, cv2.COLOR_BGR2RGB)
                elif len(wrist_frame.shape) == 3 and wrist_frame.shape[2] == 4:  # BGRA图像
                    # 将BGRA转换为RGB
                    wrist_frame_rgb = cv2.cvtColor(wrist_frame, cv2.COLOR_BGRA2RGB)
                else:
                    # 未知格式，尝试转换为RGB
                    logger.warning(f"未知的图像格式: {wrist_frame.shape}，尝试强制转换")
                    try:
                        wrist_frame_rgb = cv2.cvtColor(wrist_frame, cv2.COLOR_BGR2RGB)
                    except cv2.error:
                        # 如果转换失败，创建一个空的RGB图像
                        logger.error("图像格式转换失败，使用空图像")
                        wrist_frame_rgb = np.zeros((480, 640, 3), dtype=np.uint8)
                
                # 记录处理后的图像尺寸和通道数
                logger.info(f"处理后的手腕图像: 形状={wrist_frame_rgb.shape}, 类型={wrist_frame_rgb.dtype}, 通道数={wrist_frame_rgb.shape[2] if len(wrist_frame_rgb.shape) > 2 else 1}")
                
                observation['wrist_rgb'] = wrist_frame_rgb
                
                # 获取头部相机图像（如果可用）
                if self.video_thread.has_head_camera:
                    head_frame = self.video_thread.get_latest_head_frame()
                    if head_frame is not None:
                        # 同样确保头部图像是三通道RGB格式
                        logger.debug(f"原始头部图像形状: {head_frame.shape}")
                        
                        if len(head_frame.shape) == 2:  # 灰度图像
                            head_frame_rgb = cv2.cvtColor(head_frame, cv2.COLOR_GRAY2RGB)
                            logger.info(f"将灰度头部图像转换为RGB，形状从 {head_frame.shape} 到 {head_frame_rgb.shape}")
                        elif len(head_frame.shape) == 3 and head_frame.shape[2] == 1:  # 单通道图像
                            head_frame_rgb = cv2.cvtColor(head_frame, cv2.COLOR_GRAY2RGB)
                            logger.info(f"将单通道头部图像转换为RGB，形状从 {head_frame.shape} 到 {head_frame_rgb.shape}")
                        elif len(head_frame.shape) == 3 and head_frame.shape[2] == 3:  # BGR图像
                            head_frame_rgb = cv2.cvtColor(head_frame, cv2.COLOR_BGR2RGB)
                        elif len(head_frame.shape) == 3 and head_frame.shape[2] == 4:  # BGRA图像
                            head_frame_rgb = cv2.cvtColor(head_frame, cv2.COLOR_BGRA2RGB)
                        else:
                            # 未知格式，尝试转换为RGB
                            logger.warning(f"未知的头部图像格式: {head_frame.shape}，尝试强制转换")
                            try:
                                head_frame_rgb = cv2.cvtColor(head_frame, cv2.COLOR_BGR2RGB)
                            except cv2.error:
                                logger.error("头部图像格式转换失败，不使用头部图像")
                                head_frame_rgb = None
                        
                        if head_frame_rgb is not None:
                            logger.info(f"处理后的头部图像: 形状={head_frame_rgb.shape}, 类型={head_frame_rgb.dtype}, 通道数={head_frame_rgb.shape[2] if len(head_frame_rgb.shape) > 2 else 1}")
                            observation['head_rgb'] = head_frame_rgb
                else:
                    # 如果没有头部摄像头，使用零向量
                    observation['head_rgb'] = None
                
                # 获取机器人状态
                if self.robot and hasattr(self.robot, 'get_joint_angles'):
                    try:
                        # 获取关节角度
                        joint_angles = self.robot.get_joint_angles()
                        logger.debug(f"获取到的关节角度: {joint_angles}")
                        
                        # 确保joint_angles是一个包含6个元素的数组（六轴机器人）
                        if len(joint_angles) != 6:
                            logger.warning(f"关节角度数量不为6: {len(joint_angles)}，将进行调整")
                            if len(joint_angles) < 6:
                                # 如果元素不足，补充零
                                joint_angles = list(joint_angles) + [0.0] * (6 - len(joint_angles))
                            else:
                                # 如果元素过多，截断
                                joint_angles = joint_angles[:6]
                        
                        # 获取抓手状态
                        gripper_state = self.robot.get_gripper_state() if hasattr(self.robot, 'get_gripper_state') else 0.0
                        
                        # 创建状态向量 - 只包含关节角度和抓手状态
                        # 这对应服务器端的s_qpos参数
                        state = np.concatenate([
                            np.array(joint_angles),     # 6个关节角度
                            np.array([float(gripper_state)])  # 1个抓手状态
                        ])
                        
                        logger.debug(f"发送给服务器的状态向量: {state}")
                    except Exception as e:
                        logger.warning(f"获取机器人状态失败: {str(e)}")
                        # 使用零向量作为状态的后备方案 - 6关节角度 + 1抓手状态
                        state = np.zeros(7)
                else:
                    # 如果没有机器人，使用零向量 - 6关节角度 + 1抓手状态
                    state = np.zeros(7)
                    
                observation['state'] = state.tolist()
                
                return observation
                
            except Exception as e:
                logger.error(f"获取观察数据失败 (尝试 {attempt+1}/{max_attempts}): {str(e)}")
                attempt += 1
                time.sleep(0.1)  # Short delay before retry
        
        # If we get here, we've failed all attempts
        logger.error(f"获取观察数据失败，已达最大尝试次数 ({max_attempts})")
        return None
        
    def check_health(self):
        """检查服务器健康状态并返回是否健康"""
        try:
            status = self.health_check()
            # 处理各种可能的健康状态响应
            if isinstance(status, str):
                status_lower = status.lower()
                # 检查常见的成功状态字符串
                if (status == "OK" or status == "healthy" or status == "ok" or 
                    "ok" in status_lower or "health" in status_lower or 
                    "success" in status_lower or "good" in status_lower):
                    return True
            
            logger.warning(f"服务器返回了未预期的健康状态: {status}")
            return False
        except Exception as e:
            logger.error(f"健康检查错误: {e}")
            return False

def get_observation_from_streams(stream_wrist, stream_head, state_data):
    """从两个视频流和状态数据获取观察数据"""
    # 尝试读取手腕摄像头数据
    frame_wrist = stream_wrist.read()
    if frame_wrist is None:
        raise ValueError("无法读取wrist摄像头视频帧")
    
    # 转换wrist图像为RGB
    frame_wrist_rgb = cv2.cvtColor(frame_wrist, cv2.COLOR_BGR2RGB)
    
    # 获取head摄像头数据（如果可用）
    frame_head_rgb = None
    if stream_head is not None:
        frame_head = stream_head.read()
        if frame_head is not None:
            frame_head_rgb = cv2.cvtColor(frame_head, cv2.COLOR_BGR2RGB)
    
    # 显示图像尺寸信息
    logger.info(f"手腕摄像头图像尺寸: {frame_wrist_rgb.shape}")
    if frame_head_rgb is not None:
        logger.info(f"头部摄像头图像尺寸: {frame_head_rgb.shape}")
    else:
        logger.info("未获取到头部摄像头图像")
    
    # 创建 observation 字典
    observation = {
        "observation.images.cam_wrist": frame_wrist_rgb,
        "observation.state": torch.tensor(state_data).float()
    }
    
    # 添加head摄像头（如果可用）
    if frame_head_rgb is not None:
        observation["observation.images.cam_head"] = frame_head_rgb
    
    return observation

def main():
    parser = argparse.ArgumentParser(description="ACT Policy Client")
    parser.add_argument("--server", default="localhost:50051", help="Server address")
    parser.add_argument("--serial_number", default="396636713233", help="Serial number of the follower arm")
    parser.add_argument("--camera_wrist", default="http://192.168.237.249:8080/?action=stream", help="Wrist camera URL")
    parser.add_argument("--camera_head", default="http://192.168.237.157:8080/?action=stream", help="Head camera URL (optional)")
    parser.add_argument("--wrist_resolution", default="1280x720", help="Wrist camera resolution (WxH)")
    parser.add_argument("--head_resolution", default="1280x720", help="Head camera resolution (WxH)")
    parser.add_argument("--inference_time_s", type=int, default=300, help="Inference time in seconds")
    parser.add_argument("--control_rate", type=int, default=10, help="Control rate in Hz")
    parser.add_argument("--queue_size", type=int, default=2, help="Queue size")
    parser.add_argument("--warm_up", type=int, default=5, help="Warm-up steps")
    parser.add_argument("--task", type=str, default="pick the cube into the box", help="Task description for language-conditioned policies")
    parser.add_argument("--log_dir", type=str, default="/Users/jack/lab_intern/dummy_ctrl/lerobot_dp_traj", help="Directory to save log data") # Default path from policy_grpc_client
    parser.add_argument("--use_wandb", action="store_true", default=False, help="Enable logging to Weights & Biases") # Default to False unless specified
    parser.add_argument("--wandb_project", type=str, default="act_traj", help="WandB project name") # Different default project
    parser.add_argument("--wandb_entity", type=str, default=None, help="WandB entity/username")
    parser.add_argument("--wandb_api_key", type=str, default=None, help="WandB API key (alternative to using wandb login)")
    parser.add_argument("--disable_wandb_sync", action="store_true", help="Disable WandB process sync (use thread mode)")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode with verbose logging")
    args = parser.parse_args()

    # Parse camera resolutions
    wrist_width, wrist_height = map(int, args.wrist_resolution.split('x'))
    wrist_resolution = (wrist_width, wrist_height)
    
    head_resolution = None
    if args.head_resolution:
        head_width, head_height = map(int, args.head_resolution.split('x'))
        head_resolution = (head_width, head_height)

    # Set log level based on debug flag
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.info("Debug mode enabled with verbose logging.")
    else:
        logging.getLogger().setLevel(logging.INFO)

    # Handle WandB setup
    if args.disable_wandb_sync:
        disable_wandb_sync()

    if args.wandb_entity:
        os.environ["WANDB_ENTITY"] = args.wandb_entity
    
    if args.wandb_api_key:
        os.environ["WANDB_API_KEY"] = args.wandb_api_key
        logger.info("Using provided WandB API key")

    # Attempt WandB login if enabled
    if args.use_wandb:
        os.environ["WANDB_PROJECT"] = args.wandb_project # Set project name env var
        logger.info(f"Attempting WandB login for project: {args.wandb_project}")
        try:
            import wandb
            # Only attempt interactive login if no API key is found
            if not os.environ.get("WANDB_API_KEY"):
                logger.info("No WANDB_API_KEY detected. Attempting interactive login or using cached credentials.")
                if wandb.login(): 
                    logger.info("WandB login successful (interactive or cached).")
                else:
                    logger.warning("WandB interactive login or cached credential check failed. Disabling WandB.")
                    args.use_wandb = False # Disable wandb if login fails
            else:
                logger.info("WANDB_API_KEY detected. Skipping explicit wandb.login() call.")

            # Verify login status after potential attempt or skip
            if args.use_wandb:
                if wandb.api.api_key:
                     logger.info("Confirmed WandB is authenticated via API key or login.")
                else:
                    logger.warning("WandB authentication check failed. Disabling WandB.")
                    args.use_wandb = False

        except Exception as e:
            logger.error(f"Error during WandB login/check: {e}", exc_info=True)
            args.use_wandb = False

    # Initialize data logger
    data_logger = DataLogger(
        log_dir=args.log_dir, 
        use_wandb=args.use_wandb, 
        wandb_project=args.wandb_project, 
        wandb_entity=args.wandb_entity, 
        wandb_api_key=args.wandb_api_key,
    )

    # Initialize robot interface
    try:
        logger_fibre = fibre.utils.Logger(verbose=True)
        follower_arm = fibre.find_any(serial_number=args.serial_number, logger=logger_fibre)
        follower_arm.robot.resting()
        follower_arm.robot.set_enable(True)
        # follower_arm.robot.move_j(0, -30, 90, 0, 70, 0)
        joint_offset = np.array([0.0,-73.0,180.0,0.0,0.0,0.0])
        
        arm_controller = ArmAngle(None, follower_arm, joint_offset)
        
        # Start keyboard monitor
        keyboard_monitor = KeyboardMonitor(follower_arm).start()
        logger.info("Robot interface and keyboard monitor initialized")
    except Exception as e:
        logger.error(f"Error initializing robot: {e}")
        logger.info("Running in simulation mode without robot control")
        follower_arm = None
        arm_controller = None
        keyboard_monitor = None
    
    # Initialize wrist camera
    logger.info("Initializing wrist camera stream...")
    stream_wrist = VideoStream(url=args.camera_wrist, resolution=wrist_resolution, queue_size=args.queue_size).start()
    
    # Initialize head camera if URL provided
    stream_head = None
    if args.camera_head:
        logger.info("Initializing head camera stream...")
        stream_head = VideoStream(url=args.camera_head, resolution=head_resolution, queue_size=args.queue_size).start()
    
    # Wait for streams to start
    time.sleep(2.0)
    
    # Initialize video stream manager
    video_manager = VideoStreamManager(stream_wrist, stream_head).start()
    
    # Create ACT client
    client = ACTClient(
        server_address=args.server, 
        stream_wrist=stream_wrist, 
        stream_head=stream_head, 
        robot=arm_controller,  # Pass the arm_controller as the robot interface
        warmup_steps=args.warm_up, 
        task=args.task, 
        control_rate=args.control_rate, 
        inference_time=args.inference_time_s,
        args=args # Pass args
    )
    
    # Check server health
    health_status = client.health_check()
    logger.info(f"Server health: {health_status}")
    
    # Get model info
    model_info = client.get_model_info()
    logger.info(f"Model info: {model_info}")
    
    # Reset server state
    # reset_status = client.reset()
    # logger.info(f"Reset status: {reset_status}")
    
    # Set parameters
    # inference_time_s = args.inference_time_s # These are now handled within ACTClient init
    # control_rate = args.control_rate
    # warm_up = args.warm_up

    # logger.info(f"Begin with {warm_up} warm-up steps...") # Logging handled inside run()

    # Start the main client run loop, passing the data logger
    run_successful = client.run(data_logger=data_logger)

    # Clean up
    logger.info("Inference complete, cleaning up...")
    if keyboard_monitor:
        keyboard_monitor.stop()
    video_manager.stop()
    client.close()
    
    # Return to rest position
    if follower_arm:
        logger.info("Returning robot to rest position")
        follower_arm.robot.resting()

    # Finalize data logger
    if data_logger:
        data_logger.finalize()

    logger.info("Client finished.")

if __name__ == "__main__":
    main() 