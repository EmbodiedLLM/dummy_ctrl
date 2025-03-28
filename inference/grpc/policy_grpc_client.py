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
import pandas as pd
from typing import Dict, Any, List, Tuple
import io

from proto import policy_pb2
from proto import policy_pb2_grpc
import sys
from single_arm.arm_angle import ArmAngle

from queue import Queue
from threading import Thread
import time

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


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class VideoStream:
    def __init__(self, url, queue_size=2):
        self.url = url
        self.queue = Queue(maxsize=queue_size)
        self.stopped = False
        
    def start(self):
        Thread(target=self.update, daemon=True).start()
        return self
        
    def update(self):
        cap = cv2.VideoCapture(self.url)
        while not self.stopped:
            if not cap.isOpened():
                print("重新连接摄像头...")
                cap = cv2.VideoCapture(self.url)
                time.sleep(1)
                continue
                
            ret, frame = cap.read()
            if ret:
                if not self.queue.full():
                    self.queue.put(frame)
            else:
                time.sleep(0.01)  # 避免CPU过度使用
                
        cap.release()
        
    def read(self):
        return self.queue.get() if not self.queue.empty() else None
        
    def stop(self):
        self.stopped = True

class PolicyClient:
    def __init__(self, server_address: str = "localhost:50051"):
        """Initialize the gRPC client with server address"""
        self.server_address = server_address
        # Increase message size limits
        self.channel = grpc.insecure_channel(
            server_address,
            options=[
                ('grpc.max_send_message_length', 50 * 1024 * 1024),  # 50MB
                ('grpc.max_receive_message_length', 50 * 1024 * 1024)  # 50MB
            ]
        )
        self.stub = policy_pb2_grpc.PolicyServiceStub(self.channel)
        logger.info(f"Connected to gRPC server at {server_address}")
    
    def health_check(self) -> str:
        """Check if the server is running"""
        try:
            response = self.stub.HealthCheck(policy_pb2.HealthCheckRequest())
            return response.status
        except grpc.RpcError as e:
            logger.error(f"Health check failed: {e}")
            return f"Error: {e}"
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
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
    
    def predict(self, image: np.ndarray, state: List[float]) -> Tuple[List[float], float]:
        """
        Send prediction request to the server
        
        Args:
            image: RGB image as numpy array (H, W, C)
            state: State vector as list of floats
            
        Returns:
            Tuple of (prediction, inference_time_ms)
        """
        try:
            # Convert numpy image to JPEG bytes
            success, encoded_img = cv2.imencode('.jpg', (image * 255).astype(np.uint8))
            if not success:
                raise ValueError("Failed to encode image")
            img_bytes = encoded_img.tobytes()
            
            # Get original image dimensions (for server to reconstruct)
            img_height, img_width = image.shape[0], image.shape[1]
            
            # Create request with encoded image
            request = policy_pb2.PredictRequest(
                encoded_image=img_bytes,
                image_format="jpeg",
                image_height=img_height,
                image_width=img_width,
                state=state
            )
            
            # Time the request
            start_time = time.perf_counter()
            response = self.stub.Predict(request)
            end_time = time.perf_counter()
            
            # Calculate round-trip time
            rtt_ms = (end_time - start_time) * 1000
            logger.info(f"Round-trip time: {rtt_ms:.2f}ms, Server inference time: {response.inference_time_ms:.2f}ms")
            logger.info(f"Sent image bytes size: {len(img_bytes) / 1024:.2f}KB")
            
            return response.prediction, response.inference_time_ms
        
        except grpc.RpcError as e:
            logger.error(f"Prediction failed: {e}")
            return [], 0.0
    
    def close(self):
        """Close the gRPC channel"""
        self.channel.close()


def get_observation_from_stream(stream, state_data):
    """从视频流和状态数据获取观察数据"""
    frame = stream.read()
    if frame is None:
        raise ValueError("无法读取视频帧")
    
    # 转换为RGB并归一化
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = torch.from_numpy(frame).float() / 255.0  # 归一化到 0-1
    print("frame: ", frame.shape)
    # 获取状态数据
    state_tensor = torch.tensor(state_data).float()
    
    # 创建 observation 字典
    observation = {
        "observation.images.cam_wrist": frame,
        "observation.state": state_tensor
    }
    
    return observation


def get_state_from_parquet(parquet_path: str, frame_index: int = 0) -> List[float]:
    """Load state from a parquet file"""
    try:
        df = pd.read_parquet(parquet_path)
        state = df.iloc[frame_index]['observation.state']
        # Make numpy array writable before converting
        state = state.copy()
        return state.tolist()
    except Exception as e:
        logger.error(f"Failed to read state from parquet: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(description="Policy gRPC Client")
    parser.add_argument("--server", default="localhost:50051", help="Server address")
    parser.add_argument("--serial_number", default="396636713233", help="Serial number of the follower arm")
    parser.add_argument("--camera_url", default="http://192.168.65.124:8080/?action=stream", help="Camera URL")
    parser.add_argument("--inference_time_s", type=int, default=60, help="Inference time in seconds")
    parser.add_argument("--control_rate", type=int, default=1, help="Control rate in Hz")
    parser.add_argument("--queue_size", type=int, default=2, help="Queue size")
    parser.add_argument("--warm_up", type=int, default=30, help="Warm-up time")
    args = parser.parse_args()

    logger_fibre = fibre.utils.Logger(verbose=True)
    follower_arm = fibre.find_any(serial_number=args.serial_number, logger=logger_fibre)
    follower_arm.robot.resting()
    joint_offset = np.array([0.0,-73.0,180.0,0.0,0.0,0.0])
    follower_arm.robot.set_enable(True)
    arm_controller = ArmAngle(None, follower_arm, joint_offset)
    print("initialize video stream...")
    stream = VideoStream(url=args.camera_url, queue_size=args.queue_size).start()
    time.sleep(2.0) # Wait for stream to start

    # Create client
    client = PolicyClient(args.server)
    
    # Check server health
    health_status = client.health_check()
    logger.info(f"Server health: {health_status}")
    
    # Get model info
    model_info = client.get_model_info()
    logger.info(f"Model info: {model_info}")

    inference_time_s = args.inference_time_s
    control_rate = args.control_rate

    logger.info(f"Begin {args.warm_up} warm-up...")
    for step in range(args.warm_up):
        follow_joints = arm_controller.get_follow_joints()
        current_state = follow_joints.tolist() + [0.0]
        
        try:
            observation = get_observation_from_stream(stream, current_state)
            logger.info(f"Warm-up step {step + 1}/{args.warm_up}")
        except Exception as e:
            logger.error(f"Warm-up Error: {e}")
        
        precise_sleep(1 / control_rate, time_func=time.monotonic)
    
    logger.info("Warm-up finish,  inference...")

    for _ in range(inference_time_s * control_rate):
        # Read the follower state and access the frames from the cameras
        follow_joints = arm_controller.get_follow_joints()
        if follower_arm.robot.hand.angle <= -158.0:
            gripper = 0.0
        else:
            gripper = 1.0
        current_state = follow_joints.tolist() + [gripper]
        print("current_state: ", current_state)

        try:
            # Get image from video
            observation = get_observation_from_stream(stream, current_state)
            image = observation["observation.images.cam_wrist"].numpy()  # Get as numpy array, not list
            state = observation["observation.state"].numpy().tolist()
            # Make prediction
            prediction, inference_time_ms = client.predict(image, state)
            logger.info(f"Prediction: {prediction}")
            logger.info(f"Server inference time: {inference_time_ms:.2f}ms")
            follower_arm.robot.move_j(
                prediction[0],  # joint_1
                prediction[1],  # joint_2 
                prediction[2],  # joint_3
                prediction[3],  # joint_4
                prediction[4],  # joint_5
                prediction[5]   # joint_6
            )
            if prediction[6] >= 0.9:
                follower_arm.robot.hand.set_angle(-129.03999)  
            else:
                follower_arm.robot.hand.set_angle(-165.0) 
            
        except Exception as e:
            logger.error(f"Error: {e}")
        precise_sleep(1 / control_rate, time_func=time.monotonic)
    client.close()


if __name__ == "__main__":
    main() 
