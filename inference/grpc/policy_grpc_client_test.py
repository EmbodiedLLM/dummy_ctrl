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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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


def get_observation_from_video(video_path: str, frame_index: int = 0) -> np.ndarray:
    """Load a frame from a video file"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Cannot open video file")
    
    # Get video info
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f'Video shape=({frame_count}, {height}, {width}, 3) fps={fps}')
    
    # Set frame to read
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    
    # Read the frame
    ret, frame = cap.read()
    if not ret:
        cap.release()
        raise ValueError("Failed to read video frame")
    
    # Convert BGR to RGB
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Normalize to [0, 1]
    frame = frame.astype(np.float32) / 255.0
    
    cap.release()
    return frame, frame_count


def get_total_frames(video_path: str) -> int:
    """Get the total number of frames in a video"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Cannot open video file")
    
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return frame_count


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
    parser.add_argument("--video", default="/Users/yinzi/Downloads/Dummy_V2_workspace/dummy_ai/dummy_ctrl/data/pick_cube_20demos/videos/chunk-000/observation.images.cam_wrist/episode_000021.mp4", help="Path to video file")
    parser.add_argument("--parquet", default="/Users/yinzi/Downloads/Dummy_V2_workspace/dummy_ai/dummy_ctrl/data/pick_cube_20demos/data/chunk-000/episode_000021.parquet", help="Path to parquet file with state data")
    parser.add_argument("--frame_start", type=int, default=0, help="Frame index to start")
    parser.add_argument("--frame_end", type=int, default=100, help="Frame index to end (-1 for all frames)")
    args = parser.parse_args()
    
    # Create client
    client = PolicyClient(args.server)
    
    # Check server health
    health_status = client.health_check()
    logger.info(f"Server health: {health_status}")
    
    # Get model info
    model_info = client.get_model_info()
    logger.info(f"Model info: {model_info}")
    
    # Load DataFrame once
    df = pd.read_parquet(args.parquet)
    
    try:
        # Get total number of frames
        total_frames = get_total_frames(args.video)
        logger.info(f"Total frames in video: {total_frames}")
        
        # Determine end frame
        end_frame = args.frame_end
        if end_frame == -1 or end_frame > total_frames:
            end_frame = total_frames - 1
            
        logger.info(f"Processing frames from {args.frame_start} to {end_frame}")
        
        # Statistics tracking
        total_server_time = 0.0
        total_round_trip_time = 0.0
        inference_count = 0
        
        # Process frames in range
        for frame_idx in range(args.frame_start, end_frame + 1):
            logger.info(f"Processing frame {frame_idx}/{end_frame}")

            # Get image from video
            image, _ = get_observation_from_video(args.video, frame_idx)
            
            # Get state from dataframe
            state = df.iloc[frame_idx]['observation.state'].copy().tolist()
            
            # Time the entire process
            start_time = time.perf_counter()
            
            # Make prediction
            prediction, inference_time_ms = client.predict(image, state)
            
            # Calculate round-trip time
            end_time = time.perf_counter()
            round_trip_ms = (end_time - start_time) * 1000
            
            # Update statistics
            total_server_time += inference_time_ms
            total_round_trip_time += round_trip_ms
            inference_count += 1
            
            # Get ground truth for comparison
            gt = df.iloc[frame_idx]['action']
            
            # Display results
            logger.info(f"Frame {frame_idx} - Server time: {inference_time_ms:.2f}ms, Round-trip: {round_trip_ms:.2f}ms")
            logger.info(f"Ground truth: {gt.tolist()}")
            logger.info(f"Prediction: {prediction}")
            logger.info(f"Difference: {np.round(np.array(prediction) - np.array(gt.tolist()), 2)}")
            
        # Display summary statistics
        if inference_count > 0:
            logger.info("\n--- Performance Summary ---")
            logger.info(f"Processed {inference_count} frames")
            logger.info(f"Average server inference time: {total_server_time / inference_count:.2f}ms")
            logger.info(f"Average round-trip time: {total_round_trip_time / inference_count:.2f}ms")
            logger.info(f"Total processing time: {total_round_trip_time / 1000:.2f}s")
            
    except Exception as e:
        logger.error(f"Error: {e}")
    
    # Close client
    client.close()


if __name__ == "__main__":
    main() 