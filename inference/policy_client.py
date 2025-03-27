import time
import torch
import pandas as pd
import os
from PIL import Image
import numpy as np
import requests
import mediapy as media


def get_observation_from_video_and_state(video_path: str, parquet_path: str, timestamp: float = 0.0) -> dict:
    """Get both video frame and state data for observation"""
    # Get only the first frame from video
    with media.VideoReader(video_path) as reader:
        # Just read the first frame (frame_idx = 0)
        frame = torch.from_numpy(next(reader)).float()
    
    # Read state from parquet
    df = pd.read_parquet(parquet_path)
    closest_idx = (df['timestamp'] - timestamp).abs().argmin()
    state = df.iloc[closest_idx]['observation.state']
    # Fix: Make numpy array writable before converting to tensor
    state = state.copy()  # Create writable copy
    state_tensor = torch.from_numpy(state).float()
    
    # Create observation dict
    observation = {
        "observation.images.cam_wrist": frame,
        "observation.state": state_tensor
    }
    
    return observation

def busy_wait(duration_s: float) -> None:
    """
    Busy wait for the specified duration using monotonic time.
    
    Args:
        duration_s: Duration to wait in seconds
    """
    if duration_s <= 0:
        return
        
    start_time = time.monotonic()
    end_time = start_time + duration_s
    
    while time.monotonic() < end_time:
        pass


inference_time_s = 60
fps = 50
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

# Define the server endpoint
SERVER_URL = "http://localhost:8000/predict"

video_path = "/data1/zjyang/program/tengbo/lerobot/datasets/pick_cube_20demos/videos/chunk-000/observation.images.cam_wrist/episode_000021.mp4"
parquet_path = "/data1/zjyang/program/tengbo/lerobot/datasets/pick_cube_20demos/data/chunk-000/episode_000021.parquet"
current_frame = 0

log_file = open("eval_results.txt", "w")

for _ in range(inference_time_s * fps):
    start_time = time.perf_counter()

    # Read the follower state and access the frames from the cameras
    # observation = robot.capture_observation()
    timestamp = current_frame / fps
    next_timestamp = (current_frame + 1 ) / fps
    observation = get_observation_from_video_and_state(video_path, parquet_path, timestamp)
    next_observation = get_observation_from_video_and_state(video_path, parquet_path, next_timestamp)
    gt_action = next_observation["observation.state"]
    print("gt_action: ", gt_action)
    
    # Prepare data for server request
    image = observation["observation.images.cam_wrist"].numpy().tolist()
    state = observation["observation.state"].numpy().tolist()
    
    # Make request to server
    try:
        response = requests.post(
            SERVER_URL,
            json={"image": image, "state": state},
            timeout=10
        )
        response.raise_for_status()
        prediction_data = response.json()
        action = torch.tensor(prediction_data["prediction"])
    except Exception as e:
        print(f"Error getting prediction from server: {e}")
        # Fallback to dummy prediction if server fails
        action = torch.ones_like(gt_action)
    
    print("predicted action: ", action)
    print("gt - predicted = ", gt_action - action)
    log_file.write(f"predicted action: {action}\n")
    log_file.write(f"gt action: {gt_action}\n")
    log_file.write(f"gt - predicted = {gt_action - action}\n")
    log_file.write("\n")  # Add blank line between iterations
    log_file.flush()  # Ensure output is written immediately


    dt_s = time.perf_counter() - start_time
    busy_wait(1 / fps - dt_s)

    current_frame += 1

log_file.close()
