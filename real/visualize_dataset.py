#!/usr/bin/env python3
import argparse
import json
import pandas as pd
import rerun as rr
import cv2
from pathlib import Path

def visualize_episode(data_dir: str, episode: int = 0) -> None:
    """Visualize a single episode"""
    data_path = Path(data_dir)
    
    # Read metadata
    info_path = data_path / "meta" / "info.json"
    with open(info_path) as f:
        info = json.load(f)
    
    
    # Read episode data
    parquet_path = data_path / "data" / "chunk-000" / f"episode_{episode:06d}.parquet"
    df = pd.read_parquet(parquet_path)
    
    # Initialize rerun
    rr.init(f"{data_path.name}/episode_{episode}", spawn=True)
    
    # Get video paths
    video_paths = {}
    for cam in ["cam_wrist", "cam_head"]:
        video_path = data_path / "videos" / "chunk-000" / f"observation.images.{cam}" / f"episode_{episode:06d}.mp4"
        if video_path.exists():
            video_paths[cam] = video_path
    
    # Open video captures
    caps = {cam: cv2.VideoCapture(str(path)) for cam, path in video_paths.items()}
    # Visualize each frame
    for i, row in df.iterrows():
        rr.set_time_sequence("frame", i)
        
        # Log camera images
        for cam, cap in caps.items():
            ret, frame = cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rr.log(f"camera/{cam}", rr.Image(frame_rgb))
        
        # Log robot state
        if "observation.joint_states" in row:
            joint_states = row["observation.joint_states"]
            for j, val in enumerate(joint_states):
                rr.log(f"states/joint_{j}", rr.Scalars(float(val)))
        
        if "observation.gripper_pos" in row:
            gripper_pos = row["observation.gripper_pos"]
            rr.log(f"states/gripper_pos", rr.Scalars(float(gripper_pos)))
            
        if "observation.gripper_torque" in row:
            gripper_torque = row["observation.gripper_torque"]
            rr.log(f"states/gripper_torque", rr.Scalars(float(gripper_torque)))
        
        # Log actions
        action = row["action"]
        for j, val in enumerate(action):
            # print(action)
            rr.log(f"states/action_{j}", rr.Scalars(float(val)))
        
        # Log task
        if "task" in row:
            # print(row["task"])
            rr.log("states/task", rr.TextLog(row["task"]))
    
    # Close video captures
    for cap in caps.values():
        cap.release()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", help="Path to dataset directory")
    parser.add_argument("--episode", type=int, default=0, help="Episode number to visualize")
    
    args = parser.parse_args()
    visualize_episode(args.data_dir, args.episode)
    import time
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Ctrl-C received. Exiting.")