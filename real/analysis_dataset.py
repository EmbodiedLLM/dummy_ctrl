#%%
import json
import pandas as pd
import cv2
from pathlib import Path
data_path = Path("/Users/yinzi/dummy_ctrl/data/2025-08-28/pick_place_greencube_2025-08-28_214754")
episode = 0

# Read metadata
info_path = data_path / "meta" / "info.json"
with open(info_path) as f:
    info = json.load(f)

# Read episode data
parquet_path = data_path / "data" / "chunk-000" / f"episode_{episode:06d}.parquet"
df = pd.read_parquet(parquet_path)

# Get video paths
video_paths = {}
for cam in ["cam_wrist", "cam_head"]:
    video_path = data_path / "videos" / "chunk-000" / f"observation.images.{cam}" / f"episode_{episode:06d}.mp4"
    if video_path.exists():
        video_paths[cam] = video_path

# Open video captures
caps = {cam: cv2.VideoCapture(str(path)) for cam, path in video_paths.items()}
