#!/usr/bin/env python3
"""
重新生成episodes_stats.jsonl
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path

def calculate_episode_stats(parquet_path, episode_idx):
    """计算单个episode的统计信息"""
    df = pd.read_parquet(parquet_path)
    
    stats = {}
    
    # 处理各种观测和动作数据
    stat_keys = ["observation.joint_states", "observation.gripper_pos", "observation.gripper_torque", "action"]
    
    for key in stat_keys:
        if key in df.columns:
            try:
                data = df[key].tolist()
                arr = np.array(data)
                
                stats[key] = {
                    "mean": arr.mean(axis=0).tolist(),
                    "std": arr.std(axis=0).tolist(),
                    "min": arr.min(axis=0).tolist(), 
                    "max": arr.max(axis=0).tolist(),
                    "count": [len(df)]
                }
            except Exception as e:
                print(f"Warning: Error computing stats for {key} in episode {episode_idx}: {e}")
                continue
    
    return {
        "episode_index": episode_idx,
        "stats": stats
    }

def main():
    data_dir = Path("/Users/yinzi/dummy_ctrl/data/2025-08-29/pick_place_greencube_2025-08-29_180507")
    parquet_dir = data_dir / "data" / "chunk-000"
    stats_file = data_dir / "meta" / "episodes_stats.jsonl"
    
    # 删除旧的统计文件
    if stats_file.exists():
        stats_file.unlink()
        print("Removed old episodes_stats.jsonl")
    
    # 重新生成统计
    parquet_files = list(parquet_dir.glob("episode_*.parquet"))
    print(f"Processing {len(parquet_files)} episodes...")
    
    with open(stats_file, "w") as f:
        for parquet_file in sorted(parquet_files):
            episode_idx = int(parquet_file.stem.split('_')[1])
            episode_stats = calculate_episode_stats(parquet_file, episode_idx)
            f.write(json.dumps(episode_stats) + "\n")
            print(f"Generated stats for episode {episode_idx}")
    
    print(f"Generated new episodes_stats.jsonl with {len(parquet_files)} episodes")

if __name__ == "__main__":
    main()