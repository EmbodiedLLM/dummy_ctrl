"""
Test script for two camera timestamp synchronization
"""

import sys
sys.path.append('.')
import time
import cv2
from queue import Queue
from single_arm.camera_utils import start_camera_thread, get_latest_frame  
from single_arm.timing_utils import get_precise_timestamp

def test_two_cameras():
    print("=== Testing Two Camera Synchronization ===")
    
    # Setup timer
    timer_start = time.monotonic()
    print(f"Timer started at: {timer_start:.6f}")
    
    # Setup queues for both cameras
    camera_queues = {
        'cam0': Queue(maxsize=10),
        'cam1': Queue(maxsize=10)
    }
    
    # Start both camera threads
    print("Starting camera threads...")
    threads = {}
    threads['cam0'] = start_camera_thread('0', camera_queues['cam0'], timer_start, target_fps=30)
    threads['cam1'] = start_camera_thread('1', camera_queues['cam1'], timer_start, target_fps=30)
    
    # Wait for cameras to start
    time.sleep(2)
    
    # Collect synchronized frames for analysis
    print("\nCollecting frames for 5 seconds...")
    frame_data = {'cam0': [], 'cam1': []}
    
    start_collect = time.time()
    while time.time() - start_collect < 5.0:
        current_timestamp = get_precise_timestamp(timer_start)
        
        # Get latest frames from both cameras
        for cam_name, queue in camera_queues.items():
            frame_info = get_latest_frame(queue, max_age=0.1, timer_start_time=timer_start)
            if frame_info:
                timestamp, frame = frame_info
                frame_data[cam_name].append({
                    'timestamp': timestamp,
                    'collect_time': current_timestamp,
                    'frame_shape': frame.shape
                })
        
        time.sleep(0.1)  # Collect every 100ms
    
    # Stop threads
    print("Stopping cameras...")
    for thread in threads.values():
        if hasattr(thread, 'stop_flag'):
            thread.stop_flag['running'] = False
        thread.join(timeout=2.0)
    
    # Analyze timing data
    print("\n=== Analysis Results ===")
    for cam_name, data in frame_data.items():
        if data:
            print(f"\n{cam_name}:")
            print(f"  Frames collected: {len(data)}")
            print(f"  First timestamp: {data[0]['timestamp']:.6f}s")
            print(f"  Last timestamp: {data[-1]['timestamp']:.6f}s")
            print(f"  Frame shape: {data[0]['frame_shape']}")
            
            # Calculate frame intervals
            if len(data) > 1:
                intervals = [data[i+1]['timestamp'] - data[i]['timestamp'] 
                           for i in range(len(data)-1)]
                print(f"  Average interval: {sum(intervals)/len(intervals):.6f}s")
                print(f"  Min interval: {min(intervals):.6f}s")
                print(f"  Max interval: {max(intervals):.6f}s")
    
    # Compare synchronization between cameras
    if frame_data['cam0'] and frame_data['cam1']:
        print(f"\n=== Synchronization Analysis ===")
        
        # Find closest timestamp pairs
        sync_pairs = []
        for frame0 in frame_data['cam0']:
            closest_frame1 = min(frame_data['cam1'], 
                                key=lambda x: abs(x['timestamp'] - frame0['timestamp']))
            time_diff = abs(frame0['timestamp'] - closest_frame1['timestamp'])
            sync_pairs.append(time_diff)
        
        if sync_pairs:
            print(f"Timestamp differences between cameras:")
            print(f"  Average: {sum(sync_pairs)/len(sync_pairs)*1000:.3f}ms")
            print(f"  Min: {min(sync_pairs)*1000:.3f}ms")
            print(f"  Max: {max(sync_pairs)*1000:.3f}ms")
            
            # Count frames within acceptable sync range
            good_sync = sum(1 for diff in sync_pairs if diff < 0.033)  # 33ms threshold
            print(f"  Frames within 33ms sync: {good_sync}/{len(sync_pairs)} ({good_sync/len(sync_pairs)*100:.1f}%)")

if __name__ == "__main__":
    test_two_cameras()