"""
Simple camera utilities - functions for threaded camera capture
"""

import cv2
import threading
import time
from queue import Queue, Empty
try:
    from .timing_utils import get_precise_timestamp, precise_sleep
except ImportError:
    from timing_utils import get_precise_timestamp, precise_sleep

def start_camera_thread(camera_url, frame_queue, timer_start_time, target_fps=30):
    """Start a camera capture thread"""
    
    # Shared state for stopping
    stop_flag = {'running': True}
    
    def capture_loop():
        # Setup camera
        if isinstance(camera_url, str) and camera_url.isdigit():
            cap = cv2.VideoCapture(int(camera_url))
        else:
            cap = cv2.VideoCapture(camera_url)
        
        if not cap.isOpened():
            print(f"Failed to open camera: {camera_url}")
            return
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FPS, target_fps)
        
        print(f"Camera {camera_url} opened successfully")
        
        # Capture loop with precise timing
        frame_interval = 1.0 / target_fps
        next_time = time.monotonic()
        
        while stop_flag['running']:
            # Wait for next capture time
            current_time = time.monotonic()
            if current_time < next_time:
                precise_sleep(next_time - current_time)
            
            # Capture frame
            ret, frame = cap.read()
            if ret:
                timestamp = get_precise_timestamp(timer_start_time)
                
                # Put frame in queue (non-blocking)
                try:
                    frame_queue.put_nowait((timestamp, frame))
                except:
                    # Queue full, remove old frame and add new one
                    try:
                        frame_queue.get_nowait()
                        frame_queue.put_nowait((timestamp, frame))
                    except:
                        pass  # Skip this frame if still can't put
            
            # Schedule next capture
            next_time += frame_interval
        
        cap.release()
        print(f"Camera {camera_url} capture stopped")
    
    # Start thread
    thread = threading.Thread(target=capture_loop, daemon=True)
    thread.stop_flag = stop_flag  # Attach stop control to thread
    thread.start()
    return thread

def get_latest_frame(frame_queue, max_age=0.1, timer_start_time=None):
    """Get the most recent frame from queue"""
    latest_frame = None
    current_time = get_precise_timestamp(timer_start_time) if timer_start_time else time.time()
    
    # Drain queue to get latest frame
    try:
        while True:
            timestamp, frame = frame_queue.get_nowait()
            age = current_time - timestamp
            if age <= max_age:
                latest_frame = (timestamp, frame)
            # Keep draining even if frame is too old
    except Empty:
        pass
    
    return latest_frame

def stop_camera_thread(thread):
    """Stop camera capture thread"""
    if thread and thread.is_alive():
        # Signal thread to stop
        thread.capture_loop.running = False
        thread.join(timeout=2.0)