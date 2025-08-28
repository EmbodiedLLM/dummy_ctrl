"""
Simple camera utilities - functions for threaded camera capture
"""

import cv2
import threading
import time
import numpy as np
from collections import deque
from typing import Optional, Tuple
import logging
logger = logging.getLogger(__name__)

class CameraCapture:
    def __init__(self, camera_uri, name: str, fps: float = 30.0, buffer_size: int = 90):
        """
        简化的摄像头采集类，独立线程运行
        
        Args:
            camera_uri: 摄像头URI (int或str)
            name: 摄像头名称
            fps: 采样频率 (默认30Hz)
            buffer_size: 帧缓冲区大小 (默认90帧，约3秒@30fps)
        """
        self.camera_uri = camera_uri
        self.name = name
        self.fps = fps
        self.buffer_size = buffer_size
        self.dt = 1.0 / fps
        
        # OpenCV摄像头对象
        self.cap = None
        
        # 帧缓冲区 - 存储(timestamp, frame)元组
        self.frame_buffer = deque(maxlen=buffer_size)
        
        # 线程控制
        self.thread = None
        self.running = False
        self.lock = threading.Lock()
        
        # 状态监控
        self.last_frame_time = 0
        self.frame_count = 0
        self.error_count = 0
        self.is_healthy = False
        
        print(f"Initialized CameraCapture for {self.name} at {self.fps}fps")
        
    def start(self) -> bool:
        """启动摄像头采集线程"""
        if self.running:
            print(f"Camera {self.name} is already running")
            return True
            
        # 初始化摄像头
        self.cap = cv2.VideoCapture(self.camera_uri)
        if not self.cap.isOpened():
            print(f"Failed to open camera {self.name} at {self.camera_uri}")
            return False
            
        # 设置摄像头参数
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # 最小缓存
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        
        # 启动采集线程
        self.running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()
        
        # 等待第一帧
        start_time = time.time()
        while len(self.frame_buffer) == 0 and time.time() - start_time < 3.0:
            time.sleep(0.1)
            
        if len(self.frame_buffer) == 0:
            print(f"Camera {self.name} failed to capture first frame")
            self.stop()
            return False
            
        print(f"Camera {self.name} started successfully")
        return True
        
    def stop(self):
        """停止摄像头采集"""
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)
            
        if self.cap:
            self.cap.release()
            self.cap = None
            
        with self.lock:
            self.frame_buffer.clear()
            
        print(f"Camera {self.name} stopped")
        
    def _capture_loop(self):
        """摄像头采集主循环"""
        print(f"Starting capture loop for {self.name}")
        
        while self.running:
            loop_start = time.time()
            
            try:
                # 清理摄像头缓存
                for _ in range(2):
                    self.cap.grab()
                
                # 读取帧
                ret, frame = self.cap.read()
                
                if ret and frame is not None:
                    timestamp = time.time()
                    
                    # 添加到缓冲区
                    with self.lock:
                        self.frame_buffer.append((timestamp, frame.copy()))
                    
                    self.last_frame_time = timestamp
                    self.frame_count += 1
                    self.is_healthy = True
                    
                    if self.frame_count % 300 == 0:  # 每10秒打印一次状态
                        print(f"Camera {self.name}: {self.frame_count} frames captured, "
                               f"buffer size: {len(self.frame_buffer)}")
                else:
                    self.error_count += 1
                    self.is_healthy = False
                    print(f"Camera {self.name} failed to read frame (errors: {self.error_count})")
                    
                    # 如果连续错误太多，尝试重新初始化
                    if self.error_count > 10:
                        print(f"Camera {self.name} too many errors, attempting reinit")
                        self._reinit_camera()
                        
            except Exception as e:
                self.error_count += 1
                self.is_healthy = False
                print(f"Camera {self.name} capture error: {e}")
                
            # 控制帧率
            elapsed = time.time() - loop_start
            sleep_time = max(0, self.dt - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)
                
        print(f"Capture loop ended for {self.name}")
        
    def _reinit_camera(self):
        """重新初始化摄像头"""
        try:
            if self.cap:
                self.cap.release()
            time.sleep(0.5)
            
            self.cap = cv2.VideoCapture(self.camera_uri)
            if self.cap.isOpened():
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                self.cap.set(cv2.CAP_PROP_FPS, self.fps)
                self.error_count = 0
                print(f"Camera {self.name} reinitialized successfully")
            else:
                print(f"Camera {self.name} reinit failed")
        except Exception as e:
            print(f"Camera {self.name} reinit error: {e}")
            
    def get_frame_by_timestamp(self, target_timestamp: float, tolerance: float = 0.1) -> Optional[np.ndarray]:
        """
        根据时间戳获取最匹配的帧
        
        Args:
            target_timestamp: 目标时间戳
            tolerance: 容差范围(秒)
            
        Returns:
            匹配的帧，如果没找到返回None
        """
        with self.lock:
            if len(self.frame_buffer) == 0:
                print(f"Camera {self.name} buffer is empty")
                return None
                
            best_frame = None
            min_diff = float('inf')
            
            # 查找最接近的时间戳
            for timestamp, frame in self.frame_buffer:
                time_diff = abs(timestamp - target_timestamp)
                if time_diff <= tolerance and time_diff < min_diff:
                    min_diff = time_diff
                    best_frame = frame
                    
            if best_frame is None:
                # 没找到在容差范围内的帧
                oldest_ts = self.frame_buffer[0][0]
                newest_ts = self.frame_buffer[-1][0]
                print(f"Camera {self.name} no frame found within {tolerance}s tolerance. "
                      f"Target: {target_timestamp:.3f}, "
                      f"Available range: {oldest_ts:.3f} - {newest_ts:.3f}")
                return None
                
            return best_frame
            
    def get_latest_frame(self) -> Optional[Tuple[float, np.ndarray]]:
        """
        获取最新的帧
        
        Returns:
            (timestamp, frame) 或 None
        """
        with self.lock:
            if len(self.frame_buffer) == 0:
                return None
            return self.frame_buffer[-1]
            
    def reset(self):
        """重置摄像头状态（保持线程运行）"""
        with self.lock:
            # 清空帧缓冲区
            self.frame_buffer.clear()
            
            # 重置计数器
            self.frame_count = 0
            self.error_count = 0
            
            # 重置时间戳
            self.last_frame_time = 0
            
            # 重置健康状态
            self.is_healthy = False
            
        print(f"Camera {self.name} state reset")
        
    def get_status(self) -> dict:
        """获取摄像头状态信息"""
        with self.lock:
            buffer_len = len(self.frame_buffer)
            
        return {
            'name': self.name,
            'running': self.running,
            'healthy': self.is_healthy,
            'frame_count': self.frame_count,
            'error_count': self.error_count,
            'buffer_size': buffer_len,
            'last_frame_age': time.time() - self.last_frame_time if self.last_frame_time > 0 else float('inf')
        }
        
    def __del__(self):
        """析构函数"""
        self.stop()
