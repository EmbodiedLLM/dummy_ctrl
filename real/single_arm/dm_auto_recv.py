import threading
import time
import numpy as np
import queue

from constants import RAD2ANGLE, RAD2ANGLE

# --- Thread-Safe Cache (No changes needed here) ---
class MotorStateCache:
    """
    A thread-safe cache for storing motor states.
    """
    def __init__(self, motor_ids):
        self._motor_ids = motor_ids
        self._states = {motor_id: None for motor_id in motor_ids}
        self._lock = threading.Lock()

    def update_state(self, motor_id, state):
        """Updates the state of a single motor."""
        with self._lock:
            if motor_id in self._states:
                self._states[motor_id] = state

    def get_state(self, motor_id):
        """Retrieves the state of a single motor."""
        with self._lock:
            return self._states.get(motor_id)

    def get_all_states(self):
        """Retrieves a copy of all motor states."""
        with self._lock:
            return self._states.copy()

# --- Corrected Position Updater Thread ---
class PositionUpdaterThread(threading.Thread):
    """
    A background thread that continuously polls motor positions at a fixed frequency,
    using the MotorControl class.
    """
    def __init__(self, motor_control, motor_map, cache, zero_positions, frequency=50, disconnect_timeout=1.0):
        super().__init__(daemon=True)
        self._mc = motor_control
        self._motor_map = motor_map # A map of {master_id: Motor_object}
        self._cache = cache
        self._zero_positions = zero_positions
        self._period = 1.0 / frequency
        self._disconnect_timeout = disconnect_timeout
        self._last_seen_time = {mid: time.monotonic() for mid in self._motor_map.keys()}
        self._running = threading.Event()
        self._running.set()

    def _request_status(self, motor):
        """
        Sends a status request command for a single motor.
        This is a re-implementation of the sending part of refresh_motor_status
        to allow for batch requests.
        """
        try:
            # This uses the private __send_data method to avoid the blocking recv()
            # in the public refresh_motor_status() method.
            can_id_l = motor.SlaveID & 0xff
            can_id_h = (motor.SlaveID >> 8) & 0xff
            data_buf = np.array([np.uint8(can_id_l), np.uint8(can_id_h), 0xCC, 0, 0, 0, 0, 0], dtype=np.uint8)
            # Accessing private method for performance reasons
            self._mc._MotorControl__send_data(0x7FF, data_buf)
        except Exception as e:
            print(f"Error requesting status for motor {motor.SlaveID}: {e}")


    def run(self):
        """The main loop of the thread."""
        while self._running.is_set():
            start_time = time.monotonic()

            if self._mc.serial_.is_open:
                # 1. Request state from all motors
                for motor in self._motor_map.values():
                    self._request_status(motor)
                    time.sleep(0.001)

                # 2. Call recv() and get a list of motors that responded
                responsive_ids = self._mc.recv()

                # Update last seen time for responsive motors
                now = time.monotonic()
                for motor_id in responsive_ids:
                    self._last_seen_time[motor_id] = now

                # 3. Update cache for all motors based on their status
                for motor_id, motor in self._motor_map.items():
                    time_since_last_seen = now - self._last_seen_time[motor_id]

                    if time_since_last_seen > self._disconnect_timeout:
                        # Motor is disconnected
                        state = {
                            'status': 'disconnected',
                            'last_seen': self._last_seen_time[motor_id]
                        }
                    else:
                        # Motor is connected, update with latest data
                        abs_pos_rad = motor.getPosition()  # Keep in radians
                        rel_pos_rad = abs_pos_rad - self._zero_positions[motor_id]  # Relative in radians
                        abs_pos = abs_pos_rad * RAD2ANGLE  # Convert to degrees for display
                        rel_pos = rel_pos_rad * RAD2ANGLE  # Convert to degrees for display
                        state = {
                            'status': 'connected',
                            'pos_abs': abs_pos,  # degrees for display
                            'pos_rel': rel_pos_rad,  # radians for calculations
                            'pos_rel_deg': rel_pos,  # degrees for display
                            'vel': motor.getVelocity(),
                            'tor': motor.getTorque(),
                            'timestamp': time.time()
                        }
                    self._cache.update_state(motor_id, state)

            # 4. Sleep for the rest of the cycle
            elapsed_time = time.monotonic() - start_time
            sleep_time = self._period - elapsed_time
            if sleep_time > 0:
                time.sleep(sleep_time)

    def stop(self):
        """Stops the thread."""
        self._running.clear()


# --- Async Command Send Thread ---
class AsyncCommandSendThread(threading.Thread):
    """异步命令发送线程，支持时间戳调度"""
    
    def __init__(self, motor_control, max_command_age=0.1):
        super().__init__(daemon=True)
        self._mc = motor_control
        self._command_queue = queue.Queue()
        self._running = threading.Event()
        self._running.set()
        self._max_command_age = max_command_age  # 最大命令年龄（秒）
        self._last_command_time = {}  # 每个电机的最新命令时间戳
    
    def add_mit_command(self, motor, kp, kd, pos, vel, torque):
        """添加MIT控制命令到队列，带时间戳调度"""
        current_time = time.time()
        motor_id = motor.SlaveID
        
        cmd = {
            'type': 'MIT',
            'motor': motor,
            'motor_id': motor_id,
            'kp': kp,
            'kd': kd,
            'pos': pos,
            'vel': vel,
            'torque': torque,
            'timestamp': current_time
        }
        
        # 更新该电机的最新命令时间戳
        self._last_command_time[motor_id] = current_time
        
        try:
            self._command_queue.put_nowait(cmd)  # 非阻塞放入队列
            return True
        except queue.Full:
            print("Command queue full, dropping command")
            return False
    
    def run(self):
        """发送线程主循环，带时间戳调度"""
        while self._running.is_set():
            try:
                # 从队列获取命令，超时1ms
                cmd = self._command_queue.get(timeout=0.001)
                
                if cmd['type'] == 'MIT':
                    # 检查命令是否过时
                    if self._should_execute_command(cmd):
                        self._execute_mit_command(cmd)
                    else:
                        print(f"Dropping stale command for motor {cmd['motor_id']}, "
                              f"age: {time.time() - cmd['timestamp']:.4f}s")
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Command send thread error: {e}")
    
    def _should_execute_command(self, cmd):
        """判断是否应该执行该命令"""
        current_time = time.time()
        motor_id = cmd['motor_id']
        cmd_timestamp = cmd['timestamp']
        
        # 检查1: 命令是否超过最大年龄
        command_age = current_time - cmd_timestamp
        if command_age > self._max_command_age:
            return False
        
        # 检查2: 是否有更新的命令已经发送
        if motor_id in self._last_command_time:
            if cmd_timestamp < self._last_command_time[motor_id]:
                return False
        
        return True
    
    def _execute_mit_command(self, cmd):
        """执行MIT控制命令"""
        try:
            motor = cmd['motor']
            if motor.SlaveID not in self._mc.motors_map:
                return
                
            # 数据打包逻辑（从原始controlMIT复制）
            from DM_CAN import float_to_uint
            
            kp_uint = float_to_uint(cmd['kp'], 0, 500, 12)
            kd_uint = float_to_uint(cmd['kd'], 0, 5, 12)
            MotorType = motor.MotorType
            Q_MAX = self._mc.Limit_Param[MotorType][0]
            DQ_MAX = self._mc.Limit_Param[MotorType][1]
            TAU_MAX = self._mc.Limit_Param[MotorType][2]
            q_uint = float_to_uint(cmd['pos'], -Q_MAX, Q_MAX, 16)
            dq_uint = float_to_uint(cmd['vel'], -DQ_MAX, DQ_MAX, 12)
            tau_uint = float_to_uint(cmd['torque'], -TAU_MAX, TAU_MAX, 12)
            
            data_buf = np.array([0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00], np.uint8)
            data_buf[0] = (q_uint >> 8) & 0xff
            data_buf[1] = q_uint & 0xff
            data_buf[2] = dq_uint >> 4
            data_buf[3] = ((dq_uint & 0xf) << 4) | ((kp_uint >> 8) & 0xf)
            data_buf[4] = kp_uint & 0xff
            data_buf[5] = kd_uint >> 4
            data_buf[6] = ((kd_uint & 0xf) << 4) | ((tau_uint >> 8) & 0xf)
            data_buf[7] = tau_uint & 0xff
            
            # 发送数据 - 在独立线程中执行
            self._mc._MotorControl__send_data(motor.SlaveID, data_buf)
            
        except Exception as e:
            print(f"MIT command execution error: {e}")
    
    def stop(self):
        """停止发送线程"""
        self._running.clear()


# --- Async Control Functions ---
class AsyncMotorControl:
    """异步电机控制类，使用命令队列实现真正异步，支持时间戳调度"""
    
    def __init__(self, motor_control, max_command_age=0.05):
        """
        初始化异步电机控制
        :param motor_control: MotorControl实例
        :param max_command_age: 最大命令年龄(秒)，默认50ms适合25Hz控制频率
        """
        self._mc = motor_control
        self._send_thread = AsyncCommandSendThread(motor_control, max_command_age)
        self._send_thread.start()
        
    def control_mit_async(self, motor, kp, kd, pos, vel, torque):
        """
        异步MIT控制 - 立即返回，命令在后台线程执行，带时间戳调度
        """
        return self._send_thread.add_mit_command(motor, kp, kd, pos, vel, torque)
    
    def get_stats(self):
        """获取统计信息"""
        return {
            'queue_size': self._send_thread._command_queue.qsize(),
            'last_command_times': self._send_thread._last_command_time.copy()
        }
    
    def stop(self):
        """停止异步控制"""
        self._send_thread.stop()
