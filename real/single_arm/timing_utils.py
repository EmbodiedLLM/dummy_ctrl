"""
Timing utilities - simple functions for precise timing and timestamps
"""

import time
import math
import numpy as np
from typing import List, Tuple, Optional, Dict

def precise_sleep(dt: float, slack_time: float = 0.001):
    """Precise sleep using hybrid approach"""
    t_start = time.monotonic()
    if dt > slack_time:
        time.sleep(dt - slack_time)
    t_end = t_start + dt
    while time.monotonic() < t_end:
        pass

def precise_wait(t_end: float, slack_time: float=0.001, time_func=time.monotonic):
    t_start = time_func()
    t_wait = t_end - t_start
    if t_wait > 0:
        t_sleep = t_wait - slack_time
        if t_sleep > 0:
            time.sleep(t_sleep)
        while time_func() < t_end:
            pass
    return
