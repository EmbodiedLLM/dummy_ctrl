# Dummy Control - Real Robot Environment

This directory contains code for controlling real robotic arms in the dummy control system.

## Features

- Real robot arm control via ODrive and fibre protocol
- Single arm and dual arm teleoperation
- Policy inference services (gRPC and FastAPI)
- Data collection and evaluation tools
- Jupyter notebooks for experimentation

## Installation

Navigate to this directory and install dependencies:

```bash
cd real
uv sync
```

## Dependencies

This environment excludes simulation-specific dependencies like `gym-aloha-dummy` to keep the deployment lightweight for real robot systems.

Key dependencies:
- LeRobot: Policy models and robot control
- TeleMoMa: Teleoperation interfaces
- FastAPI/gRPC: Inference services
- OpenCV, MediaPipe: Vision processing

## Usage

### Single Arm Control
```bash
python single_arm/real_single_arm_teleop_joint.py
```

### Dual Arm Control
```bash
python dual_arm/bi_real_arm_teleop_real.py
```

### Inference Services
- gRPC: `python inference/grpc/policy_grpc_server.py`
- FastAPI: `python inference/fastapi/policy_fastapi_server.py`

## Directory Structure

- `single_arm/`: Single arm robot control
- `dual_arm/`: Dual arm robot control  
- `inference/`: Policy inference services
- `notebooks/`: Jupyter experiments
- `scripts/`: Control scripts