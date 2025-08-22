# Dummy Control System

A dual-environment robotics control system supporting both real and simulated robotic arms.

## Project Structure

This project is organized into three main directories:

### 🤖 `/real/` - Real Robot Environment
For controlling physical robotic arms. Lightweight deployment without simulation dependencies.

- **Features**: Real robot control, teleoperation, policy inference
- **Key Components**: Single/dual arm control, gRPC/FastAPI services
- **Installation**: `cd real && uv sync`

### 🎮 `/sim/` - Simulation Environment  
For MuJoCo-based robot simulation using gym-aloha environment.

- **Features**: Robot simulation, teleoperation testing, environment development
- **Key Components**: gym-aloha integration, simulation wrappers
- **Installation**: `cd sim && uv sync`

### 📚 `/shared/` - Shared Resources
Common libraries and third-party dependencies used by both environments.

- **Contents**: LeRobot, TeleMoMa, gym-aloha, fibre communication library
- **Purpose**: Centralized dependency management

## Quick Start

### 1. First Time Setup - Familiarize with Hardware

Start with the Jupyter notebook to understand the robot interface:

```bash
cd real
uv sync
jupyter notebook notebooks/dummy_playground.ipynb
```

### 2. Choose Your Environment for Teleoperation

**For Real Robot Deployment:**
```bash
cd real
uv sync
python single_arm/real_single_arm_teleop_joint.py
```

**For Simulation Development:**
```bash
cd sim  
uv sync
python single_arm/sim_single_arm_teleop.py
```

## Environment Comparison

| Feature | Real | Sim |
|---------|------|-----|
| Robot Control | Physical hardware | MuJoCo simulation |
| Dependencies | Lightweight | Includes gym-aloha |
| Use Case | Production deployment | Development & testing |
| Teleoperation | ✅ | ✅ |
| Policy Inference | ✅ | ✅ |

## Independent uv Management

Each environment has its own `pyproject.toml` and `uv.lock` for independent dependency management:

- `/real/pyproject.toml` - Excludes simulation dependencies
- `/sim/pyproject.toml` - Includes full simulation stack

## Usage Guide

### Step 1: Hardware Preparation

Before running examples with real hardware, follow the hardware connection guide available at:
https://c1afcru08q1.feishu.cn/docx/HFyIdxCenoNX6Ox7R7jcFDOcnJd?from=from_copylink

This document explains how to properly set up and connect your robotic arm hardware.

### Step 2: Familiarize with the Robot Arm Interface

Open and run `real/notebooks/dummy_playground.ipynb` to learn about the robot arm's fibre interface operations. This notebook provides an introduction to basic control concepts.

### Step 3: Single Arm Real Hardware Control

Navigate to the `real/single_arm` directory and open `real_single_arm_teleop_joint.py`. Run this script cell by cell in interactive mode to control a single physical robotic arm.

### Step 4: Single Arm Simulation Control

In the `sim/single_arm` directory, open `sim_single_arm_teleop.py`. Run this script cell by cell in interactive mode to control a single simulated robotic arm.

### Step 5: Dual Arm Simulation Control

In the `sim/dual_arm` directory, open `bi_real_arm_teleop_sim.py`. Run this script cell by cell in interactive mode to control two simulated robotic arms simultaneously.

### Step 6: Dual Arm Real Hardware Control

In the `real/dual_arm` directory, open `bi_real_arm_teleop_real.py`. Run this script cell by cell in interactive mode to control two physical robotic arms simultaneously.

## Troubleshooting

If you encounter any issues with hardware connections or software execution, please check the hardware guide or open an issue in this repository.
