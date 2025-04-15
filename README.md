# Dummy Control Repository

This repository contains examples for controlling robotic arms in both real and simulated environments.

## Setup

### 1. Clone the Repository

```bash
git clone https://github.com/EmbodiedLLm/dummy_ctrl
cd dummy_ctrl
```

### 2. Environment Setup

For Mac:
brew install libusb

This project uses `uv` for Python environment management. Set up the environment with:

```bash
uv venv
source .venv/bin/activate
uv sync
uv pip install pyusb pynput 
uv pip install -r requirements.txt
```

This will create a virtual environment and install all required dependencies.

## Usage Guide

### Step 1: Hardware Preparation

Before running examples with real hardware, follow the hardware connection guide available at:
https://c1afcru08q1.feishu.cn/docx/HFyIdxCenoNX6Ox7R7jcFDOcnJd?from=from_copylink

This document explains how to properly set up and connect your robotic arm hardware.

### Step 2: Familiarize with the Robot Arm Interface

Open and run `dummy_playground.ipynb` to learn about the robot arm's fibre interface operations. This notebook provides an introduction to basic control concepts.

### Step 3: Single Arm Real Hardware Control

Navigate to the `single_arm` directory and open `real_single_arm_teleop.py`. Run this script cell by cell in interactive mode to control a single physical robotic arm.

### Step 4: Single Arm Simulation Control

In the `single_arm` directory, open `sim_single_arm_teleop.py`. Run this script cell by cell in interactive mode to control a single simulated robotic arm.

### Step 5: Dual Arm Simulation Control

In the `dual_arm` directory, open `sim_dual_arm_teleop.py` or `bi_real_arm_teleop_sim.py`. Run this script cell by cell in interactive mode to control two simulated robotic arms simultaneously.

### Step 6: Dual Arm Real Hardware Control

In the `dual_arm` directory, open `bi_real_arm_teleop_real.py`. Run this script cell by cell in interactive mode to control two physical robotic arms simultaneously.

## Troubleshooting

If you encounter any issues with hardware connections or software execution, please check the hardware guide or open an issue in this repository.
