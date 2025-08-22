# Dummy Control - Simulation Environment

This directory contains code for simulated robotic arms using the gym-aloha environment.

## Features

- MuJoCo-based robot simulation via gym-aloha
- Single arm and dual arm simulation
- Teleoperation in simulation environment
- Compatible with real robot teleoperation interfaces

## Installation

Navigate to this directory and install dependencies:

```bash
cd sim
uv sync
```

## Dependencies

This environment includes simulation-specific dependencies:
- gym-aloha-dummy: ALOHA robot simulation environment
- All real environment dependencies for consistent interfaces

## Usage

### Single Arm Simulation
```bash
python single_arm/sim_single_arm_teleop.py
```

### Dual Arm Simulation  
```bash
python dual_arm/bi_real_arm_teleop_sim.py
```

### Environment Testing
```bash
python dummy_env_sim.py
```

## Directory Structure

- `single_arm/`: Single arm simulation
- `dual_arm/`: Dual arm simulation
- `dummy_env_sim.py`: Environment wrapper and utilities