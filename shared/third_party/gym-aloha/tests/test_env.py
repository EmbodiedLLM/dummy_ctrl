import gymnasium as gym
import pytest
from gymnasium.utils.env_checker import check_env
import numpy as np
from gym_aloha.env import AlohaEnv
import cv2

import gym_aloha  # noqa: F401

@pytest.fixture
def env_state():
    env = AlohaEnv(task="transfer_cube", obs_type="state")
    yield env
    env.close()

@pytest.fixture
def env_pixels():
    env = AlohaEnv(task="transfer_cube", obs_type="pixels")
    yield env
    env.close()

@pytest.mark.parametrize(
    "env_task, obs_type",
    [
        # ("AlohaInsertion-v0", "state"),
        ("AlohaInsertion-v0", "pixels"),
        ("AlohaInsertion-v0", "pixels_agent_pos"),
        ("AlohaTransferCube-v0", "pixels"),
        ("AlohaTransferCube-v0", "pixels_agent_pos"),
    ],
)
def test_aloha(env_task, obs_type):
    env = gym.make(f"gym_aloha/{env_task}", obs_type=obs_type)
    check_env(env.unwrapped)

def test_state_observation_space(env_state):
    # Test observation space structure
    assert "joint_positions" in env_state.observation_space.spaces
    assert "joint_velocities" in env_state.observation_space.spaces
    assert "task_state" in env_state.observation_space.spaces
    
    # Test space dimensions
    assert env_state.observation_space["joint_positions"].shape[0] == len(env_state.JOINTS)
    assert env_state.observation_space["joint_velocities"].shape[0] == len(env_state.JOINTS)
    assert env_state.observation_space["task_state"].shape[0] == 7  # 3 for position + 4 for quaternion

def test_state_observation_reset(env_state):
    obs, _ = env_state.reset()
    
    # Test observation structure
    assert isinstance(obs, dict)
    assert "joint_positions" in obs
    assert "joint_velocities" in obs
    assert "task_state" in obs
    
    # Test observation values
    assert obs["joint_positions"].dtype == np.float64
    assert obs["joint_velocities"].dtype == np.float64
    assert obs["task_state"].dtype == np.float64
    
    # Test value ranges
    assert np.all(obs["joint_positions"] >= -np.pi)
    assert np.all(obs["joint_positions"] <= np.pi)
    assert np.all(obs["joint_velocities"] >= -10.0)
    assert np.all(obs["joint_velocities"] <= 10.0)
    
    # Test quaternion validity
    quat = obs["task_state"][3:7]
    assert np.abs(np.linalg.norm(quat) - 1.0) < 1e-6  # Check unit quaternion

def test_state_observation_step(env_state):
    obs, _ = env_state.reset()
    
    # Test step with random action
    action = env_state.action_space.sample()
    next_obs, reward, terminated, truncated, info = env_state.step(action)
    
    # Test observation structure after step
    assert isinstance(next_obs, dict)
    assert "joint_positions" in next_obs
    assert "joint_velocities" in next_obs
    assert "task_state" in next_obs
    
    # Test that values change after step
    assert not np.allclose(obs["joint_positions"], next_obs["joint_positions"])
    assert not np.allclose(obs["joint_velocities"], next_obs["joint_velocities"])

@pytest.mark.parametrize("task", ["transfer_cube", "insertion"])
def test_state_observation_different_tasks(task):
    env = AlohaEnv(task=task, obs_type="state")
    obs, _ = env.reset()
    
    # Test basic observation properties for different tasks
    assert isinstance(obs, dict)
    assert "joint_positions" in obs
    assert "joint_velocities" in obs
    assert "task_state" in obs

def test_edge_cases(env_state):
    # Test extreme action values
    obs, _ = env_state.reset()
    
    # Test max action
    max_action = np.ones(env_state.action_space.shape) 
    next_obs, _, _, _, _ = env_state.step(max_action)
    assert np.all(np.isfinite(next_obs["joint_positions"]))
    
    # Test min action
    min_action = -np.ones(env_state.action_space.shape)
    next_obs, _, _, _, _ = env_state.step(min_action)
    assert np.all(np.isfinite(next_obs["joint_velocities"]))

def test_seed_reproducibility():
    env1 = AlohaEnv(task="transfer_cube", obs_type="state")
    env2 = AlohaEnv(task="transfer_cube", obs_type="state")
    
    seed = 42
    obs1, _ = env1.reset(seed=seed)
    obs2, _ = env2.reset(seed=seed)
    
    assert np.allclose(obs1["joint_positions"], obs2["joint_positions"])
    assert np.allclose(obs1["task_state"], obs2["task_state"])

def test_invalid_obs_type():
    with pytest.raises(ValueError, match="Invalid observation type: invalid_type"):
        AlohaEnv(task="transfer_cube", obs_type="invalid_type")

def test_invalid_task():
    with pytest.raises(NotImplementedError):
        AlohaEnv(task="invalid_task", obs_type="state")

def test_pixels_observation(env_pixels):
    obs, _ = env_pixels.reset()
    
    # Test observation structure
    assert isinstance(obs, dict)
    assert "top" in obs
    
    # Test image properties
    assert obs["top"].dtype == np.uint8
    assert obs["top"].shape == (480, 640, 3)  # Default size
    assert np.all(obs["top"] >= 0)
    assert np.all(obs["top"] <= 255)
    
    # Test rendering
    rendered_img = env_pixels.render()
    assert rendered_img.shape == (480, 640, 3)
    assert rendered_img.dtype == np.uint8
