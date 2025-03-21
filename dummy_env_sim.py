import gym_aloha
import numpy as np
import gymnasium as gym


class AlohaDualTelopWrapper(gym.Wrapper):
    """Wrapper for Aloha environment to handle TeleopAction format."""
    
    def __init__(self, env, joint_offset=np.array([0,-73,180,0,0,0])):
        """Initialize the wrapper.
        
        Args:
            env: The Aloha environment to wrap
        """
        super().__init__(env)
        # Original action space is (14,)
        self.original_action_space = env.action_space
        self.joint_offset = joint_offset

    def step(self, action):
        """Convert TeleopAction to the format expected by Aloha environment.
        
        Args:
            action: TeleopAction object containing left and right arm actions
            # angle unit is radian in robot angle system, just like real robot's movej
            
        Returns:
            Standard gym step return values
        """
        # Extract the first 6 values from left and right arm actions
        # (excluding the last gripper value)
        # input is from movej joint angle(in robot angle system), need to be converted to sensor angle system
        left_arm = action.left[:6] - np.deg2rad(self.joint_offset)
        right_arm = action.right[:6] - np.deg2rad(self.joint_offset)
        left_arm_gripper = action.left[6:]
        right_arm_gripper = action.right[6:]
        
        # Combine into single 14-dimensional array
        # First 7 dimensions for left arm, last 7 for right arm
        combined_action = np.concatenate([
            left_arm,              # Left arm (6)
            left_arm_gripper,      # Left arm gripper (1)
            right_arm,             # Right arm (6)
            right_arm_gripper      # Right arm gripper (1)
        ])
        # Ensure the action is in the correct range
        combined_action = np.clip(
            combined_action,
            self.original_action_space.low,
            self.original_action_space.high
        )
        
        return self.env.step(combined_action)
class AlohaSingleTelopWrapper(gym.Wrapper):
    """Wrapper for Aloha environment to handle TeleopAction format."""
    
    def __init__(self, env, joint_offset=np.array([0,-73,180,0,0,0])):
        """Initialize the wrapper.
        
        Args:
            env: The Aloha environment to wrap
        """
        super().__init__(env)
        # Original action space is (14,)
        self.original_action_space = env.action_space
        self.joint_offset = joint_offset

    def step(self, action):
        """Convert TeleopAction to the format expected by Aloha environment.
        
        Args:
            action: np.ndarray of shape (6,) or (7,)
            # angle unit is radian in robot angle system, just like real robot's movej
            
        Returns:
            Standard gym step return values
        """
        # Handle case where action is only 6 dimensions (no gripper)
        if len(action) == 6:
            action = np.append(action, 0)  # Add default gripper value
            
        # input is from movej joint angle(in robot angle system), need to be converted to sensor angle system
        left_arm = action[:6] - np.deg2rad(self.joint_offset)
        left_arm_gripper = action[6:7]  # Ensure this is a 1D array
        right_arm = np.zeros(6)
        right_arm_gripper = np.zeros(1)

        # Combine into single 14-dimensional array
        # First 7 dimensions for left arm, last 7 for right arm
        combined_action = np.concatenate([
            left_arm,              # Left arm (6)
            left_arm_gripper,      # Left arm gripper (1)
            right_arm,             # Right arm (6)
            right_arm_gripper      # Right arm gripper (1)
        ])
        
        # Ensure the action has the correct shape
        assert combined_action.shape == (14,), f"Action shape is {combined_action.shape}, expected (14,)"
        
        # Ensure the action is in the correct range
        combined_action = np.clip(
            combined_action,
            self.original_action_space.low,
            self.original_action_space.high
        )
        
        return self.env.step(combined_action)

def make_dual_arm_dummy_env_sim():
    env = gym.make("gym_aloha/AlohaDummyInsertion-v0", render_mode="human",obs_type="pixels_agent_pos")
    return AlohaDualTelopWrapper(env)

def make_single_arm_dummy_env_sim():
    env = gym.make("gym_aloha/AlohaDummyInsertion-v0", render_mode="human",obs_type="pixels_agent_pos")
    return AlohaSingleTelopWrapper(env)