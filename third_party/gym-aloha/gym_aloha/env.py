import gymnasium as gym
import numpy as np
from dm_control import mujoco
from dm_control.rl import control
from gymnasium import spaces
import cv2

from gym_aloha.constants import (
    ACTIONS,
    ASSETS_DIR,
    DT,
    JOINTS,
)
from gym_aloha.tasks.sim import BOX_POSE, InsertionTask, TransferCubeTask
from gym_aloha.tasks.sim_dummy import (
    InsertionDummyTask,
    TransferCubeDummyTask,
)
from gym_aloha.utils import sample_box_pose, sample_insertion_pose


class AlohaEnv(gym.Env):
    # TODO(aliberts): add "human" render_mode
    metadata = {"render_modes": ["rgb_array", "human"], "render_fps": 50}
    JOINTS = JOINTS  # Add this line to make JOINTS accessible as class attribute

    def __init__(
        self,
        task,
        obs_type="pixels",
        render_mode="rgb_array",
        observation_width=640,
        observation_height=480,
        visualization_width=640,
        visualization_height=480,
    ):
        super().__init__()
        self.task = task
        self.obs_type = obs_type
        self.render_mode = render_mode
        self.observation_width = observation_width
        self.observation_height = observation_height
        self.visualization_width = visualization_width
        self.visualization_height = visualization_height

        self._env = self._make_env_task(self.task)

        if self.obs_type == "state":
            # Define state observation space with joint positions and velocities
            self.observation_space = spaces.Dict({
                "joint_positions": spaces.Box(
                    low=-np.pi,  # Joint angle lower limits
                    high=np.pi,  # Joint angle upper limits
                    shape=(len(JOINTS),),
                    dtype=np.float64,
                ),
                "joint_velocities": spaces.Box(
                    low=-10.0,  # Reasonable velocity limits
                    high=10.0,
                    shape=(len(JOINTS),),
                    dtype=np.float64,
                ),
                "task_state": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(7,),  # [box_x, box_y, box_z, box_qw, box_qx, box_qy, box_qz]
                    dtype=np.float64,
                )
            })
        elif self.obs_type == "pixels":
            self.observation_space = spaces.Dict(
                {
                    "top": spaces.Box(
                        low=0,
                        high=255,
                        shape=(self.observation_height, self.observation_width, 3),
                        dtype=np.uint8,
                    )
                }
            )
        elif self.obs_type == "pixels_agent_pos":
            self.observation_space = spaces.Dict(
                {
                    "pixels": spaces.Dict(
                        {
                            "top": spaces.Box(
                                low=0,
                                high=255,
                                shape=(self.observation_height, self.observation_width, 3),
                                dtype=np.uint8,
                            )
                        }
                    ),
                    "agent_pos": spaces.Box(
                        low=-1000.0,
                        high=1000.0,
                        shape=(len(JOINTS),),
                        dtype=np.float64,
                    ),
                }
            )
        else:
            raise ValueError(f"Invalid observation type: {self.obs_type}. "
                             f"Valid options are 'state', 'pixels', 'pixels_agent_pos'")

        self.action_space = spaces.Box(low=-1000, high=1000, shape=(len(ACTIONS),), dtype=np.float32)

    def render(self):
        if self.render_mode == "human":
            image = self._render(visualize=True)
            # Convert from RGB to BGR for OpenCV
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.imshow("ALOHA Environment", image)
            cv2.waitKey(1)  # 1ms delay
            return None
        else:
            return self._render(visualize=True)

    def _render(self, visualize=False):
        width = self.visualization_width if visualize else self.observation_width
        height = self.visualization_height if visualize else self.observation_height
        return self._env.physics.render(height=height, width=width, camera_id="top")

    def _make_env_task(self, task_name):
        # time limit is controlled by StepCounter in env factory
        time_limit = float("inf")

        if task_name == "transfer_cube":
            xml_path = ASSETS_DIR / "bimanual_viperx_transfer_cube.xml"
            physics = mujoco.Physics.from_xml_path(str(xml_path))
            task = TransferCubeTask()
        elif task_name == "insertion":
            xml_path = ASSETS_DIR / "bimanual_viperx_insertion.xml"
            physics = mujoco.Physics.from_xml_path(str(xml_path))
            task = InsertionTask()
        elif task_name == "transfer_cube_dummy":
            xml_path = ASSETS_DIR / "bimanual_dummy_transfer_cube.xml"
            physics = mujoco.Physics.from_xml_path(str(xml_path))
            task = TransferCubeDummyTask()
        elif task_name == "insertion_dummy":
            xml_path = ASSETS_DIR / "bimanual_dummy_insertion.xml"
            physics = mujoco.Physics.from_xml_path(str(xml_path))
            task = InsertionDummyTask()
        else:
            raise NotImplementedError(task_name)

        env = control.Environment(
            physics, task, time_limit, control_timestep=DT, n_sub_steps=None, flat_observation=False
        )
        return env

    def _format_raw_obs(self, raw_obs):
        if self.obs_type == "state":
            # Get object pose based on task type
            if self.task == "transfer_cube":
                obj_name = "box"
            elif self.task == "insertion":
                obj_name = "peg"  # or whatever the object name is in insertion task
            elif self.task == "transfer_cube_dummy":
                obj_name = "box"
            elif self.task == "insertion_dummy":
                obj_name = "peg"
            else:
                raise ValueError(f"Unknown task: {self.task}")
            
            obj_pos = self._env.physics.named.data.xpos[obj_name]
            obj_quat = self._env.physics.named.data.xquat[obj_name]
            task_state = np.concatenate([obj_pos, obj_quat])
            
            obs = {
                "joint_positions": raw_obs["qpos"].copy(),
                "joint_velocities": raw_obs["qvel"].copy(),
                "task_state": task_state,
            }
            return obs
        elif self.obs_type == "pixels":
            obs = {"top": raw_obs["images"]["top"].copy()}
        elif self.obs_type == "pixels_agent_pos":
            obs = {
                "pixels": {"top": raw_obs["images"]["top"].copy()},
                "agent_pos": raw_obs["qpos"],
            }
        return obs

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # TODO(rcadene): how to seed the env?
        if seed is not None:
            self._env.task.random.seed(seed)
            self._env.task._random = np.random.RandomState(seed)

        # TODO(rcadene): do not use global variable for this
        if self.task == "transfer_cube":
            BOX_POSE[0] = sample_box_pose(seed)  # used in sim reset
        elif self.task == "insertion":
            BOX_POSE[0] = np.concatenate(sample_insertion_pose(seed))  # used in sim reset
        elif self.task == "transfer_cube_dummy":
            BOX_POSE[0] = np.concatenate(sample_box_pose(seed))  # used in sim reset
        elif self.task == "insertion_dummy":
            BOX_POSE[0] = np.concatenate(sample_insertion_pose(seed))  # used in sim reset
        else:
            raise ValueError(self.task)

        raw_obs = self._env.reset()

        observation = self._format_raw_obs(raw_obs.observation)

        info = {"is_success": False}
        return observation, info

    def step(self, action):
        assert action.ndim == 1
        # TODO(rcadene): add info["is_success"] and info["success"] ?

        _, reward, _, raw_obs = self._env.step(action)

        # TODO(rcadene): add an enum
        terminated = is_success = reward == 4

        info = {"is_success": is_success}

        observation = self._format_raw_obs(raw_obs)

        truncated = False
        return observation, reward, terminated, truncated, info

    def close(self):
        pass
