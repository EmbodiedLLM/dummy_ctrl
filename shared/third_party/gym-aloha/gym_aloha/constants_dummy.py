from pathlib import Path

### Simulation envs fixed constants
DT = 0.02  # 0.02 ms -> 1/0.2 = 50 hz
FPS = 50

JOINTS = [
    # absolute joint position
    "left_arm_waist",
    "left_arm_shoulder",
    "left_arm_elbow",
    "left_arm_forearm_roll",
    "left_arm_wrist_angle",
    "left_arm_wrist_rotate",
    # normalized gripper position 0: close, 1: open
    "left_arm_gripper",
    # absolute joint position
    "right_arm_waist",
    "right_arm_shoulder",
    "right_arm_elbow",
    "right_arm_forearm_roll",
    "right_arm_wrist_angle",
    "right_arm_wrist_rotate",
    # normalized gripper position 0: close, 1: open
    "right_arm_gripper",
]

ACTIONS = [
    # position and quaternion for end effector
    "left_arm_waist",
    "left_arm_shoulder",
    "left_arm_elbow",
    "left_arm_forearm_roll",
    "left_arm_wrist_angle",
    "left_arm_wrist_rotate",
    # normalized gripper position (0: close, 1: open)
    "left_arm_gripper",
    "right_arm_waist",
    "right_arm_shoulder",
    "right_arm_elbow",
    "right_arm_forearm_roll",
    "right_arm_wrist_angle",
    "right_arm_wrist_rotate",
    # normalized gripper position (0: close, 1: open)
    "right_arm_gripper",
]

START_ARM_POSE = [
    0,
    1.274,
    -1.571,
    0,
    0,
    0,
    0,
    0,
    0,
    1.274,
    -1.571,
    0,
    0,
    0,
    0,
    0,
]

ASSETS_DIR = Path(__file__).parent.resolve() / "assets"  # note: absolute path

# Left finger position limits (qpos[7]), right_finger = -1 * left_finger
PUPPET_GRIPPER_POSITION_OPEN = 0
PUPPET_GRIPPER_POSITION_CLOSE = 0.024



def normalize_puppet_gripper_position(x):
    return (x - PUPPET_GRIPPER_POSITION_CLOSE) / (
        PUPPET_GRIPPER_POSITION_CLOSE - PUPPET_GRIPPER_POSITION_OPEN
    )

def unnormalize_puppet_gripper_position(x):
    return PUPPET_GRIPPER_POSITION_CLOSE - x * (
        PUPPET_GRIPPER_POSITION_CLOSE - PUPPET_GRIPPER_POSITION_OPEN
    )


def normalize_puppet_gripper_velocity(x):
    return x / (PUPPET_GRIPPER_POSITION_CLOSE - PUPPET_GRIPPER_POSITION_OPEN)


def convert_puppet_from_position_to_joint(x):
    return x


def convert_puppet_from_joint_to_position(x):
    return x
