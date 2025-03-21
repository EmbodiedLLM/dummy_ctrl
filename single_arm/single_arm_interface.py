from telemoma.human_interface.aloha_real_arm import Aloha_Real_Arm_Interface
import numpy as np

class SingleArmInterface(Aloha_Real_Arm_Interface):
    def __init__(self, arm_fibre, joint_offset=np.array([0,-73,180,0,0,0]), angle_measure='radian',
                 gripper_angle_dict=None) -> None:
        """
        Initialize a single arm interface.
        
        Args:
            arm_fibre: The fibre connection to the arm
            joint_offset: Joint offset array
            angle_measure: 'radian' or 'degree'
            gripper_angle_dict: Dictionary containing gripper angles. If None, uses default values.
        """
        # Set default gripper angles if not provided
        if gripper_angle_dict is None:
            gripper_angle_dict = {
                'left_open_angle': -89,
                'right_open_angle': 38,
                'left_close_angle': -158,
                'right_close_angle': -33
            }
        
        # Initialize parent class with the same arm for both left and right
        # This is a hack to avoid modifying the parent class, but we'll override the methods
        # that use the right arm
        super().__init__(
            left_arm_fibre=arm_fibre,
            right_arm_fibre=arm_fibre,  # Use same arm as placeholder
            joint_offset=joint_offset,
            angle_measure=angle_measure,
            gripper_angle_dict=gripper_angle_dict
        )
        
        # Store which arm we're actually using
        self.arm = arm_fibre
        
        # Override parent class initialization of right arm
        # Disable the right arm operations that were done in parent __init__
        self.right_arm.robot.set_enable(True)
        self.right_arm.robot.move_j(0, 0, 90, 0, 0, 0)
        self.right_arm.robot.hand.set_mode(0)
        self.right_arm.robot.hand.set_torque(0)

        
    def get_gripper_action(self):
        """Override to only get left arm gripper action."""
        left_hand_angle = self.left_arm.robot.hand.angle
        left_open_ratio = 1 if abs(left_hand_angle - self.left_open_angle) < abs(left_hand_angle - self.left_close_angle) else 0
        return left_open_ratio, 0  # Return 0 for right arm (unused)
        
    def get_all_joints_angle(self):
        """Override to only get left arm joint angles."""
        left_all_joint_angles = (
            self.left_arm.robot.joint_1.angle,
            self.left_arm.robot.joint_2.angle,
            self.left_arm.robot.joint_3.angle,
            self.left_arm.robot.joint_4.angle,
            self.left_arm.robot.joint_5.angle,
            self.left_arm.robot.joint_6.angle
        )
        left_all_joint_angles = np.array(left_all_joint_angles) + self.joint_offset
        
        # Get gripper action for left arm only
        left_gripper_action, _ = self.get_gripper_action()
        
        if self.angle_measure == 'radian':
            left_all_joint_angles = np.deg2rad(left_all_joint_angles)
            
        left_all_joint_angles = np.concatenate([left_all_joint_angles, [left_gripper_action]])
        right_all_joint_angles = np.zeros_like(left_all_joint_angles)  # Return zeros for right arm
        
        return left_all_joint_angles, right_all_joint_angles
    
    def stop(self):
        """Override to only stop left arm updates."""
        self.running = False
        self.left_arm.robot.set_enable(False)  # Only disable left arm 