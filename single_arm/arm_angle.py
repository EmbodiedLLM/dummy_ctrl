import numpy as np

class ArmAngle:
    def __init__(self, teach_arm, follow_arm, joint_offset):
        """初始化机械臂角度类
        
        Args:
            teach_arm: 示教机械臂对象
            follow_arm: 跟随机械臂对象
            joint_offset: 关节偏移量数组
        """
        self.teach_arm = teach_arm
        self.follow_arm = follow_arm
        self.joint_offset = joint_offset

    def get_teach_joints(self):
        """获取示教臂的关节角度"""
        teach_joints = np.array([
            self.teach_arm.robot.joint_1.angle,
            self.teach_arm.robot.joint_2.angle,
            self.teach_arm.robot.joint_3.angle,
            self.teach_arm.robot.joint_4.angle,
            self.teach_arm.robot.joint_5.angle,
            self.teach_arm.robot.joint_6.angle
        ])+ self.joint_offset
        return teach_joints

    def get_follow_joints(self):
        follow_joints = np.array([
            self.follow_arm.robot.joint_1.angle,
            self.follow_arm.robot.joint_2.angle,
            self.follow_arm.robot.joint_3.angle,
            self.follow_arm.robot.joint_4.angle,
            self.follow_arm.robot.joint_5.angle,
            self.follow_arm.robot.joint_6.angle
        ])+ self.joint_offset
        return follow_joints
    
    def get_teach_current_pose(self):
        self.teach_arm.robot.eef_pose.update_pose_6D()
        x,y,z,rz,ry,rx = self.teach_arm.robot.eef_pose.x, self.teach_arm.robot.eef_pose.y, self.teach_arm.robot.eef_pose.z, self.teach_arm.robot.eef_pose.a, self.teach_arm.robot.eef_pose.b, self.teach_arm.robot.eef_pose.c
        current_pose = np.array([x,y,z,rz,ry,rx])
        return current_pose
    
    def get_follow_current_pose(self):
        self.follow_arm.robot.eef_pose.update_pose_6D()
        x,y,z,rz,ry,rx = self.follow_arm.robot.eef_pose.x, self.follow_arm.robot.eef_pose.y, self.follow_arm.robot.eef_pose.z, self.follow_arm.robot.eef_pose.a, self.follow_arm.robot.eef_pose.b, self.follow_arm.robot.eef_pose.c
        current_pose = np.array([x,y,z,rz,ry,rx])
        return current_pose
        