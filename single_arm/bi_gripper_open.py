# open -> 1.0
# close -> 0.0

def gripper_open(teach_gripper, follow_gripper):
    teach_gripper_open  = 33.27999
    teach_gripper_close = 2.5

    follow_gripper_open = -129.03999
    follow_gripper_close = -160.0

    if teach_gripper < teach_gripper_close:
        teach_hand = 0.0
    else:
        teach_hand = 1.0
    if follow_gripper < follow_gripper_close:
        follow_hand = 0.0
    else:
        follow_hand = 1.0
    return teach_hand, follow_hand

