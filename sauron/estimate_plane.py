import numpy as np



# TODO: Implement me :)
def estimate_plane(robot,
                   detected_points: np.ndarray):
    
    print("TODO! Implement Me")
    return None

# ty chatGPT
def transform_plane_to_global_frame(robot, center, normal):
    T_world_robot = robot.as_se2()  # Expected to be a 3x3 homogeneous transformation matrix

    R_world_robot = T_world_robot[:2, :2]  # 2x2 rotation matrix in the xy-plane
    t_world_robot = np.array([T_world_robot[0, 2], T_world_robot[1, 2], 0])  # Translation (x, y, 0)

    R_3d = np.eye(3)
    R_3d[:2, :2] = R_world_robot  # Embed 2D rotation into 3D

    # Transform center and normal to world frame
    center_world = R_3d @ center + t_world_robot
    normal_world = R_3d @ normal  # Rotation preserves normal direction
    
    robot_to_plane = center_world - robot.position
    if np.dot(robot_to_plane, normal_world) > 0:  # Flip if pointing away
        normal_world *= -1

    return center_world, normal_world
