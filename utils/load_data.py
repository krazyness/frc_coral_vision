import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation, Slerp
import yaml
from typing import Dict
import cv2
import os

def _time_sync_imgs(rgb_timestamps, depth_timestamps, tolerance=0.02):
    rgb_list = sorted(rgb_timestamps.items(), key=lambda x: x[1])
    depth_list = sorted(depth_timestamps.items(), key=lambda x: x[1])

    matched_pairs = []
    rgb_idx, depth_idx = 0, 0

    while rgb_idx < len(rgb_list) and depth_idx < len(depth_list):
        rgb_key, rgb_time = rgb_list[rgb_idx]
        depth_key, depth_time = depth_list[depth_idx]
        time_diff = abs(rgb_time - depth_time)

        if time_diff <= tolerance:
            matched_pairs.append((rgb_key, depth_key))
            rgb_idx += 1
            depth_idx += 1
        elif rgb_time < depth_time:
            rgb_idx += 1
        else:
            depth_idx += 1

    return matched_pairs

def process_imu_data(file_path):
    # Read the CSV file
    df = pd.read_csv(file_path)
    
    # Compute gravity vector at each timestep (normalize accelerometer readings)
    gravity_vectors = df[['acc_x', 'acc_y', 'acc_z']].values
    gravity_vectors /= (np.linalg.norm(gravity_vectors, axis=1, keepdims=True) + 1e-8)
    
    timestamps = df['timestamp'].values
    
    z_axis = np.array([0, 0, 1])
    rotations = []
    for g in gravity_vectors:
        rot_vec = np.cross(g, z_axis)
        angle = np.arccos(np.dot(g, z_axis))
        if np.linalg.norm(rot_vec) > 1e-6:
            rot_vec /= np.linalg.norm(rot_vec)
            R_obj = Rotation.from_rotvec(angle * rot_vec)
        else:
            R_obj = Rotation.from_matrix(np.eye(3))
        rotations.append(R_obj)
    
    breakpoint()
    rot_slerp = Slerp(timestamps, Rotation.concatenate(rotations))
    
    return rot_slerp

def load_data(data_directory, skip_num = 0):
    imu_path = os.path.join(data_directory, "imu_data.csv")
    rot_slerp = process_imu_data(imu_path)
    
    with open(os.path.join(data_directory, "depth_camera_info.yaml")) as depth_f:
        depth_info = yaml.full_load(depth_f)
    
    with open(os.path.join(data_directory, "rgb_camera_info.yaml")) as rgb_f:
        rgb_info = yaml.full_load(rgb_f)
    
    rgb_timestamps: Dict[str, float] = rgb_info['image_timestamps']
    depth_timestamps: Dict[str, float] = depth_info['image_timestamps']
    
    matched_pairs = _time_sync_imgs(rgb_timestamps, depth_timestamps)
    
    for rgb_key, depth_key in matched_pairs[skip_num:]:
        depth_time = depth_timestamps[depth_key]
        rgb = cv2.imread(os.path.join(data_directory, rgb_key))
        rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        depth = np.load(os.path.join(data_directory, depth_key))
        yield rgb, depth, rot_slerp(depth_time)
