import os
import numpy as np
import yaml
import cv2
import pandas as pd
from sensor_msgs.msg import Imu, Image, CameraInfo
from cv_bridge import CvBridge
from typing import Dict
import rosbag
import tqdm

def extract_data_from_bag(rosbag_path: str, target_directory: str, cfg: Dict[str, str]):
    print("Loading bag file from", rosbag_path)
    os.makedirs(target_directory, exist_ok=True)
    
    rgb_topic = cfg['rgb_topic']
    depth_topic = cfg['depth_topic']
    imu_topic = cfg['imu_topic']
    rgb_info_topic = cfg['rgb_info_topic']
    depth_info_topic = cfg['depth_info_topic']
    
    topics = [
        rgb_topic,
        depth_topic,
        imu_topic,
        rgb_info_topic,
        depth_info_topic,
    ]
    bridge = CvBridge()
    rgb_camera_info = {}
    depth_camera_info = {}
    rgb_image_timestamps = {}
    depth_image_timestamps = {}
    imu_data = []
    
    rgb_count = 0
    depth_count = 0
    
    os.makedirs(f"{target_directory}/rgb", exist_ok=True)
    os.makedirs(f"{target_directory}/depth", exist_ok=True)
    
    with rosbag.Bag(rosbag_path, 'r') as bag:
        for topic, msg, ts in tqdm.tqdm(bag.read_messages(topics), total=bag.get_message_count(topics)):
            timestamp = ts.to_sec()
            
            if topic == rgb_topic:
                image = bridge.imgmsg_to_cv2(msg, "bgr8")
                filename = f"rgb/rgb_{rgb_count:06d}.png"
                rgb_count += 1
                cv2.imwrite(os.path.join(target_directory, filename), image)
                rgb_image_timestamps[filename] = timestamp
                
            elif topic == depth_topic:
                image = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
                filename = f"depth/depth_{depth_count:06d}.npy"
                depth_count += 1
                np.save(os.path.join(target_directory, filename), image)
                depth_image_timestamps[filename] = timestamp
                
            elif topic == imu_topic:
                imu_row = [timestamp, msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z,
                           msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z,
                           msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w]
                imu_data.append(imu_row)
                
            elif topic == rgb_info_topic and not rgb_camera_info:
                rgb_camera_info = {
                    "K": np.array(msg.K).reshape(3,3).tolist(),
                    "D": list(msg.D),
                    "R": list(msg.R),
                    "P": list(msg.P),
                    "distortion_model": msg.distortion_model,
                    "height": msg.height,
                    "width": msg.width
                }
                
            elif topic == depth_info_topic and not depth_camera_info:
                depth_camera_info = {
                    "K": np.array(msg.K).reshape(3,3).tolist(),
                    "D": list(msg.D),
                    "R": list(msg.R),
                    "P": list(msg.P),
                    "distortion_model": msg.distortion_model,
                    "height": msg.height,
                    "width": msg.width
                }
    
    # Save IMU data as CSV
    imu_df = pd.DataFrame(imu_data, columns=["timestamp", "acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z", "quat_x", "quat_y", "quat_z", "quat_w"])
    imu_df.to_csv(os.path.join(target_directory, "imu_data.csv"), index=False)
    
    # Save RGB Camera Info YAML
    with open(os.path.join(target_directory, "rgb_camera_info.yaml"), "w") as f:
        yaml.dump({"camera_info": rgb_camera_info, "image_timestamps": rgb_image_timestamps}, f)
    
    # Save Depth Camera Info YAML
    with open(os.path.join(target_directory, "depth_camera_info.yaml"), "w") as f:
        yaml.dump({"camera_info": depth_camera_info, "image_timestamps": depth_image_timestamps}, f)
        
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("bagfile")
    parser.add_argument("output_dir")
    parser.add_argument("config")
    args = parser.parse_args()
    
    with open(args.config) as f:
        config = yaml.full_load(f)
        
    extract_data_from_bag(args.bagfile, args.output_dir, config)