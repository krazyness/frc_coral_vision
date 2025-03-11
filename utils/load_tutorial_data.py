import os, sys, yaml
from scipy.spatial.transform import Rotation
import numpy as np
import cv2
import open3d as o3d

def load_tutorial_data(data_dir, frame_id):
    # Load Data
    rgb = cv2.imread(os.path.join(data_dir, 'rgb', f"rgb_{frame_id:06d}.png"))
    depth = np.load(os.path.join(data_dir, 'depth', f"depth_{frame_id:06d}.npy")).astype(np.float32) / 1000.

    rgb_config_path = os.path.join(os.path.join(data_dir, 'rgb_camera_info.yaml'))
    depth_config_path = os.path.join(os.path.join(data_dir, 'depth_camera_info.yaml'))

    def read_K(config_path):
        with open(config_path) as f:
            config = yaml.full_load(f)
        return np.array(config['camera_info']['K'])

    rgb_K = read_K(rgb_config_path)
    depth_K = read_K(depth_config_path)

    t_depth_color = np.array([0.015254149213433266, 4.602090029948158e-06, -0.00016130083531606942])
    q_depth_color =   np.array([0.0017227759817615151, -0.0014149053022265434, 0.0018222086364403367, -0.9999958276748657])
    R_depth_color = Rotation.from_quat(q_depth_color).as_matrix()

    T_depth_color = np.eye(4)
    T_depth_color[:3, :3] = R_depth_color
    T_depth_color[:3, 3] = t_depth_color

    T_color_depth = np.eye(4)
    T_color_depth[:3, :3] = R_depth_color.T
    T_color_depth[:3, 3] = -R_depth_color.T @ t_depth_color
    
    # Convert RGB to RGB format for display
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

    depth[depth > 2] = 0.

    fx, fy = depth_K[0,0], depth_K[1,1]  # Adjust based on your camera intrinsics
    cx, cy = depth_K[0,2], depth_K[1,2]
    intrinsics = o3d.camera.PinholeCameraIntrinsic(depth.shape[1], depth.shape[0], fx, fy, cx, cy)

    # Create point cloud
    pcd = o3d.geometry.PointCloud.create_from_depth_image(o3d.geometry.Image(depth), intrinsics)

    xyz_depth = np.array(pcd.points)
    xyz_depth_homog = np.hstack((xyz_depth, np.ones_like(xyz_depth[:, 0:1]))).T

    xyz_color = (T_depth_color @ xyz_depth_homog)[:3].T
    uvw_color = rgb_K @ xyz_color.T
    uvw_color /= uvw_color[2]
    pix_color = uvw_color[:2].T

    valid_pix = (pix_color > 0).all(1) \
            & (pix_color[:, 0:1] < rgb.shape[1]).flatten() \
            & (pix_color[:, 1:2] < rgb.shape[0]).flatten()
            
    coords_valid = pix_color[valid_pix].astype(np.int32)

    colors = rgb[coords_valid[:,1], coords_valid[:,0]].astype(np.float32) / 255.
    pts_color_frame = xyz_color[valid_pix]

    pcd_colored = o3d.geometry.PointCloud()
    pcd_colored.points = o3d.utility.Vector3dVector(pts_color_frame)
    pcd_colored.colors = o3d.utility.Vector3dVector(colors)

    R_cam2world = np.array([[1,  0,  0],
                            [0,  0,  1],
                            [0, -1,  0]])

    pcd_colored.estimate_normals()
    pcd_colored.orient_normals_towards_camera_location()

    T_cam2world = np.eye(4)
    T_cam2world[:3, :3] = np.array([[1,  0,  0],
                                    [0,  0,  1],
                                    [0, -1,  0]])


    pcd_colored.transform(T_cam2world)
    return pcd_colored

def plot_clusters(points, labels):
    # Fit a cylinder to each cluster using PCA
    unique_labels = set(labels)

    cluster_pcds = []
    for cluster_label in unique_labels:
        if cluster_label == -1:
            continue  # Ignore noise

        cluster_mask = labels == cluster_label
        cluster_points = points[cluster_mask]

        # Visualize cluster as Open3D point cloud
        cluster_pcd = o3d.geometry.PointCloud()
        cluster_pcd.points = o3d.utility.Vector3dVector(cluster_points)
        cluster_pcd.paint_uniform_color(np.random.rand(3))  # Random color per cluster
        cluster_pcds.append(cluster_pcd)

    o3d.visualization.draw_geometries(cluster_pcds)