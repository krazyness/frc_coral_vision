import cv2
import numpy as np
import open3d as o3d
import os
from scipy.spatial.transform import Rotation 
import yaml
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Load Data
rgb = cv2.imread("./tutorial_data/rgb.png")


import cv2
import numpy as np



depth = np.load("./tutorial_data/depth.npy").astype(np.float32) / 1000.

rgb_config_path = os.path.join('data/reef_log_1/rgb_camera_info.yaml')
depth_config_path = os.path.join('data/reef_log_1/depth_camera_info.yaml')

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

points = np.array(pcd_colored.points)

vertical_slices = np.array([-0.395, 0.031, 0.19])  # N boundaries
N = len(vertical_slices)  # This creates N+1 bins

# Get colormap
cmap = plt.get_cmap("viridis")
colors_map = [cmap(i / (2 * N + 2))[:3] for i in range(2 * (N + 1))]  # More colors for subdivisions

# Slice the point cloud into N+1 bins
subclouds = []
bin_edges = np.concatenate(([points[:, 2].min()], vertical_slices, [points[:, 2].max()]))

for i in range(N + 1):
    mask = (points[:, 2] >= bin_edges[i]) & (points[:, 2] < bin_edges[i + 1])
    if np.any(mask):
        sub_pcd = o3d.geometry.PointCloud()
        sub_pcd.points = o3d.utility.Vector3dVector(points[mask])
        sub_pcd.paint_uniform_color(colors_map[i])  # Apply bin color
        subclouds.append(sub_pcd)

# Visualize all slices
o3d.visualization.draw_geometries(subclouds)


