import cv2
import numpy as np
import open3d as o3d

# Load Data
rgb = cv2.imread("./tutorial_data/rgb.png")
depth = np.load("./tutorial_data/depth.npy")  # Depth image (same resolution as RGB)

# Resize depth to match RGB dimensions
depth = cv2.resize(depth, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST).astype(np.float32) / 1000.

# Convert RGB to RGB format for display
rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

# Generate point cloud from RGB-D image
rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
    o3d.geometry.Image(rgb), o3d.geometry.Image(depth),
    depth_scale=1.0, depth_trunc=3.0, convert_rgb_to_intensity=False
)

# Define camera intrinsics
fx, fy = 911.2108154296875, 910.94287109375  # Adjust based on your camera intrinsics
cx, cy = rgb.shape[1] // 2, rgb.shape[0] // 2
intrinsics = o3d.camera.PinholeCameraIntrinsic(rgb.shape[1], rgb.shape[0], fx, fy, cx, cy)

# Create point cloud
pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsics)

# Flip the point cloud for Open3D visualization compatibility
pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

# Visualize point cloud
o3d.visualization.draw_geometries([pcd])