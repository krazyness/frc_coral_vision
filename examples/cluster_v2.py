import time
import cv2
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation 
import yaml
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.optimize import minimize

import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir))

from utils.load_tutorial_data import load_tutorial_data

pcd_colored = load_tutorial_data()

points = np.array(pcd_colored.points)


# Parameters
CYLINDER_RADIUS = 0.01  # Cylinder radius
CYLINDER_HEIGHT = 1.0  # Cylinder height
CYLINDER_ALPHA = 0.5  # Transparency level (0 = fully transparent, 1 = opaque)

# Cluster the entire point cloud
dbscan = DBSCAN(eps=0.1, min_samples=15).fit(points[:, :2])
labels = dbscan.labels_
unique_labels = set(labels)

# Store cylinders and clusters
cylinders = []
cluster_pcds = []

# Fit a cylinder to each cluster using PCA
for cluster_label in unique_labels:
    if cluster_label == -1:
        continue  # Ignore noise

    cluster_mask = labels == cluster_label
    cluster_points = points[cluster_mask]

    # Compute PCA to find the dominant direction
    pca = PCA(n_components=3)
    pca.fit(cluster_points)
    primary_direction = pca.components_[0]  # Principal axis

    # Compute cluster mean position
    mean_pos = np.mean(cluster_points, axis=0)

    # Create a cylinder (Blue)
    cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=CYLINDER_RADIUS, height=CYLINDER_HEIGHT)
    cylinder.paint_uniform_color([0, 0, 1])  # Blue

    # Compute rotation matrix to align with PCA direction
    z_axis = np.array([0, 0, 1])  # Default cylinder direction
    axis = np.cross(z_axis, primary_direction)
    angle = np.arccos(np.dot(z_axis, primary_direction) / np.linalg.norm(primary_direction))

    if np.linalg.norm(axis) > 1e-6:
        axis /= np.linalg.norm(axis)
        R = o3d.geometry.get_rotation_matrix_from_axis_angle(axis * angle)
    else:
        R = np.eye(3)

    # Apply transformation
    transformation = np.eye(4)
    transformation[:3, :3] = R
    transformation[:3, 3] = mean_pos  # Move to cluster mean position
    cylinder.transform(transformation)

    cylinders.append(cylinder)

    # Visualize cluster as Open3D point cloud
    cluster_pcd = o3d.geometry.PointCloud()
    cluster_pcd.points = o3d.utility.Vector3dVector(cluster_points)
    cluster_pcd.paint_uniform_color(np.random.rand(3))  # Random color per cluster
    cluster_pcds.append(cluster_pcd)

# Show visualization
o3d.visualization.draw_geometries(cluster_pcds)
o3d.visualization.draw_geometries(cluster_pcds + cylinders)