import cv2
import numpy as np
import open3d as o3d
import os, sys
from scipy.spatial.transform import Rotation 
import yaml
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir))

from utils.load_tutorial_data import load_tutorial_data


pcd_colored = load_tutorial_data()
points = np.array(pcd_colored.points)

vertical_slices = np.array([-0.395, 0.031, 0.19])  # N boundaries
N = len(vertical_slices)  # This creates N+1 bins

# Get colormap
cmap = plt.get_cmap("viridis")
colors_map = [cmap(i / (2 * N + 2))[:3] for i in range(2 * (N + 1))]  # More colors for subdivisions

subclouds = []
cylinders = []
excluded_masks = np.zeros(len(points), dtype=bool)  # Tracks which points are excluded

bin_edges = np.concatenate(([points[:, 2].min()], vertical_slices, [points[:, 2].max()]))

for i in range(N + 1):
    # Apply exclusion logic only on the first slice
    if i == 0:
        points_filtered = points
    else:
        points_filtered = points[~excluded_masks]  # Remove excluded points from future slices

    # Apply bin slicing after filtering
    mask = (points_filtered[:, 2] >= bin_edges[i]) & (points_filtered[:, 2] < bin_edges[i + 1])
    if np.any(mask):
        sub_pcd = o3d.geometry.PointCloud()
        sub_pcd.points = o3d.utility.Vector3dVector(points_filtered[mask])
        
        xy = points_filtered[mask][:, :2]

        # First round of DBSCAN
        dbscan = DBSCAN(eps=0.1, min_samples=15).fit(xy)
        labels = dbscan.labels_
        
        unique_labels = set(labels)
        num_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)

        # Generate unique colors for each cluster
        cluster_colors = np.random.rand(num_clusters, 3)  # Random colors for clusters
        point_colors = np.zeros((len(labels), 3))  # Default to black for noise

        for cluster_label in unique_labels:
            if cluster_label == -1:
                continue  # Skip noise
            
            cluster_mask = labels == cluster_label
            cluster_points = points_filtered[mask][cluster_mask]

            # Compute cluster mean position
            mean_pos = np.mean(cluster_points, axis=0)

            # Apply PCA to determine the orientation of the cluster
            pca = PCA(n_components=3)
            pca.fit(cluster_points)
            primary_direction = pca.components_[0]  # Principal axis

            # If this is the first slice, define the exclusion zone
            if i == 0:
                diffs = points - mean_pos  # Vector from mean to each point
                proj_lengths = np.dot(diffs, primary_direction)  # Project onto the cylinder axis
                proj_points = mean_pos + np.outer(proj_lengths, primary_direction)  # Projected points

                distances = np.linalg.norm(diffs - (proj_points - mean_pos), axis=1)  # Perpendicular distance
                exclusion_mask = distances < EXCLUSION_RADIUS  # Points within the exclusion radius
                excluded_masks |= exclusion_mask  # Mark these points for exclusion in future slices

            # Assign colors to the clusters
            color_idx = cluster_label % num_clusters  # Map to color array
            point_colors[cluster_mask] = cluster_colors[color_idx]

            # Create a cylinder
            cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=CYLINDER_RADIUS, height=CYLINDER_HEIGHT)
            cylinder.paint_uniform_color(cluster_colors[color_idx])  # Assign cluster color

            # Compute rotation to align cylinder with principal direction
            z_axis = np.array([0, 0, 1])  # Default cylinder direction
            axis = np.cross(z_axis, primary_direction)
            angle = np.arccos(np.dot(z_axis, primary_direction) / (np.linalg.norm(z_axis) * np.linalg.norm(primary_direction)))

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

        # Apply cluster colors to the subcloud
        sub_pcd.colors = o3d.utility.Vector3dVector(point_colors)
        subclouds.append(sub_pcd)

# Visualize all slices and cylinders
o3d.visualization.draw_geometries(subclouds + cylinders)