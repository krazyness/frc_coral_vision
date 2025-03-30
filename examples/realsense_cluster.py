USE_OFFICIAL_PYREALSENSE=True

if USE_OFFICIAL_PYREALSENSE:
    import pyrealsense2 as rs
else:
    from rs_python import RSCam


from pickle_utils import load_from_pickle, dump_to_pickle
from matplotlib import pyplot as plt
import open3d as o3d
import numpy as np
from scipy.spatial.transform import Rotation
from sklearn.cluster import DBSCAN

class CylinderDetector:
    def __init__(self, 
                 camera_angle_deg = 22.5, 
                 min_height = 0.12256, 
                 max_height = 0.18288,
                 eps=0.1,
                 min_samples=10):
        
        self.min_height = min_height
        self.max_height = max_height
        self.camera_angle_deg = camera_angle_deg
        self.clipping_distance = 1.5

        if USE_OFFICIAL_PYREALSENSE:
            self.pipeline = rs.pipeline()
            config = rs.config()
            pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
            pipeline_profile = config.resolve(pipeline_wrapper)
            device = pipeline_profile.get_device()
            
            found_rgb = False
            for s in device.sensors:
                if s.get_info(rs.camera_info.name) == 'RGB Camera':
                    found_rgb = True
                    break
            if not found_rgb:
                print("The demo requires Depth camera with Color sensor")
                exit(0)

            config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
            config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

            profile = self.pipeline.start(config)
            
            depth_sensor = profile.get_device().first_depth_sensor()
            self.depth_scale = depth_sensor.get_depth_scale()
            
            
            align_to = rs.stream.color
            self.align = rs.align(align_to)
            
            
            profile = self.pipeline.get_active_profile()
            depth_stream = profile.get_stream(rs.stream.depth)
            intrinsics = depth_stream.as_video_stream_profile().get_intrinsics()
            
            fx = intrinsics.fx
            fy = intrinsics.fy
            cx = intrinsics.ppx
            cy = intrinsics.ppy
        else:
            self.depth_scale = 0.001
            self.rs_camera = RSCam()
            
            K = self.rs_camera.GetK()
            fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
            
            
        self.intrinsics = o3d.camera.PinholeCameraIntrinsic(640, 480, fx, fy, cx, cy)
        
        self.camera_pose = np.eye(4)
        self.camera_pose[:3, :3] = Rotation.from_euler('xyz', [-self.camera_angle_deg, 0, 0], True).as_matrix()
        
        self.clusterer = DBSCAN(eps=eps, min_samples=min_samples)


    def get_all_cylinders(self, camera_height: float = 0, visualize=False, save_vis_data_path=None):
                
        if USE_OFFICIAL_PYREALSENSE:
            frames = self.pipeline.wait_for_frames()
            aligned_frames = self.align.process(frames)
            aligned_depth_frame = aligned_frames.get_depth_frame()
            
            depth = np.array(aligned_depth_frame.get_data()).astype(np.float32) * self.depth_scale    
        else:
            depth = self.rs_camera.GetDepth().astype(np.float32) * self.depth_scale
        
        depth[depth > self.clipping_distance] = 0
        
        pcd = o3d.geometry.PointCloud.create_from_depth_image(
            o3d.geometry.Image(depth), self.intrinsics
        )        

        cam_pose = self.camera_pose.copy()
        cam_pose[2, 3] = camera_height
        pcd.transform(cam_pose)

        all_points = np.array(pcd.points)

        # condition: min_height <= y <= max_height
        filtered_points = all_points[(all_points[:, 1] >= self.min_height) & (all_points[:, 1] <= self.max_height)]
        
        # Cluster the filtered points using DBSCAN
        self.clusterer.fit(filtered_points)
        labels = self.clusterer.labels_
        
        unique_labels = np.unique(labels)

        
        # Compute centroids for clusters (ignoring noise)
        centroids = []
        for label in unique_labels:
            if label == -1:
                continue
            cluster_points = filtered_points[labels == label]
            centroid = cluster_points.mean(axis=0)
            centroids.append(centroid)
        
        # Visualize the clusters along with the coordinate frame
        centroid_spheres = []
        for c in centroids:
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
            sphere.translate(c)
            sphere.paint_uniform_color([0.25, 0.25, 0.25])  # white
            centroid_spheres.append(sphere)

        if visualize or save_vis_data_path is not None:
            color_map = {}
            for label in unique_labels:
                if label == -1:
                    color_map[label] = np.array([0.0, 0.0, 0.0])
                else:
                    color_map[label] = np.array(plt.get_cmap("tab10")(label % 10)[:3])
                    
            colors = np.array([color_map[label] for label in labels])
            
            # Update point cloud colors and points
            pcd.points = o3d.utility.Vector3dVector(filtered_points)
            pcd.colors = o3d.utility.Vector3dVector(colors)
            coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
                size=0.25, origin=np.array([0., 0., 0.])
            )
            if visualize:
                o3d.visualization.draw_geometries([pcd, coord_frame] + centroid_spheres)
            if save_vis_data_path is not None:
                dump_to_pickle(save_vis_data_path, [pcd, coord_frame] + centroid_spheres)
                
        return centroids
    
if __name__ == "__main__":
    detector = CylinderDetector()
    centroids = detector.get_all_cylinders(0.16256, True)
    print("Centroids:", centroids)


    # example of dumping and loading data
    # centroids = detector.get_all_cylinders(0.16256, False, "debug.pickle")
    
    # viz_data = load_from_pickle("debug.pickle")
    # o3d.visualization.draw_geometries(viz_data)