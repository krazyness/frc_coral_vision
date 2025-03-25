# from rs_python import RSCam
from cluster import PointCloudProcessor
import open3d as o3d
import numpy as np
import pyrealsense2 as rs

class CylinderDetector:
    def __init__(self):
        self.processor = PointCloudProcessor()
        
        self.pipeline = rs.pipeline()
        config = rs.config()
        
        pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
        pipeline_profile = config.resolve(pipeline_wrapper)
        device = pipeline_profile.get_device()
        device_product_line = str(device.get_info(rs.camera_info.product_line))
        
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
        print("Depth Scale is: " , self.depth_scale)
        
        self.clipping_distance = 1.5
        
        align_to = rs.stream.color
        self.align = rs.align(align_to)
        
        
        profile = self.pipeline.get_active_profile()
        depth_stream = profile.get_stream(rs.stream.depth)
        intrinsics = depth_stream.as_video_stream_profile().get_intrinsics()
        
        fx = intrinsics.fx
        fy = intrinsics.fy
        cx = intrinsics.ppx
        cy = intrinsics.ppy
        
        self.intrinsics = o3d.camera.PinholeCameraIntrinsic(640, 480, fx, fy, cx, cy)

    def get_all_cylinders(self, camera_pose = np.eye(4)):
        
        # depth = self.cam.GetDepth().astype(float) / 1000.
        # frames = self.pipeline.wait_for_frames()
        # aligned_frames = self.align.process(frames)
        # aligned_depth_frame = aligned_frames.get_depth_frame()
        
        # depth = np.array(aligned_depth_frame.get_data()).astype(np.float32) * self.depth_scale
        
        # np.save("depth.npy", depth)
        depth = np.load("depth.npy")
        
        depth[depth > self.clipping_distance] = 0
        print("Got Depth")
        
        # target_num_points = 10000
        
        pcd = o3d.geometry.PointCloud.create_from_depth_image(o3d.geometry.Image(depth), self.intrinsics)
        
        # ratio = target_num_points / current_num_points
        # ratio = pcd.random_down_sample(ratio)
        all_points = np.array(pcd.points) + 0.16256

        # condition: 64in <= y <= 72in
        filtered_points_greater_64 = all_points[all_points[:, 1] >= 0.12256]
        filtered_points = filtered_points_greater_64[filtered_points_greater_64[:, 1] <= 0.18288]
        
        pcd.points = o3d.utility.Vector3dVector(filtered_points)
        o3d.visualization.draw_geometries([pcd])
        return self.processor.cluster(filtered_points, camera_pose, eps=0.1, min_samples=5)

if __name__ == "__main__":
    detector = CylinderDetector()
    cylinders_with_transforms = detector.get_all_cylinders()
    print("Got Cylinders:", cylinders_with_transforms)
    
    cluster_pcds = []
    for _, pts, _ in cylinders_with_transforms:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        cluster_pcds.append(pcd)
      
    detector.processor.visualize(cluster_pcds, cylinders_with_transforms)