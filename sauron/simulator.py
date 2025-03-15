import numpy as np
import pygame
import open3d as o3d
from estimate_plane import estimate_plane

class LidarSimulator:
    def __init__(self, res_x=8, res_y=8, fov_x=60, fov_y=60, noise_std=0.1):
        self.res_x = res_x
        self.res_y = res_y
        self.fov_x = fov_x
        self.fov_y = fov_y
        self.noise_std = noise_std
        self.rays = self._generate_lidar_rays()
        
        self.hex_center = np.array([0, 0, 0])
        self.hex_radius = 5
        self.plane_points, self.plane_normals = self._hexagon_planes(self.hex_center, self.hex_radius)
        self.plane_frames = self._compute_plane_frames()

    def _generate_lidar_rays(self):
        angles_x = np.radians(np.linspace(-self.fov_x / 2, self.fov_x / 2, self.res_x))
        angles_y = np.radians(np.linspace(-self.fov_y / 2, self.fov_y / 2, self.res_y))

        sin_ax, cos_ax = np.sin(angles_x), np.cos(angles_x)
        sin_ay, cos_ay = np.sin(angles_y), np.cos(angles_y)

        sin_ax, sin_ay = np.meshgrid(sin_ax, sin_ay, indexing='ij')
        cos_ax, cos_ay = np.meshgrid(cos_ax, cos_ay, indexing='ij')

        directions = np.stack([cos_ax * cos_ay, sin_ax * cos_ay, sin_ay], axis=-1)

        return directions / np.linalg.norm(directions, axis=-1, keepdims=True)

    def _hexagon_planes(self, center, radius):
        angles = np.radians(np.arange(0, 360, 60))
        edge_points = center + radius * np.column_stack([np.cos(angles), np.sin(angles), np.zeros_like(angles)])
        normals = np.column_stack([np.cos(angles), np.sin(angles), np.zeros_like(angles)])
        return edge_points, normals

    def _compute_plane_frames(self):
        frames = []
        for normal in self.plane_normals:
            u = np.array([0, 0, 1])
            v = np.cross(normal, u)
            frames.append(np.column_stack((u, v, normal)))
        return frames

    def _ray_plane_intersect(self, ray_origins, ray_dirs, plane_points, plane_normals):
        denom = np.dot(ray_dirs, plane_normals.T)
        valid = denom < 0
        pt_minus_origin = plane_points[None] - ray_origins[:, None]
        num = np.sum(pt_minus_origin * plane_normals[None], axis=-1)
        t = np.where(valid, num / denom, np.inf)
        intersections = ray_origins[:, None, :] + t[..., None] * ray_dirs[:, None, :]
        return intersections, t

    def get_robot_rays(self, sensor_rotation):
        ray_dirs = self.rays.reshape(-1, 3)  
        rotation_matrix = np.array([
            [np.cos(sensor_rotation), -np.sin(sensor_rotation), 0],
            [np.sin(sensor_rotation), np.cos(sensor_rotation), 0],
            [0, 0, 1]
        ])
        ray_dirs = (rotation_matrix @ ray_dirs.T).T
        
        return ray_dirs
    
    def simulate(self, sensor_origin, sensor_rotation):
        ray_dirs = self.get_robot_rays(sensor_rotation)

        sensor_origins = np.tile(sensor_origin, (ray_dirs.shape[0], 1))
        intersections, distances = self._ray_plane_intersect(sensor_origins, ray_dirs, self.plane_points, self.plane_normals)

        local_pts = np.concatenate([(self.plane_frames[i].T @ (intersections[:, i] - self.plane_points[i]).T).T[:, None] for i in range(6)], 1)
        dist_along_hex_sides = local_pts[..., 1]
        hex_plane_width = self.hex_radius * np.tan(np.radians(30))
        valid_mask = np.abs(dist_along_hex_sides) < hex_plane_width
        distances[~valid_mask] = np.inf

        min_dist = np.min(np.where(distances > 0, distances, np.inf), axis=1)
        min_dist += np.random.normal(0, self.noise_std, min_dist.shape)
        min_dist = min_dist.reshape(self.res_x, self.res_y)

        hit_points = sensor_origins + min_dist.reshape(-1, 1) * ray_dirs
        valid = (hit_points != np.inf).all(1)
        hit_points = hit_points[valid]
        min_dist[~valid.reshape(self.res_x, self.res_y)] = 0.
        
        T = robot.as_se3()
        hit_points_homog = np.hstack((hit_points, np.ones_like(hit_points[:, 0:1])))
        
        return (np.linalg.inv(T) @ hit_points_homog.T).T[:, :3]

class SE2Robot:
    def __init__(self, start_pos=np.array([0, 0, 1]), start_theta=0):
        self.position = start_pos.astype(float)
        self.theta = start_theta
        self.speed = 0.2  
        self.rotation_speed = 0.1  

    def move(self, forward, rotate):
        self.theta += rotate * self.rotation_speed
        direction = np.array([np.cos(self.theta), np.sin(self.theta), 0])
        self.position += forward * self.speed * direction
        
    def as_se2(self):
        return np.array([
            [np.cos(self.theta), -np.sin(self.theta), self.position[0]],
            [np.sin(self.theta), np.cos(self.theta), self.position[1]],
            [0, 0, 1]
        ])
        
    def as_se3(self):
        result = np.eye(4)
        se2 = self.as_se2()
        result[:2, :2] = se2[:2, :2]
        result[:2, 3] = se2[:2, 2]
        return result

def draw_slider(surface, x, y, width, min_val, max_val, value, label="Noise (m)"):
    pygame.draw.rect(surface, (100, 100, 100), (x, y, width, 10))  
    knob_x = x + int((value - min_val) / (max_val - min_val) * width)
    pygame.draw.circle(surface, (255, 255, 255), (knob_x, y + 5), 7)

    font = pygame.font.SysFont(None, 24)
    text = font.render(f"{label}: {value:.2f} m", True, (255, 255, 255))
    surface.blit(text, (x + width + 10, y - 5))  

def update_3d_point_cloud(vis, point_cloud, points):
    point_cloud.points = o3d.utility.Vector3dVector(points)
    vis.update_geometry(point_cloud)
    vis.poll_events()
    vis.update_renderer()
    
def draw_hexagon(surface, center, radius, color=(0, 0, 255), width=2):
    """Draws a hexagon centered at center with radius in pixels."""
    hex_points = []
    for i in range(6):
        angle = np.radians(i * 60 + 30)
        rad_corner = radius / np.cos(np.radians(30))
        x = center[0] + rad_corner * np.cos(angle)
        y = center[1] + rad_corner * np.sin(angle)
        hex_points.append((x, y))
    
    pygame.draw.polygon(surface, color, hex_points, width)
    
def draw_robot(surface, position, theta, scale=10):
    """Draws the robot as a triangle indicating position and orientation."""
    x, y = 250 + position[0] * scale, 250 - position[1] * scale
    size = 10  # Robot size in pixels
    
    # Compute triangle vertices
    front = (x + size * np.cos(theta), y - size * np.sin(theta))
    left = (x + size * np.cos(theta + 2.5), y - size * np.sin(theta + 2.5))
    right = (x + size * np.cos(theta - 2.5), y - size * np.sin(theta - 2.5))

    pygame.draw.polygon(surface, (255, 0, 0), [front, left, right])
    
def draw_robot_with_fov(surface, position, theta, lidar_rays, scale=10, alpha=50):
    """Draws the robot and overlays transparent Lidar rays."""
    x, y = 250 + position[0] * scale, 250 - position[1] * scale
    size = 10  # Robot size in pixels
    
    # Compute triangle vertices
    front = (x + size * np.cos(theta), y - size * np.sin(theta))
    left = (x + size * np.cos(theta + 2.5), y - size * np.sin(theta + 2.5))
    right = (x + size * np.cos(theta - 2.5), y - size * np.sin(theta - 2.5))
    
    # Create a transparent surface
    fov_surface = pygame.Surface(surface.get_size(), pygame.SRCALPHA)
    
    # Draw Lidar rays on the transparent surface
    for ray in lidar_rays:
        end_x, end_y = 250 + ray[0] * scale, 250 - ray[1] * scale
        pygame.draw.line(fov_surface, (0, 255, 255, alpha), (x, y), (end_x, end_y), 1)

    # Blit the transparent surface onto the main screen
    surface.blit(fov_surface, (0, 0))
    
    # Draw robot body
    pygame.draw.polygon(surface, (255, 0, 0), [front, left, right])


def draw_plane(surface, plane_center, plane_normal, scale=10, plane_size=3):
    """
    Draw the estimated plane as a filled quadrilateral in 2D visualization.

    Parameters:
    surface : pygame.Surface
        The surface to draw on.
    plane_center : ndarray (3,)
        The center of the plane in world coordinates.
    plane_normal : ndarray (3,)
        The normal vector of the plane in world coordinates.
    scale : int, optional
        Scaling factor for visualization.
    plane_size : float, optional
        The size of the drawn plane in world units.
    """
    # Convert plane center to screen coordinates
    px, py = 250 + int(plane_center[0] * scale), 250 - int(plane_center[1] * scale)

    # Find two perpendicular vectors to construct a visual patch of the plane
    if np.abs(plane_normal[2]) > 0.9:  # If the normal is mostly vertical
        tangent1 = np.array([1, 0, 0])
    else:
        tangent1 = np.cross(plane_normal, np.array([0, 0, 1]))  # First tangent direction

    tangent1 /= np.linalg.norm(tangent1)
    tangent2 = np.cross(plane_normal, tangent1)  # Second tangent direction
    tangent2 /= np.linalg.norm(tangent2)

    # Compute plane corners in world coordinates
    corner_offsets = [
        tangent1 * plane_size + tangent2 * plane_size,
        -tangent1 * plane_size + tangent2 * plane_size,
        -tangent1 * plane_size - tangent2 * plane_size,
        tangent1 * plane_size - tangent2 * plane_size
    ]
    plane_corners = [plane_center[:2] + offset[:2] for offset in corner_offsets]

    # Convert corners to screen coordinates
    screen_corners = [(250 + int(x * scale), 250 - int(y * scale)) for x, y in plane_corners]

    # Draw filled polygon for the plane
    pygame.draw.polygon(surface, (255, 255, 0), screen_corners)  # Semi-transparent blue

    # Draw plane center
    pygame.draw.circle(surface, (255, 255, 0), (px, py), 5)  # Yellow center dot

    # Draw normal vector as an arrow
    normal_end = plane_center[:2] + plane_normal[:2] * 2  # Extend normal for visibility
    nx, ny = 250 + int(normal_end[0] * scale), 250 - int(normal_end[1] * scale)
    pygame.draw.line(surface, (255, 255, 0), (px, py), (nx, ny), 3)  # Yellow normal line

if __name__ == "__main__":
    pygame.init()
    screen = pygame.display.set_mode((500, 500))
    clock = pygame.time.Clock()

    lidar = LidarSimulator(noise_std=0.5)
    robot = SE2Robot(np.array([-7, 0, 0]))

    # Open3D visualization
    # vis = o3d.visualization.Visualizer()
    # vis.create_window()
    # point_cloud = o3d.geometry.PointCloud()
    # vis.add_geometry(point_cloud)

    running = True
    slider_x, slider_y, slider_width = 50, 450, 200
    noise_std = 0.5
    
    mean, normal = None, None
    
    while running:
        screen.fill((0, 0, 0))  
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.MOUSEBUTTONDOWN:
                mx, my = pygame.mouse.get_pos()
                if slider_x <= mx <= slider_x + slider_width and slider_y - 5 <= my <= slider_y + 15:
                    noise_std = (mx - slider_x) / slider_width * 1.0  

        draw_robot_with_fov(screen, robot.position, robot.theta, robot.position + lidar.get_robot_rays(robot.theta)*10)

        keys = pygame.key.get_pressed()
        forward = keys[pygame.K_UP] - keys[pygame.K_DOWN]
        rotate = keys[pygame.K_LEFT] - keys[pygame.K_RIGHT]
        robot.move(forward, rotate)

        lidar.noise_std = noise_std  
        pt_cloud_robot_frame = lidar.simulate(robot.position, robot.theta)
        
        """
        THE IMPORTANT PART!!!
        """
        result = estimate_plane(robot, pt_cloud_robot_frame)
        if result is not None:
            mean, normal = result
        
        if mean and normal:
            draw_plane(screen, mean, normal)
        """
        DONE WITH THE IMPORTANT PART!!!
        """
        
        pt_cloud_robot_frame_homog = np.hstack((pt_cloud_robot_frame, np.ones_like(pt_cloud_robot_frame[:, 0:1])))
        pt_cloud_world_frame = (robot.as_se3() @ pt_cloud_robot_frame_homog.T).T[:, :3]
        
        hex_radius = 50  # Hexagon size in pixels
        hex_center = (250, 250)  # Fixed center in window coordinates
        
        draw_slider(screen, slider_x, slider_y, slider_width, 0, 1, noise_std, "Noise")
        draw_hexagon(screen, hex_center, hex_radius)
        
        for pt in pt_cloud_world_frame:
            if np.isfinite(pt).all():
                px, py = 250 + int(pt[0] * 10), 250 - int(pt[1] * 10)
            pygame.draw.circle(screen, (0, 255, 0), (px, py), 2)
            
        pygame.display.flip()
        clock.tick(30)

        # update_3d_point_cloud(vis, point_cloud, point_cloud_data)

    pygame.quit()
    # vis.destroy_window()
