import open3d as o3d
import numpy as np
import pickle

def dump_to_pickle(filename, vis_data):
    # vis_data = [pcd, coord_frame] + centroid_spheres

    # Dump PointCloud as numpy arrays.
    pcd = vis_data[0]
    pcd_data = {
        'points': np.asarray(pcd.points),
        'colors': np.asarray(pcd.colors) if pcd.has_colors() else None,
    }

    # For coord_frame, dump the transform (assumed identity if not modified),
    # plus parameters used to re-create it.
    coord_frame = vis_data[1]
    coord_frame_data = {
        'transform': np.eye(4),  # Replace with actual transform if available.
        'size': 0.25,
        'origin': [0.0, 0.0, 0.0],
    }

    # Dump centroid spheres by storing their translation, radius, and color.
    spheres_data = []
    for sphere in vis_data[2:]:
        vertices = np.asarray(sphere.vertices)
        translation = vertices.mean(axis=0)
        # Assume constant radius 0.01; get color from first vertex.
        color = np.asarray(sphere.vertex_colors)[0] if len(sphere.vertex_colors) > 0 else [0.25, 0.25, 0.25]
        spheres_data.append({
        'translation': translation,
        'radius': 0.01,
        'color': color,
        })

    data = {
        'pcd': pcd_data,
        'coord_frame': coord_frame_data,
        'centroid_spheres': spheres_data,
    }
    with open(filename, 'wb') as f:
        pickle.dump(data, f)


def load_from_pickle(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)

    # Re-create point cloud.
    pcd_data = data['pcd']
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pcd_data['points'])
    if pcd_data['colors'] is not None:
        pcd.colors = o3d.utility.Vector3dVector(pcd_data['colors'])

    # Re-create coordinate frame.
    cf_data = data['coord_frame']
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=cf_data.get('size', 0.25),
        origin=np.array(cf_data.get('origin', [0.0, 0.0, 0.0]))
    )
    T = cf_data.get('transform', np.eye(4))
    if not np.allclose(T, np.eye(4)):
        coord_frame.transform(T)

    # Re-create centroid spheres.
    centroid_spheres = []
    for s in data['centroid_spheres']:
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=s.get('radius', 0.01))
        sphere.translate(s['translation'])
        sphere.paint_uniform_color(s.get('color', [0.25, 0.25, 0.25]))
        centroid_spheres.append(sphere)

    return [pcd, coord_frame] + centroid_spheres
