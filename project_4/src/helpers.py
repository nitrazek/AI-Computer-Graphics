import os
import numpy as np
import torch
from scipy.spatial import cKDTree


def load_obj(path):
    vertices = []
    faces = []

    with open(path, 'r') as f:
        for line in f:
            if line.startswith('v '):
                parts = line.split()
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif line.startswith('f '):
                parts = line.split()[1:]
                # OBJ faces can be: "v", "v/vt", "v//vn", "v/vt/vn"; we only need vertex index
                indices = [int(p.split('/')[0]) - 1 for p in parts]
                # Triangulate fan-style for polygons with > 3 vertices
                for i in range(1, len(indices) - 1):
                    faces.append([indices[0], indices[i], indices[i + 1]])

    vertices = np.asarray(vertices, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.int64)
    return vertices, faces


def save_obj(path, vertices, faces=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, 'w') as f:
        for vertex in vertices:
            f.write(f"v {vertex[0]:.6f} {vertex[1]:.6f} {vertex[2]:.6f}\n")
        if faces is not None:
            for face in faces:
                # OBJ uses 1-based indexing
                f.write(f"f {face[0] + 1} {face[1] + 1} {face[2] + 1}\n")


def normalize_mesh(vertices):
    centroid = vertices.mean(axis=0, keepdims=True)
    centered = vertices - centroid
    radius = np.linalg.norm(centered, axis=1).max()
    if radius < 1e-8:
        radius = 1.0
    return (centered / radius).astype(np.float32)


def sample_points_from_mesh(vertices, faces, n_points, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    triangle_vertices = vertices[faces]  # (M, 3, 3)
    edge_a = triangle_vertices[:, 1] - triangle_vertices[:, 0]
    edge_b = triangle_vertices[:, 2] - triangle_vertices[:, 0]
    triangle_areas = 0.5 * np.linalg.norm(np.cross(edge_a, edge_b), axis=1)
    triangle_areas = np.clip(triangle_areas, 1e-12, None)
    probabilities = triangle_areas / triangle_areas.sum()

    chosen = rng.choice(len(faces), size=n_points, p=probabilities)
    u = rng.random(n_points).astype(np.float32)
    v = rng.random(n_points).astype(np.float32)
    overflow = (u + v) > 1.0
    u[overflow] = 1.0 - u[overflow]
    v[overflow] = 1.0 - v[overflow]
    w = 1.0 - u - v

    a = triangle_vertices[chosen, 0]
    b = triangle_vertices[chosen, 1]
    c = triangle_vertices[chosen, 2]
    points = (w[:, None] * a + u[:, None] * b + v[:, None] * c).astype(np.float32)
    return points


def random_rotation_matrix(rng=None):
    # Shoemake's method.
    if rng is None:
        rng = np.random.default_rng()

    u1, u2, u3 = rng.random(3)
    quaternion = np.array([
        np.sqrt(1.0 - u1) * np.sin(2.0 * np.pi * u2),
        np.sqrt(1.0 - u1) * np.cos(2.0 * np.pi * u2),
        np.sqrt(u1) * np.sin(2.0 * np.pi * u3),
        np.sqrt(u1) * np.cos(2.0 * np.pi * u3),
    ], dtype=np.float32)
    x, y, z, w = quaternion
    rotation = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float32)
    return rotation


def chamfer_distance(points_a, points_b):
    tree_a = cKDTree(points_a)
    tree_b = cKDTree(points_b)
    distances_ab, _ = tree_b.query(points_a, k=1)
    distances_ba, _ = tree_a.query(points_b, k=1)
    return float(np.mean(distances_ab ** 2) + np.mean(distances_ba ** 2))


def voxelize_points(points, resolution=64, bounds=(-1.0, 1.0)):
    low, high = bounds
    points = np.clip(points, low, high - 1e-6)
    indices = ((points - low) / (high - low) * resolution).astype(np.int64)
    indices = np.clip(indices, 0, resolution - 1)

    grid = np.zeros((resolution, resolution, resolution), dtype=bool)
    grid[indices[:, 0], indices[:, 1], indices[:, 2]] = True
    return grid


def iou_and_dice(points_a, points_b, resolution=64, bounds=(-1.0, 1.0)):
    grid_a = voxelize_points(points_a, resolution=resolution, bounds=bounds)
    grid_b = voxelize_points(points_b, resolution=resolution, bounds=bounds)

    intersection = np.logical_and(grid_a, grid_b).sum()
    union = np.logical_or(grid_a, grid_b).sum()
    sum_voxels = grid_a.sum() + grid_b.sum()

    iou = float(intersection / union) if union > 0 else 0.0
    dice = float(2.0 * intersection / sum_voxels) if sum_voxels > 0 else 0.0
    return iou, dice


def chamfer_distance_torch(predicted, target):
    differences = predicted.unsqueeze(2) - target.unsqueeze(1)
    squared_distances = (differences ** 2).sum(dim=-1)

    nearest_target, _ = squared_distances.min(dim=2)  # (B, N)
    nearest_predicted, _ = squared_distances.min(dim=1)  # (B, M)

    return nearest_target.mean() + nearest_predicted.mean()
