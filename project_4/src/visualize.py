import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from helpers import (
    load_obj,
    normalize_mesh,
    save_obj,
    random_rotation_matrix,
)
from model import VectorFieldNet


def deform_mesh(model, vertices, n_steps=8, device=None, batch_chunk=20000):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.eval()

    vertices_tensor = torch.from_numpy(vertices.astype(np.float32)).unsqueeze(0).to(device)

    with torch.no_grad():
        source_global_feature = model.encoder(vertices_tensor)

        delta_t = 1.0 / n_steps
        current_points = vertices_tensor
        trajectory = [current_points.squeeze(0).cpu().numpy()]

        for step in range(n_steps):
            time_value = torch.tensor(step * delta_t, device=device)
            n_points = current_points.shape[1]
            updated = torch.empty_like(current_points)

            for start in range(0, n_points, batch_chunk):
                end = min(start + batch_chunk, n_points)
                chunk = current_points[:, start:end, :]
                velocity = model.velocity(chunk, source_global_feature, time_value)
                updated[:, start:end, :] = chunk + delta_t * velocity

            current_points = updated
            trajectory.append(current_points.squeeze(0).cpu().numpy())

    return trajectory


def render_mesh_steps(trajectory, faces, output_path, titles=None, mesh_color='steelblue'):
    n_steps = len(trajectory)
    figure = plt.figure(figsize=(4 * n_steps, 4))

    all_points = np.concatenate(trajectory, axis=0)
    bound = float(np.max(np.abs(all_points))) * 1.05

    for index, vertices in enumerate(trajectory):
        axis = figure.add_subplot(1, n_steps, index + 1, projection='3d')
        triangle_vertices = vertices[faces]
        polygons = Poly3DCollection(
            triangle_vertices,
            alpha=0.85,
            facecolor=mesh_color,
            edgecolor='none',
            linewidths=0.0,
        )
        axis.add_collection3d(polygons)
        axis.set_xlim(-bound, bound)
        axis.set_ylim(-bound, bound)
        axis.set_zlim(-bound, bound)
        axis.set_box_aspect((1, 1, 1))
        axis.set_axis_off()
        if titles is not None:
            axis.set_title(titles[index])

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches='tight')
    plt.close(figure)


def visualize_transformation(
    model_path,
    source_obj_path,
    output_prefix,
    n_integration_steps=8,
    intermediate_steps=(2, 4, 6),
    save_mesh_obj=True,
    randomize_pose=True,
    seed=42,
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    source_vertices, source_faces = load_obj(source_obj_path)
    source_vertices = normalize_mesh(source_vertices)

    if randomize_pose:
        rng = np.random.default_rng(seed)
        rotation = random_rotation_matrix(rng=rng)
        scale = float(rng.uniform(0.8, 1.2))
        source_vertices = (source_vertices @ rotation.T) * scale

    model = VectorFieldNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    trajectory = deform_mesh(model, source_vertices, n_steps=n_integration_steps, device=device)

    selected_indices = [0, *intermediate_steps, n_integration_steps]
    selected_vertices = [trajectory[i] for i in selected_indices]
    titles = [f't = {i / n_integration_steps:.2f}' for i in selected_indices]

    output_image = f'{output_prefix}_steps.png'
    render_mesh_steps(selected_vertices, source_faces, output_image, titles=titles)
    print(f'Saved {output_image}')

    if save_mesh_obj:
        for index, vertices in zip(selected_indices, selected_vertices):
            obj_path = f'{output_prefix}_step_{index:02d}.obj'
            save_obj(obj_path, vertices, source_faces)
        print(f'Saved intermediate OBJ meshes to {output_prefix}_step_*.obj')


if __name__ == '__main__':
    visualizations_dir = '../visualizations'
    os.makedirs(visualizations_dir, exist_ok=True)

    source_objects = [
        ('bunny', '../models/bunny.obj'),
        ('dragon', '../models/dragon_small.obj'),
        ('armadillo', '../models/armadillo_small.obj'),
    ]
    asian_dragon_path = '../models/asian_dragon_really_small.obj'

    for name, source_path in source_objects:
        print(f'Visualizing {name} flow on {name}')
        visualize_transformation(
            model_path=f'../models/{name}_flow.pth',
            source_obj_path=source_path,
            output_prefix=f'{visualizations_dir}/{name}_to_teapot',
        )

    for name, _ in source_objects:
        print(f'Visualizing {name} flow on Asian Dragon')
        visualize_transformation(
            model_path=f'../models/{name}_flow.pth',
            source_obj_path=asian_dragon_path,
            output_prefix=f'{visualizations_dir}/{name}_flow_asian_dragon',
        )
