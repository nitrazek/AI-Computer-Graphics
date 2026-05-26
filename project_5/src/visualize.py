"""Stickman visualization helpers (3D animation as gif and pose grids as png)."""
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from skeleton import JOINT_CONNECTIONS


def animate_skeleton_3d(tensor_data, output_filename=None, fps=24):
    """
    Creates a 3D stickman animation from a coordinate tensor.

    tensor_data: numpy array of shape [T, 15, 3] (T - number of frames)
    Currently supported output type: .gif
    """
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_zlim(-1.5, 1.5)
    ax.set_box_aspect([1, 1, 1])
    ax.set_title("3D Stickman Motion Visualization")
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')

    points_scatter = ax.scatter([], [], [], c='red', s=40, zorder=3)
    lines = [
        ax.plot([], [], [], c='blue', lw=2, zorder=2)[0]
        for _ in range(len(JOINT_CONNECTIONS))
    ]

    def init():
        points_scatter._offsets3d = ([], [], [])
        for line in lines:
            line.set_data(np.array([]), np.array([]))
            line.set_3d_properties(np.array([]))
        return [points_scatter] + lines

    def update(frame_idx):
        frame_data = tensor_data[frame_idx]
        xs = frame_data[:, 0]
        ys = frame_data[:, 1]
        zs = frame_data[:, 2]
        points_scatter._offsets3d = (xs, ys, zs)
        for i, (start_joint, end_joint) in enumerate(JOINT_CONNECTIONS):
            x_coords = np.array([frame_data[int(start_joint), 0], frame_data[int(end_joint), 0]])
            y_coords = np.array([frame_data[int(start_joint), 1], frame_data[int(end_joint), 1]])
            z_coords = np.array([frame_data[int(start_joint), 2], frame_data[int(end_joint), 2]])
            lines[i].set_data(x_coords, y_coords)
            lines[i].set_3d_properties(z_coords)
        return [points_scatter] + lines

    n_frames = tensor_data.shape[0]
    anim = animation.FuncAnimation(
        fig, update, frames=n_frames, init_func=init, blit=False, interval=1000 / fps
    )

    if output_filename:
        os.makedirs(os.path.dirname(output_filename), exist_ok=True)
        anim.save(output_filename, writer='pillow', fps=fps)
    plt.close(fig)
    return anim


def render_pose_strip(tensor_data, output_path, frame_indices=None, title=None):
    """Render a strip of selected poses to a single PNG file (no animation)."""
    if frame_indices is None:
        frame_indices = np.linspace(0, tensor_data.shape[0] - 1, 8, dtype=int)

    fig = plt.figure(figsize=(3 * len(frame_indices), 3))
    if title:
        fig.suptitle(title)

    for column, frame_index in enumerate(frame_indices):
        ax = fig.add_subplot(1, len(frame_indices), column + 1, projection='3d')
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_zlim(-1.5, 1.5)
        ax.set_box_aspect([1, 1, 1])
        ax.set_axis_off()
        ax.set_title(f"frame {frame_index}")

        frame = tensor_data[frame_index]
        ax.scatter(frame[:, 0], frame[:, 1], frame[:, 2], c='red', s=20, zorder=3)
        for start_joint, end_joint in JOINT_CONNECTIONS:
            ax.plot(
                [frame[int(start_joint), 0], frame[int(end_joint), 0]],
                [frame[int(start_joint), 1], frame[int(end_joint), 1]],
                [frame[int(start_joint), 2], frame[int(end_joint), 2]],
                c='blue',
                lw=1.5,
                zorder=2,
            )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    # Quick smoke-test using the first training sample of each class.
    import os.path
    from skeleton import LABEL_NAMES

    processed = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 'training')
    visualizations = os.path.join(os.path.dirname(__file__), '..', 'visualizations')

    for label in LABEL_NAMES:
        data = np.load(os.path.join(processed, f'{label}.npy'))
        animate_skeleton_3d(
            data[0],
            output_filename=os.path.join(visualizations, f'reference_{label}.gif'),
        )
        render_pose_strip(
            data[0],
            output_path=os.path.join(visualizations, f'reference_{label}.png'),
            title=f'Reference {label}',
        )
        print(f'Saved reference visualizations for {label}')
