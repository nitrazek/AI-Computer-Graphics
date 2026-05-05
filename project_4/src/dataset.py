import numpy as np
import torch
from torch.utils.data import Dataset

from helpers import (
    load_obj,
    normalize_mesh,
    sample_points_from_mesh,
    random_rotation_matrix,
)


class ShapeFlowDataset(Dataset):
    def __init__(
        self,
        source_obj_path,
        target_obj_path,
        n_source_points=1024,
        n_target_points=1024,
        samples_per_epoch=512,
        scale_range=(0.6, 1.4),
        seed=None,
        augment=True,
    ):
        source_vertices, source_faces = load_obj(source_obj_path)
        target_vertices, target_faces = load_obj(target_obj_path)

        self.source_vertices = normalize_mesh(source_vertices)
        self.source_faces = source_faces
        self.target_vertices = normalize_mesh(target_vertices)
        self.target_faces = target_faces

        self.n_source_points = n_source_points
        self.n_target_points = n_target_points
        self.samples_per_epoch = samples_per_epoch
        self.scale_range = scale_range
        self.augment = augment
        self.seed = seed

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, index):
        if self.seed is not None:
            rng = np.random.default_rng(self.seed + index)
        else:
            rng = np.random.default_rng()

        source_points = sample_points_from_mesh(
            self.source_vertices, self.source_faces, self.n_source_points, rng=rng
        )
        target_points = sample_points_from_mesh(
            self.target_vertices, self.target_faces, self.n_target_points, rng=rng
        )

        if self.augment:
            rotation = random_rotation_matrix(rng=rng)
            scale = float(rng.uniform(self.scale_range[0], self.scale_range[1]))
            source_points = (source_points @ rotation.T) * scale

        return (
            torch.from_numpy(source_points.astype(np.float32)),
            torch.from_numpy(target_points.astype(np.float32)),
        )
