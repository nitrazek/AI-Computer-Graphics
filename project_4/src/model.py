import torch
import torch.nn as nn


class PointNetEncoder(nn.Module):
    def __init__(self, in_channels=3, hidden=128, feature_dim=256):
        super().__init__()
        self.point_mlp = nn.Sequential(
            nn.Linear(in_channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, feature_dim),
        )

    def forward(self, points):
        per_point_features = self.point_mlp(points)
        global_feature, _ = per_point_features.max(dim=1)
        return global_feature


class VectorFieldNet(nn.Module):
    def __init__(self, hidden=256, feature_dim=256, time_embedding_dim=16):
        super().__init__()
        self.encoder = PointNetEncoder(in_channels=3, hidden=128, feature_dim=feature_dim)
        self.time_embedding_dim = time_embedding_dim

        input_dim = 3 + feature_dim + time_embedding_dim
        self.velocity_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 3),
        )

        # Zero-init the last layer so the initial flow is the identity.
        nn.init.zeros_(self.velocity_mlp[-1].weight)
        nn.init.zeros_(self.velocity_mlp[-1].bias)

    def time_features(self, time_value, batch_size, n_points, device):
        frequencies = torch.linspace(1.0, 8.0, self.time_embedding_dim // 2, device=device)
        angles = 2.0 * torch.pi * time_value * frequencies
        embedding = torch.cat([torch.sin(angles), torch.cos(angles)], dim=0)
        return embedding.view(1, 1, -1).expand(batch_size, n_points, -1)

    def velocity(self, current_points, source_global_feature, time_value):
        batch_size, n_points, _ = current_points.shape
        device = current_points.device

        time_feature = self.time_features(time_value, batch_size, n_points, device)
        repeated_global = source_global_feature.unsqueeze(1).expand(-1, n_points, -1)

        mlp_input = torch.cat([current_points, repeated_global, time_feature], dim=-1)
        return self.velocity_mlp(mlp_input)

    def forward(self, source_points, n_steps=10, return_trajectory=False):
        source_global_feature = self.encoder(source_points)

        delta_t = 1.0 / n_steps
        current_points = source_points
        trajectory = [current_points]

        for step in range(n_steps):
            time_value = torch.tensor(step * delta_t, device=source_points.device)
            velocity = self.velocity(current_points, source_global_feature, time_value)
            current_points = current_points + delta_t * velocity
            if return_trajectory:
                trajectory.append(current_points)

        if return_trajectory:
            return current_points, trajectory
        return current_points
