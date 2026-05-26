"""1D Conv U-Net diffusion denoiser conditioned on diffusion step and motion class."""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from skeleton import NUM_JOINTS, NUM_LABELS, SEQUENCE_LENGTH


FEATURE_DIM = NUM_JOINTS * 3  # 45


def sinusoidal_time_embedding(timesteps, embedding_dim):
    half = embedding_dim // 2
    device = timesteps.device
    frequencies = torch.exp(
        -math.log(10000.0)
        * torch.arange(half, device=device, dtype=torch.float32)
        / max(half - 1, 1)
    )
    angles = timesteps.float().unsqueeze(1) * frequencies.unsqueeze(0)
    embedding = torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)
    if embedding_dim % 2 == 1:
        embedding = F.pad(embedding, (0, 1))
    return embedding


class ResidualBlock1d(nn.Module):
    def __init__(self, channels, cond_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, channels)
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.cond_projection = nn.Linear(cond_dim, channels)
        self.norm2 = nn.GroupNorm(8, channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x, cond):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.cond_projection(cond).unsqueeze(-1)
        h = self.conv2(F.silu(self.norm2(h)))
        return x + h


class MotionDenoiser(nn.Module):
    def __init__(
        self,
        feature_dim=FEATURE_DIM,
        sequence_length=SEQUENCE_LENGTH,
        num_labels=NUM_LABELS,
        base_channels=128,
        n_blocks=6,
        cond_dim=256,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.sequence_length = sequence_length

        self.time_embedding = nn.Sequential(
            nn.Linear(cond_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )
        self.label_embedding = nn.Embedding(num_labels, cond_dim)

        self.input_projection = nn.Conv1d(feature_dim, base_channels, kernel_size=1)
        self.blocks = nn.ModuleList(
            [ResidualBlock1d(base_channels, cond_dim) for _ in range(n_blocks)]
        )
        self.output_norm = nn.GroupNorm(8, base_channels)
        self.output_projection = nn.Conv1d(base_channels, feature_dim, kernel_size=1)

        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)
        self._cond_dim = cond_dim

    def forward(self, motion, timesteps, labels):
        # motion: [B, T, 15, 3]
        batch_size = motion.shape[0]
        x = motion.reshape(batch_size, self.sequence_length, self.feature_dim)
        x = x.transpose(1, 2)  # [B, C, T]

        time_emb = sinusoidal_time_embedding(timesteps, self._cond_dim)
        time_emb = self.time_embedding(time_emb)
        label_emb = self.label_embedding(labels)
        cond = time_emb + label_emb

        h = self.input_projection(x)
        for block in self.blocks:
            h = block(h, cond)
        h = self.output_projection(F.silu(self.output_norm(h)))

        h = h.transpose(1, 2).reshape(batch_size, self.sequence_length, NUM_JOINTS, 3)
        return h
