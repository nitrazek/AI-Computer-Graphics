"""PyTorch dataset that loads pre-processed motion windows for both labels."""
import os
import numpy as np
import torch
from torch.utils.data import Dataset

from skeleton import LABEL_NAMES, LABEL_TO_INDEX, SEQUENCE_LENGTH, NUM_JOINTS


PROCESSED_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')


class MotionDataset(Dataset):
    def __init__(self, split='training'):
        super().__init__()
        self.split = split
        self.samples = []
        self.labels = []

        for label_name in LABEL_NAMES:
            path = os.path.join(PROCESSED_DIR, split, f"{label_name}.npy")
            if not os.path.isfile(path):
                raise FileNotFoundError(f"Missing processed file: {path}")
            data = np.load(path)
            if data.ndim != 4 or data.shape[1:] != (SEQUENCE_LENGTH, NUM_JOINTS, 3):
                raise ValueError(
                    f"Unexpected shape {data.shape} in {path}, "
                    f"expected (N, {SEQUENCE_LENGTH}, {NUM_JOINTS}, 3)"
                )
            self.samples.append(data.astype(np.float32))
            self.labels.append(
                np.full(data.shape[0], LABEL_TO_INDEX[label_name], dtype=np.int64)
            )

        self.samples = np.concatenate(self.samples, axis=0)
        self.labels = np.concatenate(self.labels, axis=0)

    def __len__(self):
        return self.samples.shape[0]

    def __getitem__(self, index):
        return (
            torch.from_numpy(self.samples[index]),
            int(self.labels[index]),
        )

    def label_counts(self):
        return {
            name: int(np.sum(self.labels == LABEL_TO_INDEX[name])) for name in LABEL_NAMES
        }

    def class_balanced_weights(self):
        """Per-sample weights for WeightedRandomSampler to balance class frequency."""
        counts = self.label_counts()
        per_label_weight = {
            LABEL_TO_INDEX[name]: 1.0 / max(counts[name], 1) for name in LABEL_NAMES
        }
        return np.array([per_label_weight[int(label)] for label in self.labels], dtype=np.float64)
