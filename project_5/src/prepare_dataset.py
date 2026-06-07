"""Scan CMU mocap index and prepare 15-joint motion windows for training/validation."""
import os
import re
import sys
import numpy as np

from bvh import parse_bvh, forward_kinematics
from skeleton import (
    Joint,
    CMU_JOINT_MAP,
    LABEL_NAMES,
    SEQUENCE_LENGTH,
    NUM_JOINTS,
)


RAW_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw')
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')
INDEX_FILE = os.path.join(RAW_DIR, 'cmu-mocap-index-text.txt')

# Target frame rate for the generated animation (CMU mocap is recorded at 120 fps).
TARGET_FPS = 24
SOURCE_FPS = 120
FRAME_STRIDE = SOURCE_FPS // TARGET_FPS  # 5

# Sliding window stride (in target frames) when chopping long sequences into clips.
WINDOW_STRIDE = SEQUENCE_LENGTH // 2

# Global coordinate scale (CMU mocap is in inches, ~70 inches tall character).
COORDINATE_SCALE = 1.0 / 40.0

VALIDATION_FRACTION = 0.15


def classify_motions(index_path):
    """Return list of (subject, clip_id, label) entries from the CMU index file."""
    entries = []
    pattern = re.compile(r"^(\d+)_(\d+)\s+(.*)$")

    with open(index_path, 'r', encoding='utf-8') as f:
        for line in f:
            match = pattern.match(line.strip())
            if not match:
                continue
            subject = match.group(1)
            clip = match.group(2)
            description = match.group(3).lower()
            has_walk = re.search(r"\bwalk", description) is not None
            has_jump = re.search(r"\bjump", description) is not None
            if has_walk and not has_jump:
                label = 'walk'
            elif has_jump and not has_walk:
                label = 'jump'
            else:
                continue
            entries.append((subject, clip, label))
    return entries


def extract_skeleton_positions(bvh_path):
    """Parse a BVH file and return per-frame joint positions of shape [F, 15, 3]."""
    root, motion, _frame_time = parse_bvh(bvh_path)
    world_positions = forward_kinematics(root, motion)

    n_frames = motion.shape[0]
    output = np.zeros((n_frames, NUM_JOINTS, 3), dtype=np.float32)
    for joint_index, cmu_name in CMU_JOINT_MAP.items():
        if cmu_name not in world_positions:
            raise KeyError(f"Joint {cmu_name} not found in {bvh_path}")
        output[:, int(joint_index), :] = world_positions[cmu_name].astype(np.float32)
    return output


def normalize_clip(clip):
    """Remove horizontal trajectory, swap to Z-up, and scale to roughly [-1, 1].

    clip: [F, 15, 3] in BVH world space (Y-up, units of inches).
    Returns: [F, 15, 3] (Z-up, scaled).
    """
    pelvis_xz = clip[:, int(Joint.PELVIS), [0, 2]]  # shape [F, 2]
    clip = clip.copy()
    clip[:, :, 0] -= pelvis_xz[:, 0:1]
    clip[:, :, 2] -= pelvis_xz[:, 1:2]

    # Swap axes so the vertical (originally Y) becomes Z in our output.
    swapped = np.empty_like(clip)
    swapped[..., 0] = clip[..., 0]
    swapped[..., 1] = clip[..., 2]
    swapped[..., 2] = clip[..., 1]

    swapped *= COORDINATE_SCALE
    # Subtract the mean ankle height so the character roughly sits on z=0 plane.
    ankle_z = 0.5 * (
        swapped[:, int(Joint.LEFT_ANKLE), 2] + swapped[:, int(Joint.RIGHT_ANKLE), 2]
    )
    swapped[..., 2] -= ankle_z.min()
    return swapped


def make_windows(clip, variance_threshold=0.001):
    """Resample to target fps and slice into fixed-length windows of [SEQUENCE_LENGTH, 15, 3]."""
    downsampled = clip[::FRAME_STRIDE]
    if downsampled.shape[0] < SEQUENCE_LENGTH:
        return []
    windows = []
    for start in range(0, downsampled.shape[0] - SEQUENCE_LENGTH + 1, WINDOW_STRIDE):
        window = downsampled[start:start + SEQUENCE_LENGTH].astype(np.float32).copy()
        
        # --- NEW: Check for actual movement ---
        # Calculate the variance of all joint positions over the time axis (axis 0).
        # A high variance means the joints are moving; low variance means they are static.
        motion_variance = np.var(window, axis=0).mean()
        
        if motion_variance < variance_threshold:
            # Skip this window because the character is mostly standing still
            continue
        # --------------------------------------
        
        # Center the window horizontally based on the pelvis of the first frame
        pelvis_x0 = window[0, int(Joint.PELVIS), 0]
        pelvis_y0 = window[0, int(Joint.PELVIS), 1]
        window[..., 0] -= pelvis_x0
        window[..., 1] -= pelvis_y0
        
        windows.append(window)
    return windows


def save_split(name, samples_by_label):
    out_dir = os.path.join(PROCESSED_DIR, name)
    os.makedirs(out_dir, exist_ok=True)
    for label, samples in samples_by_label.items():
        if not samples:
            print(f"[warn] No {label} samples for split '{name}'")
            continue
        stacked = np.stack(samples, axis=0)
        path = os.path.join(out_dir, f"{label}.npy")
        np.save(path, stacked)
        print(f"Saved {path}: {stacked.shape}")


def main():
    if not os.path.isdir(RAW_DIR):
        sys.exit(f"Raw data folder not found: {RAW_DIR}")

    entries = classify_motions(INDEX_FILE)
    print(f"Found {len(entries)} candidate clips in index")

    rng = np.random.default_rng(0)
    rng.shuffle(entries)
    split_point = int(len(entries) * (1.0 - VALIDATION_FRACTION))
    train_entries = entries[:split_point]
    validation_entries = entries[split_point:]
    print(f"Training clips: {len(train_entries)}, validation clips: {len(validation_entries)}")

    def collect(entry_list):
        samples_by_label = {label: [] for label in LABEL_NAMES}
        for subject, clip_id, label in entry_list:
            bvh_path = os.path.join(RAW_DIR, subject, f"{subject}_{clip_id}.bvh")
            if not os.path.isfile(bvh_path):
                continue
            try:
                positions = extract_skeleton_positions(bvh_path)
            except Exception as error:
                print(f"[skip] {bvh_path}: {error}")
                continue
            normalized = normalize_clip(positions)
            windows = make_windows(normalized)
            samples_by_label[label].extend(windows)
        return samples_by_label

    train_samples = collect(train_entries)
    validation_samples = collect(validation_entries)

    for label in LABEL_NAMES:
        print(
            f"{label}: train windows={len(train_samples[label])}, "
            f"validation windows={len(validation_samples[label])}"
        )

    save_split('training', train_samples)
    save_split('validation', validation_samples)


if __name__ == '__main__':
    main()
