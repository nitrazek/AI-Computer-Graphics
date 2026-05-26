"""Minimal BVH parser with forward kinematics for CMU MoCap files."""
import re
import numpy as np


class BvhJoint:
    def __init__(self, name, parent=None):
        self.name = name
        self.parent = parent
        self.offset = np.zeros(3, dtype=np.float64)
        self.channels = []
        self.children = []
        self.end_site_offset = None
        self.channel_start = 0  # filled after parsing

    def __repr__(self):
        return f"BvhJoint({self.name})"


def _tokenize(text):
    return re.findall(r"[^\s{}]+|\{|\}", text)


def _parse_hierarchy(tokens, index):
    # tokens[index] should be ROOT or JOINT or End
    keyword = tokens[index]
    if keyword == 'End':
        # End Site { OFFSET x y z }
        assert tokens[index + 1] == 'Site'
        assert tokens[index + 2] == '{'
        assert tokens[index + 3] == 'OFFSET'
        offset = np.array(
            [float(tokens[index + 4]), float(tokens[index + 5]), float(tokens[index + 6])],
            dtype=np.float64,
        )
        assert tokens[index + 7] == '}'
        return None, offset, index + 8

    assert keyword in ('ROOT', 'JOINT')
    name = tokens[index + 1]
    assert tokens[index + 2] == '{'
    joint = BvhJoint(name)
    cursor = index + 3

    while cursor < len(tokens):
        token = tokens[cursor]
        if token == 'OFFSET':
            joint.offset = np.array(
                [float(tokens[cursor + 1]), float(tokens[cursor + 2]), float(tokens[cursor + 3])],
                dtype=np.float64,
            )
            cursor += 4
        elif token == 'CHANNELS':
            count = int(tokens[cursor + 1])
            joint.channels = tokens[cursor + 2:cursor + 2 + count]
            cursor += 2 + count
        elif token in ('JOINT', 'End'):
            child, end_offset, cursor = _parse_hierarchy(tokens, cursor)
            if child is None:
                joint.end_site_offset = end_offset
            else:
                child.parent = joint
                joint.children.append(child)
        elif token == '}':
            cursor += 1
            return joint, None, cursor
        else:
            cursor += 1

    raise ValueError("Unexpected end of hierarchy")


def parse_bvh(path):
    with open(path, 'r') as f:
        text = f.read()

    motion_split = re.split(r"\bMOTION\b", text, maxsplit=1)
    if len(motion_split) != 2:
        raise ValueError(f"No MOTION section in {path}")
    hierarchy_text, motion_text = motion_split

    tokens = _tokenize(hierarchy_text)
    # skip leading HIERARCHY token
    start = 0
    while start < len(tokens) and tokens[start] != 'ROOT':
        start += 1
    root, _, _ = _parse_hierarchy(tokens, start)

    # assign channel indices in DFS order
    channel_counter = [0]

    def assign(joint):
        joint.channel_start = channel_counter[0]
        channel_counter[0] += len(joint.channels)
        for child in joint.children:
            assign(child)

    assign(root)
    total_channels = channel_counter[0]

    # parse motion
    motion_lines = motion_text.strip().splitlines()
    frames_match = re.search(r"Frames:\s*(\d+)", motion_lines[0])
    frame_time_match = re.search(r"Frame Time:\s*([\d.]+)", motion_lines[1])
    if frames_match is None or frame_time_match is None:
        raise ValueError(f"Malformed MOTION header in {path}")
    n_frames = int(frames_match.group(1))
    frame_time = float(frame_time_match.group(1))

    data_lines = motion_lines[2:2 + n_frames]
    motion = np.array(
        [[float(x) for x in line.split()] for line in data_lines],
        dtype=np.float64,
    )
    if motion.shape != (n_frames, total_channels):
        raise ValueError(
            f"Motion shape mismatch in {path}: got {motion.shape}, expected ({n_frames}, {total_channels})"
        )

    return root, motion, frame_time


def _axis_rotation_matrices(axis, angles_rad):
    """Return rotation matrices of shape [F, 3, 3] for one axis."""
    c = np.cos(angles_rad)
    s = np.sin(angles_rad)
    n_frames = angles_rad.shape[0]
    matrices = np.zeros((n_frames, 3, 3), dtype=np.float64)
    if axis == 'X':
        matrices[:, 0, 0] = 1.0
        matrices[:, 1, 1] = c
        matrices[:, 1, 2] = -s
        matrices[:, 2, 1] = s
        matrices[:, 2, 2] = c
    elif axis == 'Y':
        matrices[:, 0, 0] = c
        matrices[:, 0, 2] = s
        matrices[:, 1, 1] = 1.0
        matrices[:, 2, 0] = -s
        matrices[:, 2, 2] = c
    elif axis == 'Z':
        matrices[:, 0, 0] = c
        matrices[:, 0, 1] = -s
        matrices[:, 1, 0] = s
        matrices[:, 1, 1] = c
        matrices[:, 2, 2] = 1.0
    else:
        raise ValueError(axis)
    return matrices


def _local_rotation_per_frame(rotation_channels, values):
    """Build per-frame local rotation matrices [F, 3, 3] from channels (degrees)."""
    n_frames = values.shape[0]
    matrix = np.tile(np.eye(3, dtype=np.float64), (n_frames, 1, 1))
    for index, channel in enumerate(rotation_channels):
        axis = channel[0]  # 'X', 'Y', or 'Z'
        rad = np.deg2rad(values[:, index])
        matrix = matrix @ _axis_rotation_matrices(axis, rad)
    return matrix


def forward_kinematics(root, motion):
    """Compute world-space joint positions for every frame, vectorized across frames.

    Returns a dict mapping joint name -> array of shape (n_frames, 3).
    """
    n_frames = motion.shape[0]
    joints = []

    def collect(joint):
        joints.append(joint)
        for child in joint.children:
            collect(child)

    collect(root)
    positions = {}
    world_rotations = {}

    for joint in joints:
        rotation_channels = [c for c in joint.channels if c.endswith('rotation')]
        position_channels = [c for c in joint.channels if c.endswith('position')]

        if rotation_channels:
            indices = [joint.channels.index(c) + joint.channel_start for c in rotation_channels]
            rotation_values = motion[:, indices]
            local_rotation = _local_rotation_per_frame(rotation_channels, rotation_values)
        else:
            local_rotation = np.tile(np.eye(3, dtype=np.float64), (n_frames, 1, 1))

        local_translation = np.tile(joint.offset, (n_frames, 1))
        if position_channels:
            for axis_index, axis_name in enumerate(('Xposition', 'Yposition', 'Zposition')):
                if axis_name in joint.channels:
                    channel_index = joint.channels.index(axis_name) + joint.channel_start
                    local_translation[:, axis_index] = (
                        local_translation[:, axis_index] + motion[:, channel_index]
                    )

        if joint.parent is None:
            parent_rotation = np.tile(np.eye(3, dtype=np.float64), (n_frames, 1, 1))
            parent_position = np.zeros((n_frames, 3), dtype=np.float64)
        else:
            parent_rotation = world_rotations[joint.parent.name]
            parent_position = positions[joint.parent.name]

        world_position = parent_position + np.einsum('fij,fj->fi', parent_rotation, local_translation)
        world_rotation = np.einsum('fij,fjk->fik', parent_rotation, local_rotation)

        positions[joint.name] = world_position
        world_rotations[joint.name] = world_rotation

    return positions
