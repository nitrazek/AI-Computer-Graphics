"""Skeleton conventions shared across project_5."""
from enum import IntEnum


class Joint(IntEnum):
    HEAD = 0
    NECK = 1
    PELVIS = 2
    RIGHT_SHOULDER = 3
    RIGHT_ELBOW = 4
    RIGHT_WRIST = 5
    LEFT_SHOULDER = 6
    LEFT_ELBOW = 7
    LEFT_WRIST = 8
    RIGHT_HIP = 9
    RIGHT_KNEE = 10
    RIGHT_ANKLE = 11
    LEFT_HIP = 12
    LEFT_KNEE = 13
    LEFT_ANKLE = 14


JOINT_CONNECTIONS = [
    (Joint.PELVIS, Joint.NECK),
    (Joint.NECK, Joint.HEAD),
    (Joint.NECK, Joint.RIGHT_SHOULDER),
    (Joint.RIGHT_SHOULDER, Joint.RIGHT_ELBOW),
    (Joint.RIGHT_ELBOW, Joint.RIGHT_WRIST),
    (Joint.NECK, Joint.LEFT_SHOULDER),
    (Joint.LEFT_SHOULDER, Joint.LEFT_ELBOW),
    (Joint.LEFT_ELBOW, Joint.LEFT_WRIST),
    (Joint.PELVIS, Joint.RIGHT_HIP),
    (Joint.RIGHT_HIP, Joint.RIGHT_KNEE),
    (Joint.RIGHT_KNEE, Joint.RIGHT_ANKLE),
    (Joint.PELVIS, Joint.LEFT_HIP),
    (Joint.LEFT_HIP, Joint.LEFT_KNEE),
    (Joint.LEFT_KNEE, Joint.LEFT_ANKLE),
]


# CMU BVH joint name -> 15-joint skeleton index.
CMU_JOINT_MAP = {
    Joint.HEAD: 'Head',
    Joint.NECK: 'Neck1',
    Joint.PELVIS: 'Hips',
    Joint.RIGHT_SHOULDER: 'RightArm',
    Joint.RIGHT_ELBOW: 'RightForeArm',
    Joint.RIGHT_WRIST: 'RightHand',
    Joint.LEFT_SHOULDER: 'LeftArm',
    Joint.LEFT_ELBOW: 'LeftForeArm',
    Joint.LEFT_WRIST: 'LeftHand',
    Joint.RIGHT_HIP: 'RightUpLeg',
    Joint.RIGHT_KNEE: 'RightLeg',
    Joint.RIGHT_ANKLE: 'RightFoot',
    Joint.LEFT_HIP: 'LeftUpLeg',
    Joint.LEFT_KNEE: 'LeftLeg',
    Joint.LEFT_ANKLE: 'LeftFoot',
}


LABEL_NAMES = ('walk', 'jump')
NUM_LABELS = len(LABEL_NAMES)
LABEL_TO_INDEX = {name: index for index, name in enumerate(LABEL_NAMES)}

# Output animation tensor shape: [SEQUENCE_LENGTH, NUM_JOINTS, 3].
SEQUENCE_LENGTH = 48
NUM_JOINTS = 15
