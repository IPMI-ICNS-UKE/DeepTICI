from enum import Enum, auto

"""implements output and model modes to prevent using strings"""


class OutputMode(Enum):
    last_frame = auto()
    all_frames = auto()


class ModelMode(Enum):
    train = auto()
    inference = auto()
