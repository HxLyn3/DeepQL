from .replay_buffer import AtariReplayBuffer
from .replay_buffer import AtariPriorReplayBuffer

BUFFER = {
    "atari-vanilla": AtariReplayBuffer,
    "atari-per": AtariPriorReplayBuffer,
}