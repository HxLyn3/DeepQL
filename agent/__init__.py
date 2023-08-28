from .dqn import DQNAgent
from .rainbow import RainbowAgent

AGENT = {
    "dqn": DQNAgent,
    "rainbow": RainbowAgent
}