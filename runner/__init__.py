from .q_trainer import QTrainer
from .q_tester import QTester

RUNNER = {
    "q-train": QTrainer,
    "q-test": QTester,
    "ac-train": None,
    "ac-test": None
}