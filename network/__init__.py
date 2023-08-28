from .mlp_qnet import MLPQ
from .cnn_qnet import CNNQ
from .rnn_qnet import RNNQ
from .cnn_dist_qnet import CNNDistQ

NET = {
    "Q": {
        "mlp": MLPQ,
        "cnn": CNNQ,
        "rnn": RNNQ
    },

    "DistQ": {
        "cnn": CNNDistQ
    }
}