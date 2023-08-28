import torch
from torch import nn

class RNNQ(nn.Module):
    """ rnn-based q-network """
    def __init__(self, input_shape, hidden_dim, num_actions, dueling=False):
        super(RNNQ, self).__init__()

        self.input_shape = input_shape
        self.hidden_dim = hidden_dim
        self.num_actions = num_actions

    def forward(self, x):
        pass