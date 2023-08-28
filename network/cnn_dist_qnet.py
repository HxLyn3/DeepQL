from turtle import forward
import torch
from torch import nn
from torch.nn import functional as F
from .noisy_linear import NoisyLinear

class CNNDistQ(nn.Module):
    """ cnn-based distributional q-network """
    def __init__(self, input_shape, hidden_dim, num_actions, num_atoms, noisy, noisy_std=0.5, dueling=False):
        super(CNNDistQ, self).__init__()

        self.input_shape = input_shape
        self.hidden_dim = hidden_dim
        self.num_actions = num_actions
        self.num_atoms = num_atoms
        self.noisy = noisy
        self.noisy_std = noisy_std

        # cnn block
        self.to_feature = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(inplace=True)
        )

        self.feature_size = self.to_feature(torch.zeros(1, *self.input_shape)).view(1, -1).size(1)

        # fc layer
        if self.noisy: 
            make_linear = lambda in_size, out_size: NoisyLinear(in_size, out_size, self.noisy_std)
        else: 
            make_linear = lambda in_size, out_size: nn.Linear(in_size, out_size)
        self.fc1 = make_linear(self.feature_size, 512)
        self.fc2 = make_linear(512, self.num_actions*self.num_atoms)

        self.dueling = dueling
        if self.dueling:
            # state value function
            self.fc_v1 = make_linear(self.feature_size, 512)
            self.fc_v2 = make_linear(512, self.num_atoms)

    def forward(self, x, log=False):
        self.feat = self.to_feature(x).view(x.size(0), -1)
        out = self.fc2(F.relu(self.fc1(self.feat))).view(-1, self.num_actions, self.num_atoms)
        if self.dueling:
            # Q = V + A
            v = self.fc_v2(F.relu(self.fc_v1(self.feat))).view(-1, 1, self.num_atoms)
            out += (v-out.mean(dim=1, keepdim=True)).expand(-1, self.num_actions, self.num_atoms)
        if log:
            # use log softmax for numerical stability
            out = F.log_softmax(out, dim=2)
        else:
            out = F.softmax(out, dim=2)
        return out

    def reset_noise(self):
        """ reset noise of noisy network """
        if self.noisy:
            for name, module in self.named_children():
                if "fc" in name:
                    module.reset_noise()
