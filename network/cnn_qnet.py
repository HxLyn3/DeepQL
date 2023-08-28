import torch
from torch import nn

class CNNQ(nn.Module):
    """ cnn-based q-network """
    def __init__(self, input_shape, hidden_dim, num_actions, dueling=False):
        super(CNNQ, self).__init__()

        self.input_shape = input_shape
        self.hidden_dim = hidden_dim
        self.num_actions = num_actions

        # cnn block
        self.to_feature = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(inplace=True)
        )

        # fc layer
        self.feature_size = self.to_feature(torch.zeros(1, *self.input_shape)).view(1, -1).size(1)
        self.fc = nn.Sequential(
            nn.Linear(self.feature_size, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, self.num_actions)
        )

        self.dueling = dueling
        if self.dueling:
            # state value function
            self.to_v = nn.Sequential(
                nn.Linear(self.feature_size, 512),
                nn.ReLU(inplace=True),
                nn.Linear(512, 1)
            )

    def forward(self, x):
        self.feat = self.to_feature(x).view(x.size(0), -1)
        out = self.fc(self.feat)
        if self.dueling:
            # Q = V + A
            v = self.to_v(self.feat)
            out += (v-out.mean(dim=-1, keepdim=True)).expand(-1, self.num_actions)
        return out
