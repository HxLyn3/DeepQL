import copy
import torch
import numpy as np
from torch import Tensor
from torch.optim import Adam
from torch.nn import functional as F

from network import NET

class DQNAgent():
    """ Deep Q-Network """
    def __init__(self, input_shape, hidden_dim, num_actions, backbone="cnn",
        dueling=False, double=False, gamma=0.99, multi_step_n=1, lr=1e-4, device="cuda:0"):
        self.qnet = NET["Q"][backbone](input_shape, hidden_dim, num_actions, dueling).to(device)
        self.target_qnet = copy.deepcopy(self.qnet)
        self.optim = Adam(self.qnet.parameters(), lr=lr)

        self.num_actions = num_actions
        self.gamma = gamma
        self.multi_step_n = multi_step_n
        self.double = double
        self.device = device

    def act(self, obs: Tensor, epsilon=0):
        """ choose action """
        if np.random.rand() >= epsilon:
            with torch.no_grad():
                q_value = self.qnet(obs.unsqueeze(0))
                action = q_value.max(-1)[1].item()
        else:
            action = np.random.randint(self.num_actions)
        return action

    def learn(self, s: Tensor, a: Tensor, r: Tensor, s_: Tensor, done: Tensor, wt=None):
        """ backward from TD error """
        # calculate TD-error
        q_values = torch.gather(self.qnet(s), dim=-1, index=a).flatten()
        with torch.no_grad():
            if self.double:
                a_ = self.qnet(s_).max(dim=-1, keepdim=True)[1]
                q_targets = torch.gather(self.target_qnet(s_), dim=-1, index=a_)
            else:
                q_targets = self.target_qnet(s_).max(dim=-1, keepdim=True)[0]
        targets = r.flatten() + (1-done.flatten())*(self.gamma**self.multi_step_n)*q_targets.flatten()
        td_error = F.smooth_l1_loss(q_values, targets, reduction='none')
        loss = (td_error*wt.flatten()).mean() if wt is not None else td_error.mean()

        # backward
        self.optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.qnet.parameters(), 10)
        self.optim.step()

        return td_error.detach().cpu().numpy()

    def update_target(self):
        self.target_qnet.load_state_dict(self.qnet.state_dict())

    def save_model(self, filepath):
        torch.save(self.qnet.state_dict(), filepath)

    def load_model(self, filepath):
        self.qnet.load_state_dict(torch.load(filepath))

    def train(self):
        self.qnet.train()

    def eval(self):
        self.qnet.eval()
