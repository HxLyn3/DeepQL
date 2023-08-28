import copy
import torch
from torch import Tensor
from torch.optim import Adam

from network import NET

class RainbowAgent():
    """ Rainbow: Nature DQN + Doulbe + Dueling + PER + C51 + Noisy """
    def __init__(self, input_shape, hidden_dim, num_actions, 
        backbone="cnn", num_atoms=51, noisy=True, noisy_std=0.5, v_min=-10, v_max=10, 
        dueling=True, double=True, gamma=0.99, multi_step_n=3, lr=1e-4, device="cuda:0"):
        self.qnet = NET["DistQ"][backbone](
            input_shape, hidden_dim, num_actions, num_atoms, noisy, noisy_std, dueling).to(device)
        self.target_qnet = copy.deepcopy(self.qnet)
        self.optim = Adam(self.qnet.parameters(), lr=lr)
        self.train()

        self.num_actions = num_actions

        # parameters of distributional RL
        self.num_atoms = num_atoms
        self.v_min = v_min
        self.v_max = v_max
        self.support = torch.linspace(self.v_min, self.v_max, self.num_atoms).to(device)
        self.delta_z = (self.v_max-self.v_min)/(self.num_atoms-1)

        # other parameters of RL
        self.noisy = noisy
        self.gamma = gamma
        self.multi_step_n = multi_step_n
        self.double = double
        self.device = device

    def reset_noise(self):
        self.qnet.reset_noise()

    def act(self, obs: Tensor):
        """ choose action """
        with torch.no_grad():
            q_value = (self.qnet(obs.unsqueeze(0))*self.support).sum(dim=2)
            action = q_value.max(-1)[1].item()
        return action

    def learn(self, s: Tensor, a: Tensor, r: Tensor, s_: Tensor, done: Tensor, wt=None):
        """ backward """
        bs = s.size(0)

        # return distribution (log) of current state (s)
        log_dist_sa = self.qnet(s, log=True)[range(bs), a.flatten()]            # log p(s, a)

        with torch.no_grad():
            # calculate return distribution of n-th next state (s_)
            self.target_qnet.reset_noise()
            target_dist_s = self.target_qnet(s_)                                # p_(s_, ·)
            if self.double:
                dist_s_ = self.qnet(s_)                                         # p(s_, ·)
                a_ = (dist_s_*self.support.expand_as(dist_s_)).sum(2).argmax(1)
            else:
                a_ = (target_dist_s*self.support.expand_as(target_dist_s)).sum(2).argmax(1)
            target_dist_sa = target_dist_s[range(bs), a_]                       # p_(s_, a_)

            # compute Tz (Bellman operator T applied to z)
            Tz = r + (1-done)*(self.gamma**self.multi_step_n)*self.support.unsqueeze(0)
            Tz = Tz.clamp(min=self.v_min, max=self.v_max)
            # compute L2 projection of Tz onto fixed support z
            b = (Tz-self.v_min)/self.delta_z                                    # b = (Tz-Vmin)/Δz
            l, u = b.floor().to(torch.int64), b.ceil().to(torch.int64)
            # fix disappearing probability mass when l = b = u (b is int)
            l[(u>0)*(l==u)] -= 1
            u[(l<(self.num_atoms-1))*(l==u)] += 1

            # distribution of Tz
            m = s.new_zeros(bs, self.num_atoms)
            offset = torch.linspace(0, ((bs-1)*self.num_atoms), bs).unsqueeze(1).expand(bs, self.num_atoms).to(a)
            m.view(-1).index_add_(0, (l+offset).view(-1), (target_dist_sa*(u.float()-b)).view(-1))  # m_l = m_l + p_(s_, a_)(u - b)
            m.view(-1).index_add_(0, (u+offset).view(-1), (target_dist_sa*(b-l.float())).view(-1))  # m_u = m_u + p_(s_, a_)(b - l)

        # cross-entropy loss (minimises KL(m||p(s_t, a_t)))
        cse_loss = -torch.sum(m*log_dist_sa, 1).flatten()
        loss = (cse_loss*wt.flatten()).mean() if wt is not None else cse_loss.mean()
        self.optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.qnet.parameters(), 10)
        self.optim.step()

        return cse_loss.detach().cpu().numpy()

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
