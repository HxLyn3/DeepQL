import numpy as np


class AtariReplayBuffer:
    """ replay buffer for Atari env """
    def __init__(self, buffer_size, gamma=0.99, multi_step_n=1):
        self.memory = {
            "s":    np.zeros((buffer_size, 4, 84, 84), dtype=np.uint8),
            "a":    np.zeros((buffer_size, 1), dtype=np.int64),
            "r":    np.zeros((buffer_size, 1), dtype=np.float32),
            "done": np.zeros((buffer_size, 1), dtype=np.float32),
        }

        # for multi-step TD
        self.gamma = gamma
        self.multi_step_n = multi_step_n
        self.memory["multi_step_r"] = np.zeros((buffer_size, 1), dtype=np.float32)
        self.memory["next_s_idx"]   = np.zeros((buffer_size, 1), dtype=np.int64)

        self.capacity = buffer_size
        self.size = 0
        self.cnt = 0

    def store(self, s: np.ndarray, a: np.int64, r: np.float32, s_: np.ndarray, done: np.float32):
        """ store transition (s, a, r, s_, done) """
        self.memory["s"][self.cnt]  = s
        self.memory["a"][self.cnt]  = a
        self.memory["r"][self.cnt]  = r
        self.memory["done"][self.cnt] = done
        self.cnt = (self.cnt+1)%self.capacity
        self.size = min(self.size+1, self.capacity)

        # update multi-step reward
        self.memory["multi_step_r"][self.cnt-1] = 0
        for i in range(min(self.multi_step_n, self.size)):
            self.memory["multi_step_r"][self.cnt-i-1] += (self.gamma**i)*r
            self.memory["next_s_idx"][self.cnt-i-1] = self.cnt

        self.memory["s"][self.cnt] = s_

    def sample(self, batch_size):
        """ sample a batch of transitions """
        indices = np.random.randint(0, self.size, batch_size)
        return indices, {
            "s":    self.memory["s"][indices],
            "a":    self.memory["a"][indices],
            "r":    self.memory["multi_step_r"][indices],
            "s_":   self.memory["s"][self.memory["next_s_idx"][indices, 0]],
            "done": self.memory["done"][indices]
        }


class AtariPriorReplayBuffer(AtariReplayBuffer):
    """ replay buffer with priority for Atari env """
    def __init__(self, buffer_size, gamma=0.99, multi_step_n=1):
        super(AtariPriorReplayBuffer, self).__init__(buffer_size, gamma, multi_step_n)
        self.memory["prior"] = np.ones((buffer_size, 1), dtype=np.float32)

    def store(self, s: np.ndarray, a: np.int64, r: np.float32, s_: np.ndarray, done: np.float32):
        """ store transition (s, a, r, s_, done) """
        super(AtariPriorReplayBuffer, self).store(s, a, r, s_, done)
        self.memory["prior"][self.cnt-1] = np.max(self.memory["prior"])

    def sample(self, batch_size, prior_alpha, IS_beta):
        """ sample a batch of transitions with priority """
        priors = np.power(self.memory["prior"][:self.size, 0], prior_alpha)
        probs = priors/np.sum(priors)
        indices = np.random.choice(np.arange(self.size), size=batch_size, p=probs)
        weights = np.power(self.capacity*probs[indices], -IS_beta)
        weights /= np.max(weights)
        return indices, {
            "s":    self.memory["s"][indices],
            "a":    self.memory["a"][indices],
            "r":    self.memory["multi_step_r"][indices],
            "s_":   self.memory["s"][self.memory["next_s_idx"][indices, 0]],
            "done": self.memory["done"][indices],
            "wt":   weights
        }

    def update_prior(self, indices, priors):
        """ update priorities of transitions """
        self.memory["prior"][indices, 0] = priors
