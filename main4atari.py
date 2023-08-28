import random
import argparse
import setproctitle

import torch
import numpy as np

from runner import RUNNER
from config import value_based_algos, policy_based_algos

def get_args():
    parser = argparse.ArgumentParser(description="DRL")

    # environment settings
    parser.add_argument("--env", type=str, default="atari")                             # env platform, atari/mujoco
    parser.add_argument("--env-name", type=str, default="PongNoFrameskip-v4")

    # algorithm parameters
    parser.add_argument("--algo", type=str, default="rainbow")                          # algorithm
    parser.add_argument("--backbone", type=str, default="cnn")                          # network type
    parser.add_argument("--hidden-dim", type=int, default=512)                          # dimension of hidden layer
    parser.add_argument("--gamma", type=float, default=0.99)                            # discount factor
    parser.add_argument("--lr", type=float, default=1e-4)                               # learning rate
    parser.add_argument("--epsilon", type=float, default=1.0)                           # ε of ε-greedy policy
    parser.add_argument("--epsilon-min", type=float, default=0.05)                      # min value of ε
    parser.add_argument("--epsilon-tau", type=float, default=1e6)                       # annealing time of linear decay
    parser.add_argument("--batch-size", type=int, default=32)                           # mini-batch size

    # replay buffer parameters
    parser.add_argument("--buffer-size", type=int, default=int(1e6))                    # capacity of replay buffer
    parser.add_argument("--prior-alpha", type=float, default=0.5)                       # priority exponents
    parser.add_argument("--IS-beta", type=float, default=0.4)                           # importance-sampling beta in PER

    # running parameters
    parser.add_argument("--train", action="store_true", default=False)
    parser.add_argument("--test", action="store_true", default=False)
    parser.add_argument("--n-steps", type=int, default=int(1e7))
    parser.add_argument("--start-agent-learning", type=int, default=int(5e4))
    parser.add_argument("--learning-interval", type=int, default=4)
    parser.add_argument("--target-update-interval", type=int, default=int(1e4))
    parser.add_argument("--eval-interval", type=int, default=int(5e4))
    parser.add_argument("--eval-n-episodes", type=int, default=5)
    parser.add_argument("--test-n-episodes", type=int, default=100)
    parser.add_argument("--render", action="store_true", default=False)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()
    return args

def main():
    seed_start = 0
    seed_end = 3
    args = get_args()
    if not args.train and not args.test:
        raise ValueError("Argument 'train' and 'test' can't be both False")

    """ main function """
    for seed in range(seed_start, seed_end):
        args.seed = seed
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

        args.stage = "train" if args.train else "test"
        if args.algo in value_based_algos:
            runner = RUNNER["q-{}".format(args.stage)](args)
        elif args.algo in policy_based_algos:
            runner = RUNNER["ac-{}".format(args.stage)](args)
        runner.run()

if __name__ == "__main__":
    setproctitle.setproctitle("Deep Q Learning...")
    main()
