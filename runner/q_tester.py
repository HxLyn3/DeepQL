import os
import cv2
import torch
import numpy as np
from tqdm import tqdm

from env import ENV
from agent import AGENT
from config import CONFIG

from .base import BaseRunner

class QTester(BaseRunner):
    """ train value-based policy """
    def __init__(self, args):
        super(QTester, self).__init__(args)

        # init env
        env_config = CONFIG["env"][args.env]
        env_config["episode_life"] = False
        self.env = ENV[args.env](args.env_name, **env_config)
        self.env.seed(args.seed)
        self.env.action_space.seed(args.seed)

        # init agent
        self.agent = AGENT[CONFIG["algo"][args.algo]["agent"]](
            input_shape=self.env.observation_space.shape,
            hidden_dim=args.hidden_dim,
            num_actions=self.env.action_space.n,
            backbone=args.backbone,
            gamma=args.gamma,
            lr=args.lr,
            device=args.device,
            **CONFIG["algo"][args.algo]["config"]
        )
        self.model_dir = "./result/{}/{}/{}/model".format(args.env, args.env_name, args.algo)
        self.agent.load_model(os.path.join(self.model_dir, "model_seed-{}.pth".format(args.seed)))
        self.agent.eval()

        # other parameters
        self.test_n_episodes = args.test_n_episodes
        self.render = args.render
        if self.render:
            cv2.namedWindow("render")
        self.device = args.device
        self.seed = args.seed

    def run(self):
        """ test {args.algo} on {args.env} for {args.test_n_episodes} episodes"""
        pbar = tqdm(range(self.test_n_episodes), desc= "Testing {} on {}.{} (seed: {})".format(
            self.args.algo.upper(), self.args.env.title(), self.args.env_name, self.seed))

        episode_rewards = []
        for _ in pbar:
            # init
            done = False
            episode_rewards.append(0)
            obs = self.env.reset()
            while not done:
                obs_torch = torch.from_numpy(obs.astype(np.float32)/255.).to(self.device)
                action = self.agent.act(obs_torch)
                next_obs, reward, done, _ = self.env.step(action)
                episode_rewards[-1] += reward
                obs = next_obs

                # render
                if self.render:
                    img = obs[0:3].transpose(1,2,0)
                    cv2.imshow("render", img)
                    cv2.waitKey(10)

            pbar.set_postfix(test_reward=episode_rewards[-1])

        print("[#] test reward | mean: {:.4f}, min: {:.4f}, max: {:.4f}".format(
            np.mean(episode_rewards), np.min(episode_rewards), np.max(episode_rewards)))
