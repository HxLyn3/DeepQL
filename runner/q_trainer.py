import os
import cv2
import json
import torch
import numpy as np
from tqdm import tqdm

from env import ENV
from agent import AGENT
from utils import BUFFER
from config import CONFIG

from .base import BaseRunner

class QTrainer(BaseRunner):
    """ train value-based policy """
    def __init__(self, args):
        super(QTrainer, self).__init__(args)

        # init env
        env_config = CONFIG["env"][args.env]
        self.env = ENV[args.env](args.env_name, **env_config)
        self.env.seed(args.seed)
        self.env.action_space.seed(args.seed)

        env_config["episode_life"] = False
        env_config["clip_rewards"] = False
        self.eval_env = ENV[args.env](args.env_name, **env_config)
        self.eval_env.seed(args.seed)
        self.eval_env.action_space.seed(args.seed)

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
        self.agent.train()
        
        # whether noisy
        self.noisy = getattr(self.agent, "noisy", False)

        # exploration
        self.epsilon_update = lambda it: \
            args.epsilon + (args.epsilon_min-args.epsilon)*min(it/args.epsilon_tau, 1)

        # init replay buffer
        make_buffer = lambda **params: BUFFER["{}-{}".format(
            args.env, CONFIG["algo"][args.algo]["buffer"])](**params)
        buffer_params = {
            "buffer_size": args.buffer_size,
            "gamma": args.gamma,
            "multi_step_n": CONFIG["algo"][args.algo]["config"]["multi_step_n"]
        }
        self.memory = make_buffer(**buffer_params)

        self.per = "per" in CONFIG["algo"][args.algo]["buffer"]
        if self.per:
            self.prior_alpha = args.prior_alpha
            self.IS_beta_update = lambda it: \
                args.IS_beta + (1-args.IS_beta)*it/args.n_steps

        # other parameters
        self.n_steps = args.n_steps
        self.batch_size = args.batch_size
        self.start_red_learning = args.start_red_learning
        self.start_agent_learning = args.start_agent_learning
        self.learning_interval = args.learning_interval
        self.target_update_interval = args.target_update_interval
        self.eval_interval = args.eval_interval
        self.eval_n_episodes = args.eval_n_episodes
        self.render = args.render
        if self.render:
            cv2.namedWindow("render")
        self.device = args.device
        self.seed = args.seed

        self.model_dir = "./result/{}/{}/{}/model".format(args.env, args.env_name, args.algo)
        self.record_dir = "./result/{}/{}/{}/record".format(args.env, args.env_name, args.algo)
        if not os.path.exists(self.model_dir): 
            os.makedirs(self.model_dir)
        if not os.path.exists(self.record_dir):
            os.makedirs(self.record_dir)

    def __eval_policy(self):
        # evaluate policy
        episode_rewards = []
        for _ in range(self.eval_n_episodes):
            done = False
            episode_rewards.append(0)
            obs = self.eval_env.reset()
            while not done:
                obs_torch = torch.from_numpy(obs.astype(np.float32)/255.).to(self.device)
                action = self.agent.act(obs_torch)
                next_obs, reward, done, _ = self.eval_env.step(action)
                episode_rewards[-1] += reward
                obs = next_obs
        return episode_rewards

    def run(self):
        """ train {args.algo} on {args.env} for {args.n_steps} steps"""

        # init
        records = {"step": [], "loss": [], "reward_mean": [],\
            "reward_std": [], "reward_min": [], "reward_max": []}
        obs = self.env.reset()

        red_loss, td_loss, eval_reward = None, None, None
        pbar = tqdm(range(self.n_steps), desc= "Training {} on {}.{} (seed: {})".format(
            self.args.algo.upper(), self.args.env.title(), self.args.env_name, self.seed))

        for it in pbar:
            if self.noisy and it%self.learning_interval == 0:
                self.agent.reset_noise()

            # step
            obs_torch = torch.from_numpy(obs.astype(np.float32)/255.).to(self.device)
            if self.noisy:
                action = self.agent.act(obs_torch)
            else:
                action = self.agent.act(obs_torch, self.epsilon_update(it))
            next_obs, reward, done, _ = self.env.step(action)
            self.memory.store(obs, action, reward, next_obs, done)
            obs = next_obs

            if done:
                obs = self.env.reset()

            # render
            if self.render:
                img = obs[0:3].transpose(1,2,0)
                cv2.imshow("render", img)
                cv2.waitKey(10)

            # train agent
            if it > self.start_agent_learning:
                if it%self.learning_interval == 0:
                    if self.per:
                        # prioritized experience replay
                        IS_beta = self.IS_beta_update(it)
                        indices, transitions = self.memory.sample(
                            self.batch_size, self.prior_alpha, IS_beta)
                    else:
                        # vanilla experience replay
                        indices, transitions = self.memory.sample(self.batch_size)
                    transitions["s"]  = transitions["s"].astype(np.float32)/255.
                    transitions["s_"] = transitions["s_"].astype(np.float32)/255.
                    for key in transitions.keys():
                        transitions[key] = torch.from_numpy(transitions[key]).to(self.device)

                    td_losses = self.agent.learn(**transitions)
                    td_loss = np.mean(td_losses).item()

                    # update priority if {per}
                    if self.per:
                        self.memory.update_prior(indices, td_losses)

                    # target update
                    if it%self.target_update_interval == 0:
                        self.agent.update_target()

                # evaluate policy
                if it%self.eval_interval == 0:
                    episode_rewards = self.__eval_policy()
                    records["step"].append(it)
                    records["loss"].append(td_loss)
                    records["reward_mean"].append(np.mean(episode_rewards))
                    records["reward_std"].append(np.std(episode_rewards))
                    records["reward_min"].append(np.min(episode_rewards))
                    records["reward_max"].append(np.max(episode_rewards))
                    eval_reward = records["reward_mean"][-1]

            pbar.set_postfix(red_loss=red_loss, td_loss=td_loss, eval_reward=eval_reward)

        # save
        self.agent.save_model(os.path.join(self.model_dir, "model_seed-{}.pth".format(self.seed)))
        with open(os.path.join(self.record_dir, "record_seed-{}.txt".format(self.seed)), "w") as f:
            json.dump(records, f)
