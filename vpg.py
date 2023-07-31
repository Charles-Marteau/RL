# RL

# -------------------------------------------------------------

# Vanilla Policy Gradient

# -------------------------------------------------------------

# Implementation of a Vanilla Policy Gradient algorithm.
# Most of the code comes from OpenAI's spinningup repository:
# https://github.com/openai/spinningup

# We use it for comparison with our own implementation of PPO.

import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.optim import Adam
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import utilities
import os

# -------------------------------------------------------------

class vpg(torch.nn.Module):

    def __init__(
            self,
            env_name='CartPole-v1',
            hidden_sizes=[64], 
            lr=1e-2, 
            epochs_training=150, 
            batch_size=5000
            ):
        super().__init__()
        # environment properties
        self.env_name = env_name
        self.env = gym.make(env_name)
        self.obs_dim = self.env.observation_space.shape[0]
        self.n_acts = self.env.action_space.n
        # training parameters
        self.hidden_sizes = hidden_sizes
        self.lr = lr
        self.epochs_training = epochs_training
        self.batch_size = batch_size 
        self.logits_net = self.mlp(
            sizes=[self.obs_dim] + hidden_sizes + [self.n_acts]
            )
         
    def mlp(self, 
            sizes, 
            activation=nn.Tanh, 
            output_activation=nn.Identity
            ):
        # Build a feedforward neural network.
        layers = []
        for j in range(len(sizes)-1):
            act = activation if j < len(sizes) - 2 else output_activation
            layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
        return nn.Sequential(*layers)
    
    # function to compute action distribution
    def get_policy(self, obs_tensor):
        logits = self.logits_net(obs_tensor)
        return Categorical(logits=logits)

    # action selection function 
    def get_action(self, obs):
        return self.get_policy(obs).sample().item()
    
    # function whose gradient, for the right data, is policy gradient
    def compute_loss(self, obs, act, weights):
        logp = self.get_policy(obs).log_prob(act)
        return -(logp * weights).mean()

    def full_training(self):
      
        optimizer = Adam(self.logits_net.parameters(), lr=self.lr)

        def train_one_epoch():
            batch_obs = []       # observations
            batch_acts = []      # actions
            batch_weights = []   # R(tau) weighting in policy gradient
            batch_rets = []      # measuring episode returns

            # reset episode-specific variables
            obs, _ = self.env.reset()  # first obs comes from starting distribution
            terminated = False   
            truncated = False    # signal from environment that episode is over
            ep_rews = []         # list for rewards accrued throughout ep

            # collect experience by acting in the environment with current policy
            ended_trajectories = 0
            while True:

                # save obs
                batch_obs.append(obs)

                # act in the environment
                act = self.get_action(torch.from_numpy(obs))
            
                obs, rew, terminated, truncated, _ = self.env.step(act)

                # save action, reward
                batch_acts.append(act)
                ep_rews.append(rew)

                if terminated or truncated:

                    # we count the number of trajectories that ended
                    ended_trajectories += 1

                    # if episode is over, record info about episode
                    ep_ret, ep_len = sum(ep_rews), len(ep_rews)
                    batch_rets.append(ep_ret)

                    # the weight for each logprob(a|s) is R(tau)
                    batch_weights += [ep_ret] * ep_len
                    
                    # reset episode-specific variables
                    obs, _ = self.env.reset()
                    terminated, truncated, ep_rews = False, False, []

                    # end experience loop if we have enough of it
                    if len(batch_obs) > self.batch_size:
                        break
            
            # take a single policy gradient update step
            optimizer.zero_grad()
            batch_loss = self.compute_loss(obs=torch.from_numpy(np.array(batch_obs)),
                                    act=torch.from_numpy(np.array(batch_acts)),
                                    weights=torch.from_numpy(np.array(batch_weights))
                                    )
            batch_loss.backward()
            optimizer.step()
            self.env.close()
            return batch_loss, batch_rets, ended_trajectories

        # training loop
        mean_batch_rets_list = []
        for i in range(self.epochs_training):
            _, batch_rets, ended_trajectories = train_one_epoch()
            mean_batch_rets = np.mean(batch_rets)
            mean_batch_rets_list.append(mean_batch_rets) 
            print('epoch: %3d \t return: %.3f \t full trajectories: %.3d'%
                    (i, mean_batch_rets, ended_trajectories))
            
        # plot the result
        fig = plt.figure(figsize=(7, 5), dpi=70)
        axis = fig.add_axes((0.1, 0.1, 0.9, 0.9))
        axis.plot(list(range(len(mean_batch_rets_list))), mean_batch_rets_list)
        current_directory = os.path.dirname(os.path.abspath(__file__))
        fig.savefig(current_directory + '/' + 'total_return_evolution_vpg.png')
        plt.close()
            
        # create a gif with the trained model
        env = gym.make(self.env_name, render_mode="rgb_array_list")
        obs, _ = env.reset()
        act = self.get_action(torch.from_numpy(obs))
        ep_done = False
        while not ep_done:
            act = self.get_action(torch.from_numpy(obs))  
            obs, _, terminated, truncated, _ = env.step(act)
            ep_done = terminated or truncated
            
        frames = env.render()
        utilities.save_gif('trained_episode_vpg', frames)
        env.close()