# RL

# -------------------------------------------------------------

# Proximal Policy Optimization

# -------------------------------------------------------------

# Our own implementation of Proximal Policy Optimization.


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

class ppo(torch.nn.Module):

    def __init__(
                self,
                env_name='CartPole-v1', 
                hidden_sizes=[64], 
                batch_size=500,
                epochs_training=150,
                epochs_policy=10, 
                epochs_value=10,
                lr_policy=1e-3, 
                lr_value=1e-3, 
                damping=0.99,
                epsilon=0.1,
                kl_cutoff=0.025
                ):
        super().__init__()
        # network hidden dimensions
        self.hidden_sizes = hidden_sizes
        # training parameters
        self.batch_size = batch_size
        self.epochs_training = epochs_training
        self.epochs_policy = epochs_policy
        self.epochs_value = epochs_value
        self.lr_policy = lr_policy
        self.lr_value = lr_value
        self.damping = damping
        self.epsilon = epsilon
        self.kl_cutoff = kl_cutoff
        # make environment, check spaces, get obs / act dims
        self.env_name = env_name
        self.env = gym.make(env_name)
        self.obs_dim = self.env.observation_space.shape[0]
        self.n_acts = self.env.action_space.n
        # policy network
        self.logits_net = self.mlp(sizes=[self.obs_dim] + hidden_sizes + [self.n_acts])
        # value network
        self.value_net = self.mlp(sizes=[self.obs_dim] + hidden_sizes + [1], output_activation=nn.ReLU)

    # build a feedforward neural network.
    def mlp(self, sizes, activation=nn.Tanh, output_activation=nn.Identity):
        layers = []
        for j in range(len(sizes)-1):
            act = activation if j < len(sizes) - 2 else output_activation
            layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
        return nn.Sequential(*layers)

    # damped cumulative sum
    def cum_sum(self, x, damping):
            """
            out_t = sum_k (damping^k * x_{t + k})
            """
            length = len(x)
            out = []
            x.reverse()
            future = x[0]
            out.append(future)
            for k in range(1, length):
                    future = x[k] + damping * future
                    out.append(future)
            out.reverse()
            x.reverse()
            return out
    
    # function to compute action distribution
    def get_policy(self, obs_tensor):
        logits = self.logits_net(obs_tensor)
        return Categorical(logits=logits)
    
    # function to compute estimated value
    def get_value(self, obs_tensor):
         logit_value = self.value_net(obs_tensor)
         return logit_value.item()

    # action selection function (outputs int actions, sampled from policy)
    def get_action(self, obs):
        return self.get_policy(obs).sample().item()
    
    # function to compute the probabilities prob(at|st)
    def get_probs(self, action, obs_tensor):
        return self.get_policy(obs_tensor).probs[action]

    # collecting experience
    def play(self):
        # type of the content of each list
        # batch_obs: numpy.array
        # batch_acts: int
        # batch_return_to_go: float
        # batch_return_to_go_damped: float
        # batch_est_vals: float
        # batch_rets: float
        # batch_lens: int
        # batch_advantages: float
        # batch_probs: torch.Tensor
        # make some empty lists for logging.
        batch_obs = []          # observations: st
        batch_acts = []         # actions: at
        batch_return_to_go = []      # return to go: Rt(tau) 
        batch_return_to_go_damped = [] # damped return to go: Rt_damped(tau)
        batch_est_vals = []     # value function: V(st)
        batch_rets = []         # episode returns: sum_{t} rt
        batch_lens = []         # episode lengths
        batch_probs = []        # probabilities: prob(at|st)

        # reset episode-specific variables
        obs, _ = self.env.reset()       # first obs comes from starting distribution
        terminated = False   
        truncated = False         # signal from environment that episode is over
        ep_rews = []            # list for rewards accrued throughout ep

       
        # collect experience by acting in the environment with current policy
        ended_trajectories = 0 # we count the number of completed trajectories
        while True:

            # save obs
            batch_obs.append(obs)

            # value estimation
            est_val = self.get_value(torch.from_numpy(obs))
            batch_est_vals.append(est_val)

            # act in the environment
            act = self.get_action(torch.from_numpy(obs))

            # save the probabilities prob(at|st)
            prob = self.get_probs(act, torch.from_numpy(obs))
            batch_probs.append(prob.data.unsqueeze(dim=0))
            
            # env step
            obs, rew, terminated, truncated, _ = self.env.step(act)

            # save action, reward
            batch_acts.append(act)
            ep_rews.append(rew)

            if terminated or truncated:

                # add 1 to the number of completed trajectories 
                ended_trajectories += 1

                # if episode is over, record info about episode
                ep_ret, ep_len = sum(ep_rews), len(ep_rews)
                batch_rets.append(ep_ret)
                batch_lens.append(ep_len)

                # return to go: Rt(tau) = sum_{t'>=t} r_t'
                return_to_go = self.cum_sum(ep_rews, 1)
                batch_return_to_go += return_to_go

                # damped return to go: Rt_damping(tau) = sum_{t'>=t} damping^{t'-t} * r_t'
                return_to_go_damped = self.cum_sum(ep_rews, self.damping)
                batch_return_to_go_damped += return_to_go_damped

                # estimated advantage (GAE with lambda = 1)
                batch_advantages = (np.array(batch_return_to_go_damped) \
                                             - np.array(batch_est_vals)).tolist()

                # reset episode-specific variables
                obs, _ = self.env.reset()
                terminated, truncated, ep_rews = False, False, []

                # end experience loop if we have enough of it
                if len(batch_obs) > self.batch_size:
                    break
        
        return  batch_obs, \
                batch_acts, \
                batch_return_to_go, \
                batch_return_to_go_damped, \
                batch_est_vals, \
                batch_rets, \
                batch_lens, \
                batch_advantages, \
                batch_probs, \
                ended_trajectories
    
  
    # compute KL divergence of the current distribution prob(a|s)
    def KLdiv(self, prob_old, prob_new):
        return (prob_new * ((prob_new / prob_old).log())).mean()
        

    # clipped term in the policy loss
    def clipped(self, A, epsilon):
        out = torch.zeros_like(A)
        out[A >= 0] = (1 + epsilon) * A[A >= 0]
        out[A < 0] = (1 - epsilon) * A[A < 0] 
        return out
    
    # compute the loss minimized at each step of the policy optimization
    def policy_loss(self, prob_old, prob_new, adv_old, epsilon):
        loss = torch.min(adv_old * prob_new / prob_old, self.clipped(adv_old, epsilon)).mean()
        return loss
    
    # compute the loss minimized for value network optimization
    def value_loss (self, est_val, return_to_go):
        loss = (est_val - return_to_go).matmul(est_val - return_to_go).mean()
        return loss
         
    # policy loss minimization
    def train_policy_net(self, batch_obs, batch_acts, prob_old, adv_old, epsilon, kl_cutoff):
        losses = []
        optimizer = Adam(self.logits_net.parameters(), lr=self.lr_policy, maximize=True) 
        for _ in range(self.epochs_policy):
            optimizer.zero_grad()
            batch_obs_tensor = torch.from_numpy(np.array(batch_obs))
            batch_acts_tensor = torch.tensor(batch_acts)
            policy = self.get_policy(batch_obs_tensor)
            arange = torch.arange(batch_obs_tensor.shape[0])
            prob_new = policy.probs[arange, batch_acts_tensor]
            loss = self.policy_loss(prob_old, prob_new, adv_old, epsilon)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            if self.KLdiv(prob_old, prob_new) > kl_cutoff:
                print('KLdiv to large')
                break

        return np.mean(np.array(losses))

    # value loss minimization
    def train_value_net(self, batch_obs, batch_return_to_go):
        losses = []
        optimizer = Adam(self.value_net.parameters(), lr=self.lr_value) 
        for _ in range(self.epochs_value):
            optimizer.zero_grad()
            est_val = self.value_net(torch.from_numpy(np.array(batch_obs))) # est_val.shape = (batch_size, 1)
            loss = self.value_loss(est_val.squeeze(), torch.tensor(batch_return_to_go))
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        return np.mean(np.array(losses))
            
    # full training
    def full_training(self):
        mean_batch_rets_list = []
        for i in range(self.epochs_training):
            batch_obs, batch_acts, batch_return_to_go, _, _, batch_rets, \
                    _, batch_advantages, batch_probs, ended_trajectories = self.play()
            prob_old = torch.cat(batch_probs, dim=0).detach()
            adv_old = torch.from_numpy(np.array(batch_advantages))
            self.train_policy_net(batch_obs, batch_acts, prob_old, adv_old, self.epsilon, self.kl_cutoff)
            self.train_value_net(batch_obs, batch_return_to_go)
            mean_batch_rets = np.mean(np.array(batch_rets))
            mean_batch_rets_list.append(mean_batch_rets) 
            print('epoch: %3d \t return: %.3f \t full trajectories: %.3d'%
                    (i, mean_batch_rets, ended_trajectories)
                    )
            
         # plot the result
        fig = plt.figure(figsize=(7, 5), dpi=70)
        axis = fig.add_axes((0.1, 0.1, 0.9, 0.9))
        axis.plot(list(range(len(mean_batch_rets_list))), mean_batch_rets_list)
        current_directory = os.path.dirname(os.path.abspath(__file__))
        fig.savefig(current_directory + '/' + 'total_return_evolution_ppo.png')
        plt.close()

        # created a gif with the trained model
        env = gym.make('CartPole-v1', render_mode="rgb_array_list")
        obs, _ = env.reset()
        act = self.get_action(torch.from_numpy(obs))
        ep_done = False
        while not ep_done:
            act = self.get_action(torch.from_numpy(obs))  
            obs, _, terminated, truncated, _ = env.step(act)
            ep_done = terminated or truncated
            
        frames = env.render()
        utilities.save_gif('trained_episode_ppo', frames)
        env.close()