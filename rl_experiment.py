# RL

# -------------------------------------------------------------

# Experiments

# -------------------------------------------------------------

# We run some experiments using our VPG and PPO algorithms.
# We obtain the environment from the gymnasium library:
# https://gymnasium.farama.org/
# All gif and plots are saved in the same directory as this
# python script.

# MAIN CONCLUSION: after running multiple time each policy
# training, the observation is that vpg requires a much larger
# batch_size (10x larger) to get a significant improving of 
# the average return, and even after such significant impro-
# vement, the variance remains quite big (order 100), which 
# can lead to deterioration of the results. The goal of ppo
# is exactly to avoid this kind of issue, as explained in the
# ppo.py file, the training is designed so that each step must
# preserve a small 'distance' between the old and new policies.
# The result is that the training requires a much smaller batch_
# size and when the average return approaches the maximum, it 
# rarely decreases back.

import utilities
import vpg
import ppo


# -------------------------------------------------------------

env_name = 'CartPole-v1'

# We start by playing an example of episode that uses 
# a policy which samples uniformly over actions.

utilities.random_episode(env_name)
# output: random_episode.gif

# -------------------------------------------------------------

# We train a policy parameterized by the 'logits_net',
# a one hidden layer mlp. We start by training it using
# a procedure called Vanilla Policy Gradient. The latter
# defines a loss after the collection of 'batch_size' obs
# whose gradient matches the average return gradient. See 
# the file vpg.py for more details on the algorithm.
# For even more details on the algorithm, see:
# https://spinningup.openai.com/en/latest/algorithms/vpg.html


vpg_example = vpg.vpg(
                    env_name=env_name,
                    hidden_sizes=[64], 
                    lr=1e-2, 
                    epochs_training=150, 
                    batch_size=5000
                    ).full_training()
# outputs:
# trained_episode_vpg.gif: 
# total_return_evolution_vpg.png: 

# -------------------------------------------------------------


# We then train the same policy using a different method 
# called Proximal Policy Optimization. The philosophy of 
# the latter is a little bit different. The idea is to 
# allow for bigger step in the policy parameter update but
# still making sure that the update doesn't ruin the per-
# formance. Two complication are added compared to the pre-
# vious algorithm. The first one is that we need to define
# another mlp called 'value_net' whose sole role is to 
# estimate the value function V(s) by minimizing at each step 
# a loss which measures quadratic difference between the 
# prediction of the net and the return-to-go. This estimation
# of the value function is then used to estimate the advan-
# age function. For this last step we use GAE at lambda=0
# (see: https://arxiv.org/abs/1506.02438). Armed with this
# estimation of the advantage function we then define a loss
# which is itself maximized at each step of the policy update.
# Intuitively, this loss becomes more positive when the new
# policy increases the probabilities of actions that are 
# advantageous (the latter being measured with the advantage
# function). Now the policy update is controlled in two ways.
# First, the loss saturates when the new policy is too far
# from the old one. This is controlled by the clipping
# parameter epsilon. Moreover, after each policy step, the
# KL-divergence of new policy from the old is measured and if
# the latter is larger than some treshold, the update stops.
# For more details on the algorithm, see the ppo.py file.
# For even more details on the algorithm, see:
# https://spinningup.openai.com/en/latest/algorithms/ppo.html



ppo_example = ppo.ppo(
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
                kl_cutoff=0.025).full_training()
# outputs:
# trained_episode_vpg.gif: 
# total_return_evolution_vpg.png: 