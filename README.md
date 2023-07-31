# RL

Implementation of the [Proximal Policy Optimization algorithm](https://spinningup.openai.com/en/latest/algorithms/ppo.html). 

We choose the 'Cartpole-v1' environement of [Gymnasium](https://gymnasium.farama.org/) and compare our PPO with a Vanilla Policy Gradient borrowed from OpenAI's [spinningup repository](https://github.com/openai/spinningup/blob/master/spinup/examples/pytorch/pg_math/1_simple_pg.py). 

The VPG algorithm is in vpg.py, the PPO algorithm is in ppo.py, some useful functions are in utilities.py and finally the experiments are run in rl_experiment.py.

