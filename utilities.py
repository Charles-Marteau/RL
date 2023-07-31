# RL

# -------------------------------------------------------------

# Utilities

# -------------------------------------------------------------

# The first function generates a gif out of the saved frames
# of an episode and save it in the current file directory.

# The second function plays an episode with a choice of policy
# that samples uniformly over the actions.



import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from matplotlib import animation
import os

# -------------------------------------------------------------

# Generate a gif with the frames and save it in the current file
# directory.
# The code below was obtained from:
# https://gist.github.com/botforge/64cbb71780e6208172bbf03cd9293553

def save_gif(filename, frames):
    current_directory = os.path.dirname(os.path.abspath(__file__))
    plt.figure(
        figsize=(
        frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), 
        dpi=72
        );
    patch = plt.imshow(frames[0])
    plt.axis('off')
    def animate(i):
        patch.set_data(frames[i])
    anim = animation.FuncAnimation(
        plt.gcf(), animate, frames = len(frames), interval=50
        )
    if os.path.exists(current_directory + '/'+ filename + '.gif'):
        os.remove(current_directory + '/' + filename + '.gif')
    anim.save(current_directory + '/' + filename + '.gif', fps=60)
    plt.close()

# -------------------------------------------------------------

# We run an episode with policy = uniform sampling over actions
def random_episode(env_name):
    env = gym.make(
        env_name, render_mode="rgb_array_list"
        )
    obs, info = env.reset() # random initial observation
    ep_done = False # flag that will tell us when the episode is over
    while not ep_done:
        act = env.action_space.sample() # uniform sampling of an action
        obs, rew, term, trun, info = env.step(act) # env step
        ep_done = term or trun
        
    frames = env.render() # returns list of arrays, one for each frame 
    save_gif('random_episode', frames)
    env.close() # close the environment