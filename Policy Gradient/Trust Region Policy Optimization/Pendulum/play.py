import pickle, os.path, math, glob, gym, time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib import animation
from TRPO import *

def play(env, agent, episode_n=1, output_gif=False):
    episode_rewards = []
    gifs = []
    for i in range(episode_n):
        state, _ = env.reset()
        done = False
        episode_reward = 0
        episode_gifs = []
        episode_gifs.append(env.render())
        while not done:
            action = agent.take_action(state)
            state, reward, done1, done2 ,_ = env.step(action)
            done = done1 or done2
            episode_reward += reward
            episode_gifs.append(env.render())
        episode_rewards.append(episode_reward)
        # 只保存奖励最高的一轮的gif
        if episode_reward >= max(episode_rewards):
            gifs = episode_gifs
    if output_gif:
        save_gif(gifs)

    return episode_rewards

def save_gif(gifs):
    '''
    保存动画为.gif
    input:
    gifs: list of env.render()
    '''
    patch = plt.imshow(gifs[0])
    plt.axis("off")
    def animate(i):
        patch.set_data(gifs[i])
    anim = animation.FuncAnimation(plt.gcf(), animate, frames=len(gifs), interval=5)
    anim.save("A2C_play_Pendulum.gif", writer='pillow', fps=30)

    
if __name__ == '__main__':

    num_episodes = 20
    hidden_dim = 128
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(device)
    
    # Playing TRPO in Pendulum-v1
    env = gym.make('Pendulum-v1', render_mode='rgb_array')
    torch.manual_seed(0)
    agent = TRPOContinuous(hidden_dim, env.observation_space, env.action_space, device=device)
    # 加载模型
    agent.actor.load_state_dict(torch.load('Pendulum_TRPO_actor_net1.pt'))
    agent.critic.load_state_dict(torch.load('Pendulum_TRPO_critic_net1.pt'))

    # 开玩
    all_rewards = play(env=env, agent=agent, episode_n=num_episodes, output_gif=True)

    fig = plt.figure()
    axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    axes.plot(list(range(num_episodes)), all_rewards)
    axes.set_title("A2C_play_rewards")
    axes.set_xlabel("episode")
    axes.set_ylabel("episode reward")
    fig.savefig('A2C_play_rewards.png')

    print('A2C playing ended successfully!')
    env.close()
