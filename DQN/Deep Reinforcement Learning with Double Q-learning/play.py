import pickle, os.path, math, glob, gym,time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import imageio

from gym.wrappers.atari_preprocessing import AtariPreprocessing # 预处理封装器
from gym.wrappers.frame_stack import LazyFrames # 去重函数，相同画面的图像就不重复存储了
from gym.wrappers.frame_stack import FrameStack # 每次都将最后n帧当成状态
from matplotlib import animation
from Double_DQN import *


def play(env, agent, episode_n=1, output_gif=False):
    episode_rewards = []
    gifs = []
    for i in range(episode_n):
        frame, _ = env.reset()
        done = False
        episode_reward = 0
        episode_gifs = []
        episode_gifs.append(env.render())
        while not done:
            state_tensor = agent.observe(frame)
            action = agent.act(state_tensor, epsilon=0) # 确定性策略
            frame, reward, done1, done2 ,_ = env.step(action)
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
    anim.save("DDQN_playPong.gif", writer='pillow', fps=30)


if __name__ == '__main__':
    USE_CUDA = torch.cuda.is_available()
    dtype = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor
    Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if USE_CUDA \
    else autograd.Variable(*args, **kwargs)
    print(USE_CUDA)

    # Playing DQN in PongNoFrameskip-v4
    env = gym.make('PongNoFrameskip-v4', render_mode='rgb_array')
    env = AtariPreprocessing(env, scale_obs=False, terminal_on_life_loss=True) # 在丢失生命时会结束游戏
    env = FrameStack(env, num_stack=4) # 4帧画面会合并为1个输入，加快学习

    action_space = env.action_space
    action_dim = env.action_space.n
    state_dim = env.observation_space.shape[1]
    state_channel = env.observation_space.shape[0]
    episode_n = 10

    agent = DDQNagent(in_channels=state_channel, action_space=action_space, USE_CUDA=USE_CUDA, epsilon=0)
    # 载入保存的神经网络模型
    myQnet = Q_net(in_channels=state_channel, n_actions=action_space.n)
    myQnet.load_state_dict(torch.load('Pong_DDQN_Qnet2.pt'))
    if USE_CUDA:
        myQnet = myQnet.cuda()
    agent.Qnet = myQnet
    agent.target_Qnet = myQnet

    # 开玩
    all_rewards = play(env=env, agent=agent, episode_n=episode_n, output_gif=True)

    fig = plt.figure()
    axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    axes.plot(list(range(episode_n)), all_rewards)
    fig.savefig('DDQN_play_rewards.png')
    
    print('Double DQN playing ended successfully!')
    env.close()

   