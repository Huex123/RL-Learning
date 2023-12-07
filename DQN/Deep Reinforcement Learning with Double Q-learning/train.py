import pickle, os.path, math, glob, gym,time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from gym.wrappers.atari_preprocessing import AtariPreprocessing # 预处理封装器
from gym.wrappers.frame_stack import LazyFrames # 去重函数，相同画面的图像就不重复存储了
from gym.wrappers.frame_stack import FrameStack # 每次都将最后n帧当成状态
from torch.utils.tensorboard import SummaryWriter
from Double_DQN import *


if __name__ == '__main__':

    USE_CUDA = torch.cuda.is_available()
    dtype = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor
    Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if USE_CUDA \
    else autograd.Variable(*args, **kwargs)
    print(USE_CUDA)

    # Training DQN in PongNoFrameskip-v4
    env = gym.make('PongNoFrameskip-v4')#, render_mode='human')
    env = AtariPreprocessing(env, scale_obs=False, terminal_on_life_loss=True) # 在丢失生命时会结束游戏
    env = FrameStack(env, num_stack=4) # 4帧画面会合并为1个输入，加快学习

    gamma = 0.99
    epsilon_max = 1
    epsilon_min = 0.01
    eps_decay = 30000
    frames = 1500000
    learning_rate = 2e-4
    max_buff = 100000
    update_tar_interval = 1000
    batch_size = 32
    print_interval = 1000
    log_interval = 1000
    learning_start = 10000
    win_reward = 18     # Pong-v4
    win_break = True

    action_space = env.action_space
    action_dim = env.action_space.n
    state_dim = env.observation_space.shape[1]
    state_channel = env.observation_space.shape[0]

    agent = DDQNagent(in_channels = state_channel, action_space= action_space, \
                     USE_CUDA = USE_CUDA, lr = learning_rate, memory_size = max_buff)

    frame, _ = env.reset()

    episode_reward = 0
    all_rewards = []
    losses = []
    episode_num = 0
    is_win = False
    # tensorboard
    summary_writer = SummaryWriter(log_dir = "train_DDQN_stackframe", comment= "good_makeatari")

    # e-greedy decay
    epsilon_by_frame = lambda frame_idx: epsilon_min + (epsilon_max - epsilon_min) * math.exp(
                -1. * frame_idx / eps_decay)
    # plt.plot([epsilon_by_frame(i) for i in range(10000)])

    for i in range(frames):
        epsilon = epsilon_by_frame(i)
        state_tensor = agent.observe(frame)
        action = agent.act(state_tensor, epsilon)

        next_frame, reward, done1, done2 ,_ = env.step(action)
        done = done1 or done2

        episode_reward += reward
        agent.memory_buffer.push(frame, action, reward, next_frame, done)
        frame = next_frame

        loss = 0
        if agent.memory_buffer.size() >= learning_start:
            loss = agent.learn(batch_size)
            losses.append(loss)

        if i % print_interval == 0:
            print("frames: %5d, reward: %5f, loss: %4f, epsilon: %5f, episode: %4d" % \
                  (i, np.mean(all_rewards[-10:]), loss, epsilon, episode_num))
            summary_writer.add_scalar("Temporal Difference Loss(TD loss)", loss, i)
            summary_writer.add_scalar("Mean Reward", np.mean(all_rewards[-10:]), i)
            summary_writer.add_scalar("Epsilon", epsilon, i)

        if i % update_tar_interval == 0:
            agent.target_Qnet.load_state_dict(agent.Qnet.state_dict())

        if done:
            frame, _ = env.reset()
            all_rewards.append(episode_reward)
            episode_reward = 0
            episode_num += 1
            avg_reward = float(np.mean(all_rewards[-100:]))

    # 保存神经网络模型
    torch.save(agent.Qnet.state_dict(), 'Pong_DDQN_Qnet.pt')
    '''
    再次打开，可以先创建一个Qnet类，然后实例化，再载入参数，eg:
    class Qnet:
            ...
    myQnet = Qnet()
    myQnet.load_state_dict(torch.load('Pong_DDQN_Qnet.pt'))
    '''
    
    # 保存rewards
    df_rewards = pd.DataFrame({'rewards':all_rewards}, index=list(range(len(all_rewards))), dtype=pd.Float32Dtype)
    df_rewards.to_csv('DDQN_train_rewards.csv')
    '''
    再次打开：
    data = pd.read_csv('DDQN_train_rewards.csv')
    '''
    print('Double DQN training ended successfully!')
    summary_writer.close()
    env.close()