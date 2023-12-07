import pickle, os.path, math, glob, gym, time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from torch.utils.tensorboard import SummaryWriter
from reinforce import *


if __name__ == '__main__':

    learning_rate = 1e-3
    gamma = 0.98
    num_episodes = 3000
    hidden_dim = 128
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(device)
    
    # Training REINFORCE in CartPole-v1
    env = gym.make('CartPole-v1')#, render_mode='human')
    torch.manual_seed(0)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = REINFORCE(state_dim, hidden_dim, action_dim, learning_rate, gamma, device)

    # tensorboard
    summary_writer = SummaryWriter(log_dir = "train_REINFORCE_episodes", comment= "CartPole-v1")

    episode_reward = 0
    all_rewards = []

    for i in range(num_episodes):

        state, _ = env.reset()
        done = False
        episode_reward = 0
        transiton_dict = {
            'states' : [],
            'actions' : [],
            'next_states' : [],
            'rewards' : [],
            'dones' : []
        }

        while not done:
            action = agent.take_action(state, True)
            next_state, reward, done1, done2, _ = env.step(action)
            done = done1 or done2
            transiton_dict['states'].append(state)
            transiton_dict['actions'].append(action)
            transiton_dict['next_states'].append(next_state)
            transiton_dict['rewards'].append(reward)
            transiton_dict['dones'].append(done)
            state = next_state
            episode_reward += reward
        all_rewards.append(episode_reward)
        agent.update(transiton_dict)

        if i % 5 == 0:
            print("episode:{:d}, episode_reward:{:.4f}".format(i, episode_reward))
            summary_writer.add_scalar("episode reward", episode_reward, i)


    # 保存神经网络模型
    torch.save(agent.policy_net.state_dict(), 'CartPole_REINFORCE_policy_net.pt')
    '''
    再次打开，可以先创建一个Qnet类，然后实例化，再载入参数，eg:
    class PolicyNet:
            ...
    my_net = PolicyNet()
    my_net.load_state_dict(torch.load('Pong_DDQN_Qnet.pt'))
    '''
    
    # 保存rewards
    df_rewards = pd.DataFrame({'rewards':all_rewards}, index=list(range(len(all_rewards))), dtype=pd.Float32Dtype)
    df_rewards.to_csv('REINFORCE_train_rewards.csv')
    '''
    再次打开：
    data = pd.read_csv('REINFORCE_train_rewards.csv')
    画图：
    fig = plt.figure()
    axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    axes.plot(list(range(len(data))), data.iloc[:, 1])
    axes.set_title("REINFORCE_train_rewards")
    axes.set_xlabel("episode")
    axes.set_ylabel("episode reward")
    fig.savefig('REINFORCE_train_rewards.png')
    '''
    print('REINFORCE training ended successfully!')
    summary_writer.close()
    env.close()
