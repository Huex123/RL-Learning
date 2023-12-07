import pickle, os.path, math, glob, gym, time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from torch.utils.tensorboard import SummaryWriter
from TRPO import *


if __name__ == '__main__':
    num_episodes = 6000
    hidden_dim = 128
    gamma = 0.9
    lmbda = 0.9
    critic_lr = 1e-2
    kl_constraint = 0.00005
    alpha = 0.5
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(device)
    
    # Training TRPO in Pendulum-v1
    env = gym.make('Pendulum-v1')#, render_mode='human')
    torch.manual_seed(0)
    agent = TRPOContinuous(hidden_dim, env.observation_space, env.action_space, lmbda, 
                           kl_constraint, alpha, critic_lr, gamma, device)

    # tensorboard
    summary_writer = SummaryWriter(log_dir = "train_TRPO_episodes", comment= "Pendulum-v1")

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
            action = agent.take_action(state)
            next_state, reward, done1, done2, _ = env.step(action)
            done = done1 or done2
            transiton_dict['states'].append(state)
            transiton_dict['actions'].append(action)
            transiton_dict['next_states'].append(next_state)
            transiton_dict['rewards'].append(reward)
            transiton_dict['dones'].append(done)
            episode_reward += reward
            state = next_state

        all_rewards.append(episode_reward)
        agent.update(transiton_dict)

        if i % 5 == 0:
            print("episode:{:d}, episode_reward:{:.4f}".format(i, episode_reward))
            summary_writer.add_scalar("episode reward", episode_reward, i)


    # 保存神经网络模型(Actor、Critic)
    torch.save(agent.actor.state_dict(), 'Pendulum_TRPO_actor_net.pt')
    torch.save(agent.critic.state_dict(), 'Pendulum_TRPO_critic_net.pt')
    '''
    再次打开，可以先创建PolicyNet类、ValueNet，然后实例化，再载入参数，eg:
    class PolicyNet:
            ...
    my_net = PolicyNet()
    my_net.load_state_dict(torch.load('CartPole_A2C_actor_net.pt'))
    或者直接在agent中：
    agent = TRPO(state_dim, hidden_dim, action_dim, device=device)
    agent.actor.load_state_dict(torch.load('CartPole_A2C_actor_net.pt'))
    agent.critic.load_state_dict(torch.load('CartPole_A2C_critic_net.pt'))
    '''
    
    # 保存rewards
    df_rewards = pd.DataFrame({'rewards':all_rewards}, index=list(range(len(all_rewards))), dtype=pd.Float32Dtype)
    df_rewards.to_csv('TRPO_train_rewards.csv')
    '''
    再次打开：
    data = pd.read_csv('TRPO_train_rewards.csv')
    画图：
    fig = plt.figure()
    axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    axes.plot(list(range(len(data))), data.iloc[:, 1])
    axes.set_title("TRPO_train_rewards")
    axes.set_xlabel("episode")
    axes.set_ylabel("episode reward")
    fig.savefig('TRPO_train_rewards.png')
    '''
    print('TRPO training ended successfully!')
    summary_writer.close()
    env.close()
