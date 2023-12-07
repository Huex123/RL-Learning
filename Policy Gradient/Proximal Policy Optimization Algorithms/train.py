import pickle, os.path, math, glob, time, gymnasium

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from torch.utils.tensorboard import SummaryWriter
from PPO import *


if __name__ == '__main__':

    env_name = "Pendulum-v1"  # "CartPole-v1" "Ant-v4" "Hopper-v4"
    args = {
        'lr_actor': 3e-4,
        'lr_critic': 3e-4,
        'l2_rate': 0.001,
        'n_hidden': 256,
        'gamma': 0.99,
        'lmbda': 0.95,  # GAE参数
        'eps': 0.2,  # PPO-Clip参数
        'beta': 3,  # PPO-Penalty参数
        'kl_target': 0.01,  # PPO-Penalty参数
        'entropy_coef': 1e-2,
        'critic_coef': 0.5,
        'max_grad_norm': 0.5,
        'reward_scaling': 0.1,
        'traj_length': 2048,
        'train_epoch': 10,
        'batch_size': 64,
    }
    num_episodes = 700

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(device)

    # tensorboard
    summary_writer = SummaryWriter(log_dir="train_PPO_episodes", comment="mujoco")

    # Training PPO
    env = gymnasium.make(env_name)  # , render_mode='human')
    n_state = env.observation_space.shape[0]
    n_action = env.action_space.shape[0]
    torch.manual_seed(0)
    np.random.seed(0)

    agent = PPO(n_state, n_action, args, summary_writer, device)
    # if torch.cuda.is_available():
    #     agent = agent.cuda()
    state_rms = RunningMeanStd(n_state)

    episode_reward = 0
    all_rewards = []
    score_lst = []
    score = 0.0
    state_lst = []

    state_ = (env.reset())[0]
    state = np.clip((state_ - state_rms.mean) / (state_rms.var ** 0.5 + 1e-8), -5, 5)
    for n_epi in range(num_episodes):
        for t in range(args['traj_length']):
            state_lst.append(state_)
            mu, sigma = agent.get_action(torch.from_numpy(state).float().to(device))
            dist = torch.distributions.Normal(mu, sigma[0])
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(-1, keepdim=True)
            next_state_, reward, done1, done2, info = env.step(action.cpu().numpy())
            next_state = np.clip((next_state_ - state_rms.mean) / (state_rms.var**0.5 + 1e-8), -5, 5)
            done = done1 or done2
            transition = make_transition(state, action.cpu().numpy(),
                                         np.array([reward * args['reward_scaling']]), next_state,
                                         np.array([done]), log_prob.detach().cpu().numpy())
            agent.put_data(transition)
            score += reward
            if done:
                state_ = (env.reset())[0]
                state = np.clip((state_ - state_rms.mean) / (state_rms.var ** 0.5 + 1e-8), -5, 5)
                score_lst.append(score)
                # print("episode:{:d}, episode_reward:{:.4f}".format(n_epi, score))
                score = 0
            else:
                state = next_state
                state_ = next_state_

        agent.train_net(n_epi)
        state_rms.update(np.vstack(state_lst))
        if n_epi % 1 == 0 and n_epi != 0:
            x = sum(score_lst) / len(score_lst)
            all_rewards.append(x)
            summary_writer.add_scalar("episode reward", x, n_epi)
            print("episode:{:d}, average reward:{:.4f}".format(n_epi, x))
            score_lst = []

    # 保存神经网络模型(Actor、Critic)
    torch.save(agent.actor.state_dict(), env_name + '_PPO_actor_net.pt')
    torch.save(agent.critic.state_dict(), env_name + '_PPO_critic_net.pt')
    '''
    再次打开，可以先创建PolicyNet类、ValueNet，然后实例化，再载入参数
    '''

    # 保存rewards
    df_rewards = pd.DataFrame({'rewards': all_rewards}, index=list(range(len(all_rewards))), dtype=pd.Float32Dtype)
    df_rewards.to_csv(env_name + '_PPO_train_rewards.csv')
    '''
    再次打开：
    data = pd.read_csv('PPO_train_rewards.csv')
    画图：
    fig = plt.figure()
    axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    axes.plot(list(range(len(data))), data.iloc[:, 1])
    axes.set_title("PPO_train_rewards")
    axes.set_xlabel("episode")
    axes.set_ylabel("episode reward")
    fig.savefig('PPO_train_rewards.png')
    '''

    print('PPO training ended successfully!')
    summary_writer.close()
    env.close()
