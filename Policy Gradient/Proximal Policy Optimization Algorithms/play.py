import pickle, os.path, math, glob, gymnasium, time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib import animation
from PPO import *


def play(env, agent, episode_n, env_name, state_rms, output_gif=False):

    episode_rewards = []
    gifs = []
    for i in range(episode_n):
        state_, _ = env.reset()
        state = np.clip((state_ - state_rms.mean) / (state_rms.var ** 0.5 + 1e-8), -5, 5)
        done = False
        episode_reward = 0
        episode_gifs = []
        episode_gifs.append(env.render())
        while not done:
            mu, sigma = agent.get_action(torch.from_numpy(state).float().to(device))
            dist = torch.distributions.Normal(mu, sigma[0])
            action = dist.sample()
            state_, reward, done1, done2, _ = env.step(action.cpu().numpy())
            state = np.clip((state_ - state_rms.mean) / (state_rms.var ** 0.5 + 1e-8), -5, 5)
            done = done1 or done2
            episode_reward += reward
            episode_gifs.append(env.render())
        episode_rewards.append(episode_reward)
        # 只保存奖励最高的一轮的gif
        if episode_reward >= max(episode_rewards):
            gifs = episode_gifs
    if output_gif:
        save_gif(gifs, env_name)

    return episode_rewards

def save_gif(gifs, env_name):
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
    anim.save(env_name+"_PPO_play.gif", writer='pillow', fps=30)


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
    num_episodes = 20

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(device)

    # Playing PPO
    env = gymnasium.make(env_name, render_mode='rgb_array')
    n_state = env.observation_space.shape[0]
    n_action = env.action_space.shape[0]
    torch.manual_seed(0)
    np.random.seed(0)
    agent = PPO(n_state, n_action, args, device=device)
    state_rms = RunningMeanStd(n_state)

    # 加载模型
    agent.actor.load_state_dict(torch.load(env_name+'_PPO_actor_net3.pt'))
    agent.critic.load_state_dict(torch.load(env_name+'_PPO_critic_net3.pt'))

    # 开玩
    all_rewards = play(env, agent, num_episodes, env_name, state_rms, output_gif=True)

    # gifs = []
    # episode_gifs = []
    # episode_reward = 0
    # all_rewards = []
    # score_lst = []
    # score = 0.0
    # state_lst = []
    # state_ = (env.reset())[0]
    # state = np.clip((state_ - state_rms.mean) / (state_rms.var ** 0.5 + 1e-8), -5, 5)
    # for n_epi in range(num_episodes):
    #     for t in range(args['traj_length']):
    #         state_lst.append(state_)
    #         mu, sigma = agent.get_action(torch.from_numpy(state).float().to(device))
    #         dist = torch.distributions.Normal(mu, sigma[0])
    #         action = dist.sample()
    #         log_prob = dist.log_prob(action).sum(-1, keepdim=True)
    #         next_state_, reward, done1, done2, info = env.step(action.cpu().numpy())
    #
    #         episode_gifs.append(env.render())
    #
    #         next_state = np.clip((next_state_ - state_rms.mean) / (state_rms.var**0.5 + 1e-8), -5, 5)
    #         done = done1 or done2
    #         transition = make_transition(state, action.cpu().numpy(),
    #                                      np.array([reward * args['reward_scaling']]), next_state,
    #                                      np.array([done]), log_prob.detach().cpu().numpy())
    #         agent.put_data(transition)
    #         score += reward
    #         if done:
    #             state_ = (env.reset())[0]
    #             state = np.clip((state_ - state_rms.mean) / (state_rms.var ** 0.5 + 1e-8), -5, 5)
    #             score_lst.append(score)
    #
    #             if score >= max(score_lst):
    #                 gifs = episode_gifs
    #             episode_gifs = []
    #
    #             score = 0
    #         else:
    #             state = next_state
    #             state_ = next_state_
    #
    #     state_rms.update(np.vstack(state_lst))
    #     if n_epi % 1 == 0 and n_epi != 0:
    #         x = sum(score_lst) / len(score_lst)
    #         all_rewards.append(x)
    #         print("episode:{:d}, average reward:{:.4f}".format(n_epi, x))
    #         score_lst = []
    #
    # save_gif(gifs, env_name)

    fig = plt.figure()
    axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    axes.plot(list(range(num_episodes)), all_rewards)
    axes.set_title("PPO_play_rewards")
    axes.set_xlabel("episode")
    axes.set_ylabel("episode reward")
    fig.savefig(env_name+'_PPO_play_rewards.png')

    print('PPO playing ended successfully!')
    env.close()
