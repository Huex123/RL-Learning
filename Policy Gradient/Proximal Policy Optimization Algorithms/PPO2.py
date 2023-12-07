import random, copy
import torch

import numpy as np
import pandas as pd
import torch.nn.functional as F

from utils import *


class PolicyNet(torch.nn.Module):
    def __init__(self, n_state, n_hidden, n_action):
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(n_state, n_hidden)
        self.fc2 = torch.nn.Linear(n_hidden, n_hidden)
        self.sigma = torch.nn.Linear(n_hidden, n_action)
        self.mu = torch.nn.Linear(n_hidden, n_action)
        self.mu.weight.data.mul_(0.1)
        self.mu.bias.data.mul_(0.0)
        # self.set_init([self.fc1, self.fc2, self.mu, self.sigma])
        self.distribution = torch.distributions.Normal

    def set_init(self, layers):
        for layer in layers:
            torch.nn.init.normal_(layer.weight, mean=0, std=0.1)
            torch.nn.init.constant_(layer.bias, 0.0)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        mu = self.mu(x)
        log_sigma = self.sigma(x)
        sigma = torch.exp(log_sigma)
        return mu, sigma


class ValueNet(torch.nn.Module):
    def __init__(self, n_state, n_hidden):
        super(ValueNet, self).__init__()
        self.fc1 = torch.nn.Linear(n_state, n_hidden)
        self.fc2 = torch.nn.Linear(n_hidden, n_hidden)
        self.fc3 = torch.nn.Linear(n_hidden, 1)
        self.fc3.weight.data.mul_(0.1)
        self.fc3.bias.data.mul_(0.0)
        # self.set_init([self.fc1, self.fc2, self.fc2])

    def set_init(self, layers):
        for layer in layers:
            torch.nn.init.normal_(layer.weight, mean=0., std=0.1)
            torch.nn.init.normal_(layer.bias, 0.)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        values = self.fc3(x)
        return values


class PPO:
    def __init__(self, n_state, n_action, args, writer=None, device=torch.device("cpu")):
        self.args = args
        self.actor = PolicyNet(n_state, self.args['n_hidden'], n_action).to(device) # 策略网络
        self.critic = ValueNet(n_state, self.args['n_hidden']).to(device) # 价值网络
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.args['lr_actor'])
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.args['lr_critic'],
                                                 weight_decay=self.args['l2_rate'])
        self.critic_loss_func = torch.nn.MSELoss()
        self.gamma = args['gamma']
        self.lmbda = args['lmbda'] # GAE参数
        self.eps = args['eps'] # PPO-Clip参数
        self.beta = args['beta'] # PPO-Penalty参数
        self.kl_target = args['kl_target'] # PPO-Penalty参数
        self.entropy_coef = args['entropy_coef']
        self.critic_coef = args['critic_coef']
        self.max_grad_norm = args['max_grad_norm']
        self.train_epoch = args['train_epoch'] # 每次调用更新进行多少次更新
        self.batch_size = args['batch_size']
        self.device = device

        self.writer = writer
        self.data = ReplayBuffer(action_prob_exist=True, max_size=self.args['traj_length'], state_dim=n_state,
                                 n_action=n_action)

    def get_action(self, x):
        mu, sigma = self.actor(x)
        return mu, sigma

    def v(self, x):
        return self.critic(x)

    def put_data(self, transition):
        self.data.put_data(transition)

    def kl_divergence(self, old_mu, old_sigma, mu, sigma):
        old_mu = old_mu.detach()
        old_sigma = old_sigma.detach()
        kl = torch.log(old_sigma) - torch.log(sigma) + (old_sigma.pow(2) + (old_mu - mu).pow(2)) / \
             (2.0 * sigma.pow(2)) - 0.5
        return kl.sum(1, keepdims=True)

    def get_GAE(self, states, rewards, next_states, dones):
        values = self.v(states).detach()
        td_target = rewards + self.gamma * self.v(next_states) * (1 - dones)
        delta = td_target - values
        delta = delta.detach().cpu().numpy()
        advantage_lst = []
        advantage = 0.0
        for idx in reversed(range(len(delta))):
            if dones[idx] == 1:
                advantage = 0.0
            advantage = self.gamma * self.lmbda * advantage + delta[idx][0]
            advantage_lst.append([advantage])
        advantage_lst.reverse()
        advantages = torch.tensor(advantage_lst, dtype=torch.float).to(self.device)
        return values, advantages

    def train_net(self, n_epi):
        data = self.data.sample(shuffle=False)
        states, actions, rewards, next_states, dones, old_log_probs = \
            convert_to_tensor(self.device, data['state'], data['action'], data['reward'],
                              data['next_state'], data['done'], data['log_prob'])

        old_values, advantages = self.get_GAE(states, rewards, next_states, dones)
        returns = advantages + old_values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-3)

        for i in range(self.train_epoch):
            for state, action, old_log_prob, advantage, return_, old_value \
                    in make_mini_batch(self.batch_size, states, actions,
                                       old_log_probs, advantages, returns, old_values):
                curr_mu, curr_sigma = self.get_action(state)
                value = self.v(state).float()
                curr_dist = torch.distributions.Normal(curr_mu, curr_sigma)
                entropy = curr_dist.entropy() * self.entropy_coef
                curr_log_prob = curr_dist.log_prob(action).sum(1, keepdim=True)

                # policy clipping
                ratio = torch.exp(curr_log_prob - old_log_prob.detach())
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * advantage
                actor_loss = (-torch.min(surr1, surr2) - entropy).mean()

                # value clipping (PPO2 technic)
                old_value_clipped = old_value + (value - old_value).clamp(-self.eps, self.eps)
                value_loss = (value - return_.detach().float()).pow(2)
                value_loss_clipped = (old_value_clipped - return_.detach().float()).pow(2)
                critic_loss = 0.5 * self.critic_coef * torch.max(value_loss, value_loss_clipped).mean()

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()

                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.critic_optimizer.step()

            if self.writer != None and i == 1:
                self.writer.add_scalar("loss/actor_loss", actor_loss.item(), n_epi)
                self.writer.add_scalar("loss/critic_loss", critic_loss.item(), n_epi)
