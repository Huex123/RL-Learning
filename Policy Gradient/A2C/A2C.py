import random
import torch

import numpy as np
import pandas as pd
import torch.optim as optim
import torch.nn.functional as F

class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return F.softmax(self.fc2(x), dim=1)


class ValueNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(ValueNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class A2Cagent:
    def __init__(self, state_dim, hidden_dim, action_dim, actor_lr=1e-3,critic_lr=1e-2, \
                 gamma=0.98, device=torch.device("cpu")):
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim).to(device) # 策略网络
        self.critic = ValueNet(state_dim, hidden_dim).to(device) # 价值网络
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr) # 使用Adam优化器
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.gamma = gamma
        self.device = device

    def take_action(self, state, is_train=True):
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        probs = self.actor(state)

        if is_train:
            action_dist = torch.distributions.Categorical(probs)
            action = action_dist.sample()
            return action.item()
        else:
            action = probs.cpu().detach().numpy().argmax(1)[0]
            return action
        
    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device)
        
        # state = torch.tensor([state], dtype=torch.float).to(self.device)
        # action = torch.tensor([action]).view(-1, 1).to(self.device)
        # reward = torch.tensor([reward], dtype=torch.float).view(-1, 1).to(self.device)
        # next_state = torch.tensor([next_state], dtype=torch.float).to(self.device)
        # done = torch.tensor([done], dtype=torch.float).to(self.device)

        # 时序差分目标
        td_target = rewards + self.gamma * self.critic(next_states)*(1-dones) # - 20*done
        # 时序差分误差
        td_delta = td_target - self.critic(states)

        # safe_logs = torch.clip(self.actor(states).gather(1,actions), 1e-10, 1)
        # log_probs = torch.log(safe_log)
        log_probs = torch.log(self.actor(states).gather(1, actions))

        # detach()将requires_grad为false，得到的这个tensor永远不需要计算其梯度，不具有grad
        critic_loss = torch.mean(F.mse_loss(self.critic(states), td_target.detach()))
        actor_loss = torch.mean(-log_probs * td_delta.detach())
        self.critic_optimizer.zero_grad()
        self.actor_optimizer.zero_grad()
        critic_loss.backward()
        actor_loss.backward()
        self.critic_optimizer.step()
        self.actor_optimizer.step()

