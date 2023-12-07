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


class REINFORCE:
    def __init__(self, state_dim, hidden_dim, action_dim, learning_rate=1e-3, gamma=0.98, device=torch.device("cpu")):
        self.policy_net = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=learning_rate) # 使用Adam优化器
        self.gamma = gamma
        self.device = device

    def take_action(self, state, is_train=True):
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        probs = self.policy_net(state)

        if is_train:
            action_dist = torch.distributions.Categorical(probs)
            action = action_dist.sample()
            return action.item()
        else:
            action = probs.cpu().detach().numpy().argmax(1)[0]
            return action
        
    def update(self, transition_dict):
        rewards = transition_dict['rewards']
        states = transition_dict['states']
        actions = transition_dict['actions']

        G = 0
        self.optimizer.zero_grad()
        for i in reversed(range(len(rewards))): # 从最后一步算起
            reward = rewards[i]
            state = torch.tensor([states[i]], dtype=torch.float).to(self.device)
            action = torch.tensor([actions[i]]).view(-1, 1).to(self.device)
            log_prob = torch.log(self.policy_net(state).gather(1, action))
            G = self.gamma * G + reward
            loss = -log_prob * G # 每一步的损失函数
            loss.backward() # 反向传播计算梯度
        self.optimizer.step() # 梯度下降

