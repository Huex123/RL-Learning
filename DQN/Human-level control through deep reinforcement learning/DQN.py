import random

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd

class Q_net(nn.Module):
    def __init__(self, in_channels=4, n_actions=6):
        super(Q_net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
        self.fc4 = nn.Linear(7*7*64, 512)
        self.fc5 = nn.Linear(512, n_actions)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.fc4(x.reshape(x.size(0), -1)))
        return self.fc5(x)
    
class SumTree:
    write = 0
    
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2*capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.n_entries = 0
    
    def _propagate(self, idx, change):
        parent = (idx - 1)//2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)
    
    def _retrieve(self, idx, s):
        left = 2*idx + 1
        right = left + 1
        if left >= len(self.tree):
            return idx
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])
        
    def total(self):
        return self.tree[0]
    
    def add(self, p, data):
        idx = self.write + self.capacity - 1
        self.data[self.write] =  data
        self.update(idx, p)
        self.write += 1
        if self.write >= self.capacity:
            self.write = 0
        if self.n_entries < self.capacity:
            self.n_entries += 1

    def update(self, idx, p):
        change = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1
        return (idx, self.tree[idx], self.data[dataIdx])


class Memory_Buffer_PER(object):
    # stored as (s, a, r, s_) in SumTree
    def __init__(self, memory_size=100, a=0.6, e=0.01):
        self.tree = SumTree(memory_size)
        self.memory_size = memory_size
        self.prio_max = 0.1
        self.a = a
        self.e = e

    def push(self, state, action, reward, next_action, done):
        data = (state, action, reward, next_action, done)
        p = (np.abs(self.prio_max) + self.e) ** self.a # proportional priority
        self.tree.add(p, data)

    def sample(self, batch_size):
        states, actions, rewards, next_states, dones = [],[],[],[],[]
        idxs = []
        segment = self.tree.total() / batch_size
        priorities = []
        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            idx, p, data = self.tree.get(s)
            state, action, reward, next_state, done = data
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)
            priorities.append(p)
            idxs.append(idx)
        return idx, np.concatenate(states), actions, np.concatenate(next_states), dones
    
    def update(self, idxs, errors):
        self.prio_max = max(self.prio_max, max(np.abs(errors)))
        for i, idx in enumerate(idxs):
            p = (np.abs(errors[i]) + self.e) ** self.a
            self.tree.update(idx, p)

    def size(self):
        return self.tree.n_entries


class Memory_Buffer(object):
    def __init__(self, memory_size=10000):
        self.buffer = []
        self.memory_size = memory_size
        self.next_idx = 0

    def push(self, state, action, reward, next_state, done):
        data = (state, action, reward, next_state, done)
        if len(self.buffer) <= self.memory_size : # buffer is not full
            self.buffer.append(data)
        else: # buffer is full, then throw out the earliest element and join the new element
            self.buffer[self.next_idx] = data
        self.next_idx = (self.next_idx + 1) % self.memory_size

    def sample(self, batch_size):
        states, actions, rewards, next_states, dones = [],[],[],[],[]
        for i in range(batch_size):
            idx = random.randint(0, self.size() -1)
            data = self.buffer[idx]
            state, action, reward, next_state, done = data
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)
        return np.concatenate(states), actions, rewards, np.concatenate(next_states), dones
    
    def size(self):
        return len(self.buffer)
    

class DQNagent:
    def __init__(self, in_channels=1, action_space=[], USE_CUDA=False, memory_size=10000, \
                 prio_a=0.6, prio_e=0.001, epsilon=1, lr=1e-4):
        self.epsilon = epsilon
        self.action_space = action_space
        self.memory_buffer = Memory_Buffer_PER(memory_size, a=prio_a, e=prio_e)
        self.Qnet = Q_net(in_channels=in_channels, n_actions=action_space.n)
        self.target_Qnet = Q_net(in_channels=in_channels, n_actions=action_space.n)
        self.target_Qnet.load_state_dict(self.Qnet.state_dict())
        
        self.USE_CUDA = USE_CUDA
        if USE_CUDA:
            self.Qnet = self.Qnet.cuda()
            self.target_Qnet = self.target_Qnet.cuda()
        self.optimizer = optim.RMSprop(self.Qnet.parameters(), lr=lr, eps=0.001, alpha=0.95)

    def observe(self, lazyframe):
        # 将Atari环境每一步返回的observation（4*84*84）转为状态（pytorch tensor）
        # from Lazy frame to tensor
        state = torch.from_numpy(lazyframe.__array__()[None]/255).float()
        if self.USE_CUDA:
            state = state.cuda()
        return state
    
    def value(self, state):
        # 返回状态的Q值
        q_values = self.Qnet(state)
        return q_values
    
    def act(self, state, epsilon=None):
        # sample actions with epsilon-greedy policy
        if epsilon is None:
            epsilon = self.epsilon
        q_values = self.value(state).cpu().detach().numpy()
        if random.random() < epsilon:
            action = random.randrange(self.action_space.n)
        else:
            action = q_values.argmax(1)[0]
        return action
    
    def compute_td_loss(self, states, actions, rewards, next_states, is_done, gamma=0.99):
        # compute td loss
        actions = torch.tensor(actions).long() # shape: [batch_size]
        rewards = torch.tensor(rewards, dtype=torch.float) # shape: [batch_size]
        is_done = torch.tensor(is_done).bool() # shape: [batch_size]
        if self.USE_CUDA:
            actions = actions.cuda()
            rewards = rewards.cuda()
            is_done = is_done.cuda()

        predicted_qvalues = self.Qnet(states) # 网络输出为每个动作对应的Q值
        predicted_qvalues_for_actions = predicted_qvalues[range(states.shape[0]), actions]
        predicted_next_values = self.target_Qnet(next_states)
        td_target = rewards + gamma*predicted_next_values.max(-1)[0]
        # 若为最后一个动作，则td_target = reward, 无下一个Q
        td_target = torch.where(is_done, rewards, td_target)
        td_loss = F.smooth_l1_loss(predicted_qvalues_for_actions, td_target.detach())
        return td_loss
    
    def sample_from_buffer(self, batch_size):
        states, actions, rewards, next_states, dones = [],[],[],[],[]
        idxs = []
        segment = self.memory_buffer.tree.total() / batch_size
        priorities = []
        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            idx, p, data = self.memory_buffer.tree.get(s)

            frame, action, reward, next_frame, done = data
            states.append(self.observe(frame))
            actions.append(action)
            rewards.append(reward)
            next_states.append(self.observe(next_frame))
            dones.append(done)
            priorities.append(p)
            idxs.append(idx)
        return torch.cat(states), actions, rewards, torch.cat(next_states), dones
    
    def learn(self, batch_size):
        if self.memory_buffer.size() > batch_size:
            states, actions, rewards, next_states, dones = self.sample_from_buffer(batch_size)
            td_loss = self.compute_td_loss(states, actions, rewards, next_states, dones)
            self.optimizer.zero_grad()
            td_loss.backward()
            for param in self.Qnet.parameters():
                param.grad.data.clamp_(-1, 1) # 将张量每个元素的值压缩到区间 [min,max]
            self.optimizer.step()
            return td_loss.item()
        else:
            return 0

        