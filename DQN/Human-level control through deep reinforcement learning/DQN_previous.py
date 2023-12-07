import random, pickle, os.path, math, glob, gym,time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import pdb

from gym.wrappers.atari_preprocessing import AtariPreprocessing # 预处理封装器
from gym.wrappers.frame_stack import LazyFrames # 去重函数，相同画面的图像就不重复存储了
from gym.wrappers.frame_stack import FrameStack # 每次都将最后n帧当成状态
from torch.utils.tensorboard import SummaryWriter

USE_CUDA = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor
Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if USE_CUDA \
else autograd.Variable(*args, **kwargs)
print(USE_CUDA)


class Q_net(nn.Module):
    def __init__(self, in_channels=4, n_actions=5):
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
    def __init__(self, in_channels=1, action_space=[], USE_CUDA=False, memory_size=10000, epsilon=1, lr=1e-4):
        self.epsilon = epsilon
        self.action_space = action_space
        self.memory_buffer = Memory_Buffer(memory_size)
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
        # 若为最后一个动作，则td_target=reward, 无下一个Q
        td_target = torch.where(is_done, rewards, td_target)
        td_loss = F.smooth_l1_loss(predicted_qvalues_for_actions, td_target.detach())
        return td_loss
    
    def sample_from_buffer(self, batch_size):
        states, actions, rewards, next_states, dones = [],[],[],[],[]
        for i in range(batch_size):
            idx = random.randint(0, self.memory_buffer.size()-1)
            data = self.memory_buffer.buffer[idx]
            frame, action, reward, next_frame, done = data
            states.append(self.observe(frame))
            actions.append(action)
            rewards.append(reward)
            next_states.append(self.observe(next_frame))
            dones.append(done)
        return torch.cat(states), actions, rewards, torch.cat(next_states), dones
    
    def learn(self, batch_size):
        if self.memory_buffer.size() > batch_size:
            states, actions, rewards, next_states, dones = self.sample_from_buffer(batch_size)
            td_loss = self.compute_td_loss(states, actions, rewards, next_states, dones)
            self.optimizer.zero_grad()
            td_loss.backward()
            for param in self.Qnet.parameters():
                param.grad.data.clamp_(-1, 1)
            self.optimizer.step()
            return td_loss.item()
        else:
            return 0
        

# if __name__ == '__main__':

# Training DQN in PongNoFrameskip-v4
env = gym.make('PongNoFrameskip-v4')#, render_mode='human')
env = AtariPreprocessing(env, scale_obs=False, terminal_on_life_loss=True) # 在丢失生命时会结束游戏
env = FrameStack(env, num_stack=4) # 4帧画面会合并为1个输入，加快学习

gamma = 0.99
epsilon_max = 1
epsilon_min = 0.05
eps_decay = 30000
frames = 2000000
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
agent = DQNagent(in_channels = state_channel, action_space= action_space, USE_CUDA = USE_CUDA, lr = learning_rate, memory_size = max_buff)

frame, _ = env.reset()

episode_reward = 0
all_rewards = []
losses = []
episode_num = 0
is_win = False
# tensorboard
summary_writer = SummaryWriter(log_dir = "DQN_stackframe", comment= "good_makeatari")

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
        print("frames: %5d, reward: %5f, loss: %4f, epsilon: %5f, episode: %4d" % (i, np.mean(all_rewards[-10:]), loss, epsilon, episode_num))
        summary_writer.add_scalar("Temporal Difference Loss", loss, i)
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

summary_writer.close()

    
        