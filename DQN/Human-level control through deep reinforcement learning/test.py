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
env = gym.make('PongNoFrameskip-v4', render_mode='rgb_array')
env = AtariPreprocessing(env, scale_obs=False, terminal_on_life_loss=True) # 在丢失生命时会结束游戏
env = FrameStack(env, num_stack=4) # 4帧画面会合并为1个输入，加快学习

# print(env.observation_space.shape[0])
all_rewards = [1,3.25,4,5.72]
df_rewards = pd.DataFrame({'rewards':all_rewards}, index=list(range(len(all_rewards))), dtype=pd.Float32Dtype)
df_rewards.to_csv('test.csv')