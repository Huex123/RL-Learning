import gymnasium
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib import animation

env_name = "Pendulum-v1"  # "CartPole-v1" "Ant-v4" "Hopper-v4"
data = pd.read_csv(env_name+'_PPO_train_rewards3.csv')
fig = plt.figure()
axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])
axes.plot(list(range(len(data))), data.iloc[:, 1])
axes.set_title(env_name+"_PPO_train_rewards")
axes.set_xlabel("episode")
axes.set_ylabel("episode reward")
fig.savefig(env_name+'_PPO_train_rewards3.png')
