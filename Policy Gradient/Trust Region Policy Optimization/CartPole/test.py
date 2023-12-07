import pickle, os.path, math, glob, gym, time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from TRPO import *

data = pd.read_csv('TRPO_train_rewards1.csv')

fig = plt.figure()
axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])
axes.plot(list(range(len(data))), data.iloc[:, 1])
axes.set_title("TRPO_train_rewards")
axes.set_xlabel("episode")
axes.set_ylabel("episode reward")
fig.savefig('TRPO_train_rewards1.png')
