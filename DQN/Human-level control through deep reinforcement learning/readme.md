# Human-level control through deep reinforcement learning

文章：[ Nature网页版](https://www.nature.com/articles/nature14236)、[pdf](https://www.nature.com/articles/nature14236.pdf)
参考：[中文翻译](https://blog.csdn.net/strin__aaa/article/details/134269364)、[代码实现](https://blog.csdn.net/weixin_45681037/article/details/117714761)

## 概要

使用深度卷积神经网络来逼近optimal action-value function。

当使用神经网络等非线性函数近似表示Q函数时，强化学习是**不稳定**的，原因：1、观测序列中存在的相关性；2、对Q函数小的更新可能会大大改变策略从而改变数据分布；3、action-value(Q)和target value的相关性。
解决办法：

1. **experience replay**，提高数据效率；对数据进行随机化，从而消除观测序列中的相关性；平滑数据分布的变化。在经验缓冲区中只存放最近的N个tuple。
2.  **target Q-network**，降低Q和target的相关性。

**Loss Function**：$L_i(\theta_i) = E_{(s,a,r,s')\in U(D)}[(r+\gamma\max_{a'}Q(s',a';\theta_i^-)-Q(s,a;\theta_i))^2]$ .
$\theta_i$ are the parameters of **Q-network**，$\theta_i^-$ are the parameters of **target Q-network** which aims to compute the target value.   The target Q-network parameters $\theta_i^-$ are only updated with Q-network parameters $\theta_i$ every $C$ steps and are held fixed between individual updates.

## Methods

### 预处理

减低输入维度：为了对单个帧进行编码，采用正在编码的帧和前一帧上每个像素颜色值的最大值；从 RGB 帧中提取 Y 通道（亮度），并将其重新调整为 84*84；最后取最近的4帧画面组成 $84 \times 84 \times4$ 的输入张量。

### 模型架构

Q网络：输入：状态，$84 \times 84 \times 4$，输出：该状态下每个动作$a_i$的action value，即$Q(s,a_i)$，能够计算给定状态下所有可能动作的 Q 值。
输入预处理生成的 $84 \times 84 \times 4$ 图像，
第一个隐藏层将 $32$ 个步长为 $4$ 的 $8 \times 8$ 滤波器与输入图像进行卷积，并应用非线性整流器。
第二个隐藏层使用 $64$ 个 $4 \times 4$ 的卷积核进行步长为 $2$ 的卷积操作，并应用非线性整流器。
第三个卷积层使用 $64$ 个 $3 \times 3$ 的卷积核进行步长为 $1$ 的卷积操作，并应用整流器。
最后一个隐藏层是 $512$ 个整流单元组成的全连接层。
输出层是一个线性全连接层，每个有效动作对应一个输出。

### 训练详情

对 49 款 Atari 2600 游戏进行了实验。每个游戏都训练了不同的网络：所有游戏都使用相同的网络架构、学习算法和超参数设置，表明我们的方法足够强大，可以在各种游戏上工作，同时只包含最少的先验知识。由于不同游戏的分数大小差异很大，我们**将所有正奖励削减为 1，将所有负奖励削减为 -1，保留 0 奖励不变**。以这种方式削减奖励可以限制误差导数的规模，并且可以更轻松地在多个游戏中使用相同的学习率，但是agent将无法区分不同大小的奖励。

behaviour policy在训练中采用 $\varepsilon-greedy$ 策略，$\varepsilon$ 在前一百万帧中从 1.0 线性退火到 0.1，此后固定为 0.1。

frame-skipping（跳帧）：agent每 k 帧选择一次动作，在跳过的帧中重复其最后一个动作。

### Algorithm



![image-20231114210109369](C:\Users\Lenovo\AppData\Roaming\Typora\typora-user-images\image-20231114210109369.png)



# CODE IMPLEMENTATION

[WTF-DeepRL/01_DQN/readme.md at master · AmazingAng/WTF-DeepRL (github.com)](https://github.com/AmazingAng/WTF-DeepRL/blob/master/01_DQN/readme.md)

特点：target Q-network、exprience replay、Prioritized experience replay

训练Atari的Pong游戏

## architecture

- `DQN.py`：DQN模型
  - Q_net
  - Memory_Buffer_PER、SumTree
  - DQNagent
- `train.py`：训练模型并保存Qnet参数`Pong_Qnet.pt`、train_rewards
- `play.py`：导入训练好的Qnet参数`Pong_Qnet.pt`，生成最终模型，进行游戏并导出游戏画面gif、play_rewards图像png。

## results

训练得到 无Prioritized experience replay的模型（序号1）和 Prioritized experience replay的模型（序号2、3）

单纯experience replay无Prioritized experience replay（深蓝色） 和 Prioritized experience replay（玫红色、浅蓝色）的训练对比：

![image-20231119153156638](C:\Users\Lenovo\AppData\Roaming\Typora\typora-user-images\image-20231119153156638.png)

![image-20231118214020034](C:\Users\Lenovo\AppData\Roaming\Typora\typora-user-images\image-20231118214020034.png)

可以看到收敛速度有所提高。
