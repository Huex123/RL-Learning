# Deep Reinforcement Learning with Double Q-learning

文章：[pdf](https://arxiv.org/pdf/1509.06461.pdf)
参考：[Double DQN(DDQN)原理与实现-知乎](https://zhuanlan.zhihu.com/p/97853300)、[DQN及其变种(Double DQN，优先回放，Dueling DQN)](https://blog.csdn.net/weixin_45526117/article/details/123308645?spm=1001.2101.3001.6661.1&utm_medium=distribute.pc_relevant_t0.none-task-blog-2~default~BlogCommendFromBaidu~Rate-1-123308645-blog-120989159.235^v38^pc_relevant_sort_base2&depth_1-utm_source=distribute.pc_relevant_t0.none-task-blog-2~default~BlogCommendFromBaidu~Rate-1-123308645-blog-120989159.235^v38^pc_relevant_sort_base2&utm_relevant_index=1)

## 概要

Q-learning中会发生对action value的过估计，影响实验性能。过估计本身并不一定是个问题，如果所有的价值都一样高，相对的行动偏好就会保持不变。但是如果所有价值不是都过估计，过估计量是不均匀的，就会产生问题。
对于DQN，即使其使用神经网络来更灵活地近似Q函数，有可能实现较低的渐近逼近误差，还是会过估计action value。

之前的Double Q-learning算法（在此论文之前已提出）是用在表格上的，现在将其用在神经网络上，即产生了Double DQN。

## Background

### Deep Q Networks

With **target Q-network**、**experience replay**.

For the tuple $(S_t, A_t, R_{t+1},S_{t+1})$

TD target is：$Y_t^{DQN}=R_{t+1}+\gamma \max_a{Q(S_{t+1},a;\theta _t^-)}$，
TD error is：$Y_t^{DQN}-Q(S_t,A_t;\theta _t)=R_{t+1}+\gamma \max_a{Q(S_{t+1},a;\theta _t^-)}-Q(S_t,A_t;\theta _t)$.

$S_T$通过$\theta_t$生成$A_t$，$\theta_t$是$\varepsilon-greedy$策略.

称 $\theta _t$ 为 **online network**，因为其每次都更新；而 $\theta _t^-$ 间隔性与 $\theta _t$ 同步。

### Double Q-learning

- 动作选择

求解TD target时选择动作A*使max成立

- 动作评估

选出A*后用其action value 来构造TD target

传统DQN用同一个值来*选择动作*和*评估动作*，即用 $\theta_t^-$ 来选择使 $R_{t+1}+\gamma \max_a{Q(S_{t+1},a;\theta _t^-)}$ 成立的A*，再使用同样的 $\theta_t^-$ 计算TD target，会使其选择高估的值，并导致过高的评估。
即：$Y_t^Q=R_{t+1}+\gamma Q(~S_{t+1},~ \arg\max_a{Q(S_{t+1}, a; \theta_t^-); ~\theta_t^-}~)$.

Double Q-learning通过将选择动作和评估动作分割开来避免过高估计的问题，即用 $\theta_t$ 来选择 $R_{t+1}+\gamma \max_a{Q(S_{t+1},a;\theta _t)}$ ，再使用 $\theta_t'$ 来计算TD target。
即：$Y_t^{DoubleQ}=R_{t+1}+\gamma Q(~S_{t+1},~ \arg\max_a{Q(S_{t+1}, a; \theta_t); ~\theta_t'}~)$.

## Overoptimism due to estimation errors

上界估计、下界估计，Double Q-learning的下界绝对误差为0。

## Double DQN

如上面Double Q-learning所述，为将*动作选择*和*动作评估*分隔开，需要两个value function，而DQN结构刚好有两个Q-network，故不需要再引入额外的network。即用 $\theta_t$ 来选择动作，用 $\theta_t^-$ 来计算TD target。

即：${Y_t^{DoubleQ}=R_{t+1}+\gamma Q(~S_{t+1},~ \arg\max_a{Q(S_{t+1}, a; \theta_t); ~\theta_t^-}~)}$.

## Empirical results

DQN最终收敛结果偏高，而且与真实值有较大差距；Double DQN最终收敛结果与真实值较为接近。





# CODE IMPLEMENTATION

[WTF-DeepRL/03_DDQN/readme.md at master · AmazingAng/WTF-DeepRL (github.com)](https://github.com/AmazingAng/WTF-DeepRL/blob/master/03_DDQN/readme.md)

特点：target Q-network、exprience replay、Prioritized experience replay、Double DQN

## architecture

- `Double_DQN.py`：DQN模型
  - Q_net
  - Memory_Buffer_PER、SumTree
  - DDQNagent
- `train.py`：训练模型并保存Qnet参数`Pong_Doube_Qnet.pt`、train_rewards
- `play.py`：导入训练好的Qnet参数`Pong_Double_Qnet.pt`，生成最终模型，进行游戏并导出游戏画面gif、play_rewards图像png。



## results

Double DQN（深蓝色）和DQN（玫红色）都使用Prioritized experience replay，在Pong上训练结果如下：

![train_rewards](.\train_rewards.png)

在Pong上的Double DQN效果并不明显。
