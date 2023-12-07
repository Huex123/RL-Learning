# Proximal Policy Optimization Algorithms

论文：[pdf](https://arxiv.org/pdf/1707.06347.pdf)
参考：[翻译](https://blog.csdn.net/qq_28385535/article/details/105041637)、

## 概要

由于标准的策略梯度算法每轮数据（每轮数据定义：从游戏开始到结束的一轮完整数据）只能更新一个梯度，提出了一种新的方法，使得每个数据集上可以更新多次。近端策略优化（PPO）具有信赖域策略优化（TRPO）的一些优点，但PPO实施起来更简单，更通用，并且具有更好的样本复杂性。信赖域策略优化（TRPO）相对复杂，并且对于具有噪声（例如，dropout）或参数共享（在策略和价值函数之间或与辅助任务之间）的网络结构不兼容。

TRPO合理的理论实际上建议使用惩罚项而不是约束项，但是很难选择一个在不同问题上甚至在特征随学习过程变化的单个问题内表现良好的系数$\beta$值，故采用约束的形式。PPO则用来改进这个问题。

PPO 有两种形式，一是PPO-截断（PPO-Clip），二是PPO-惩罚（PPO-Penalty）；大量实验表明，PPO-截断总是比 PPO-惩罚表现得更好。

##  Clipped Surrogate Objective

令 $r_t(\theta)=\frac{\pi_{\theta}(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$，则 $r(\theta_{old})=1$。TRPO的目标是最大化：$L^{CPI}(\theta)=\hat{\mathbb{E}}_t[\frac{\pi_{\theta}(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}\hat{A}_t]=\hat{\mathbb{E}}_t[r_t(\theta)\hat{A}_t]$ 且保证约束。在没有约束项的情况下，$L^{CPI}$ 的最大化将导致策略更新过大。

**PPO-截断**的目标是最大化：$L^{CLIP}(\theta)=\hat{\mathbb{E}}_t[\min(r_t(\theta)\hat{A}_t,~clip(r_t(\theta),1-\epsilon,1+\epsilon)\hat(A)_t)]$。以惩罚$r_t(\theta)$远离1的策略更新，即保证前后两策略不要相差太大。

<img src=".\p1.png" alt="image-20231129185124790" style="zoom: 67%;" />

$L^{CLIP}$ 是 $L^{CPI}$ 的下界。

## Adaptive KL Penalty Coefficient

使用一个KL散度作为惩罚项，并自适应调整惩罚系数 $\beta$，以使每次策略更新都能达到KL散度的一些目标值 $d_{targ}$ 。

**PPO-惩罚**的目标是最大化：$L^{KLPEN}(\theta)=\hat{\mathbb{E}}_t[~\frac{\pi_{\theta}(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}\hat{A}-\beta ~KL[\pi_{\theta_{old}}(\cdot|s_t),~\pi_{\theta}(\cdot|s_t)]~]$；
再
$$
\begin{align}
计算& d=\hat{\mathbb{E}}_t[~KL[\pi_{\theta_{old}}(\cdot|s_t),~\pi_{\theta}(\cdot|s_t)]~]\\
&if~d<d_{targ}/1.5,~ \beta=\beta/2 \\
&if~d>d_{targ}\times1.5,~\beta=\beta \times 2  \\
\end{align}
$$
更新后的 $\beta$ 用于下一个策略更新。上面的参数1.5和2是通过试探法选择的，但是算法对其并不十分敏感。$\beta$ 的初始值是另一个超参数，但在实践中并不重要，因为该算法会对其进行快速调整。

## Algorithm

如果使用在策略和价值函数之间共享参数的神经网络体系结构，则必须使用将策略损失和价值损失项组合在一起的损失函数，通过增加熵值奖赏以确保足够的探索来进一步增强此目标函数。结合这些损失项，我们获得以下目标函数，并在每次迭代时最大化该目标函数：$L^{CLIP+VF+S}_t(\theta)=\hat{\mathbb{E}}_t[L^{CLIP}_t(\theta)-c_1L^{VF}_t(\theta)+c_2S[\pi_{\theta}](s_t)]$ 。
其中，$c_1、c_2$ 是系数，$S$ 是entropy bonus，$L^{VF}_t=(V_{\theta}(s_t)-V^{targ}_t)^2$是均方误差损失。 

<img src=".\algorithm1.png" alt="image-20231129192557375" style="zoom: 50%;" />

<img src=".\algorithm2.png" alt="image-20231129202844790" style="zoom:50%;" />

PPO是off-policy，故可采样一个episode，然后用这些数据更新n次，然后再继续下一个episode。

# CODE IMPLEMENTATION

[Mujoco-Pytorch/agents/ppo.py at main · seolhokim/Mujoco-Pytorch (github.com)](https://github.com/seolhokim/Mujoco-Pytorch/blob/main/agents/ppo.py)

[PPO-pytorch-Mujoco/PPO.py at master · qingshi9974/PPO-pytorch-Mujoco (github.com)](https://github.com/qingshi9974/PPO-pytorch-Mujoco/blob/master/PPO.py)

[ChatGPT强化学习大杀器——近端策略优化（PPO）_ppo近端策略优化-CSDN博客](https://blog.csdn.net/jarodyv/article/details/129355131)

[Reinforcement-Implementation/code/ppo.py at master · zhangchuheng123/Reinforcement-Implementation (github.com)](https://github.com/zhangchuheng123/Reinforcement-Implementation/blob/master/code/ppo.py)

## architecture

- `PPO.py`：PPO模型
  - PolicyNet：策略网络
  - ValueNet：价值网络
  - PPO：PPO智能体
- `train.py`：训练模型并保存PolicyNet和ValueNet参数、train_rewards
- `play.py`：导入训练好的PolicyNet、ValueNet参数，生成最终模型，进行游戏并导出游戏画面gif、play_rewards图像png。



## results

`Hopper-v4`环境下训练奖励如下：

![Hopper-v4_PPO_train_rewards3](D:\StudyNotes\Papers_Reproduction\RL\Policy Gradient\Proximal Policy Optimization Algorithms\2\Hopper-v4\Hopper-v4_PPO_train_rewards3.png)

