# Advantage Actor-Critic

[Actor-Critic 算法 (boyuai.com)](https://hrl.boyuai.com/chapter/2/actor-critic算法)

## introduction

基础的Actor-Critic中，Critic输出Q value用于策略梯度的计算，这种方式也有跟 policy gradient 中类似的问题：High variance(高方差)。解决方法：引入baseline $b(S_t)$。

AC with baseline ：$\nabla_{\theta}J(\pi_{\theta})=\mathbb{E}_{\tau \sim \pi_{\theta}}[\sum_{t=0}^T \nabla_{\theta}\log{\pi_{\theta}(A_t|S_t)}~(Q^{\pi_\theta}(S_t,A_t)-b(S_t))]$ .
 用 $V(S_t)$ 作为 baseline，得到优势函数Advantage function：$A^{\pi_\theta}(S_t,A_t)=Q^{\pi_\theta}(S_t,A_t)-V^{\pi_\theta}(S_t)$。用其代替上式，即得到 A2C。

具体实现时不需要让Critic网络同时输出Q值和V值，只需输出V值，Advantage可通过时序差分残差近似得到：
$A^{\pi_\theta}(S_t,A_t)=Q^{\pi_\theta}(S_t,A_t)-V^{\pi_\theta}(S_t)=R_t+\gamma V^{\pi_\theta}(S_{t+1})-V^{\pi_\theta}(S_t)$ .

**A2C** :

Actor : $\nabla_{\theta}J(\pi_{\theta})=\mathbb{E}_{\tau \sim \pi_{\theta}}[\sum_{t=0}^T \nabla_{\theta}\log{\pi_{\theta}(A_t|S_t)}~A^{\pi_\theta}(S_t,A_t)]$ .

Critic : $L(w)=\frac{1}{2}(r+\gamma V_w(s_{t+1})-V_w(s_t))^2$.
采取类似于目标网络的方法，将上式中$r+\gamma V_w(s_{t+1})$作为时序差分目标，TD目标不会产生梯度来更新价值函数。
即：$\nabla_wL(w)=-(r+\gamma V_w(s_{t+1})-V_w(s_t))\nabla_wV_w(s_t)$ .



AC单步更新，因为直接可求出优势函数A，而reinforce则蒙特卡洛跑完一回合才得到序列的R。根据更新公式，V值网络显然可以单步更新；策略网络是累加的更新，可以单步逐步更新来达到要求。

![p1](https://github.com/Huex123/RL-Learning/blob/main/Policy%20Gradient/A2C/p1.png)

小技巧：

![p2](https://github.com/Huex123/RL-Learning/blob/main/Policy%20Gradient/A2C/p2.png)

伪代码：

![img](https://github.com/Huex123/RL-Learning/blob/main/Policy%20Gradient/A2C/algorithm.png)

但此次训练还是1个episode更新一次。

## code

### architecture

- `reinforce.py`：A2C模型
  - PolicyNet：Actor 策略网络
  - ValueNet：Critic 价值网络
  - A2Cagent：智能体
- `train.py`：训练模型并保存PolicyNet和ValueNet的参数`CartPole_A2C_actor_net.pt、CartPole_A2C_critic_net.pt`、train_rewards。
- `play.py`：导入训练好的网络参数，生成最终模型，进行游戏并导出游戏画面gif、play_rewards图像png。
- `test.py`：进行测试的临时脚本



### results

`CartPole-v1`环境最大奖励值为500。

训练过程中出现reward断崖式下跌，

[神经网络训练，碰到断崖式下跌，且每次训练效果都相差极大-知乎](https://www.zhihu.com/question/61076394/answer/183685893)

![failed](https://github.com/Huex123/RL-Learning/blob/main/Policy%20Gradient/A2C/failed.png)

根据实验结果我们可以发现，Actor-Critic 算法很快便能收敛到最优策略，并且训练过程非常稳定，抖动情况相比 REINFORCE 算法有了明显的改进，这说明价值函数的引入减小了方差。

A2C(红色)、REINFORCE(蓝色)。

![train_rewards](https://github.com/Huex123/RL-Learning/blob/main/Policy%20Gradient/A2C/train_rewards.png)
