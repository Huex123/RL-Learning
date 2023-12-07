# Reinforcement Learning

[yingchengyang/Reinforcement-Learning-Papers(github.com)](https://github.com/yingchengyang/Reinforcement-Learning-Papers)

[The latest in Machine Learning | Papers With Code](https://paperswithcode.com/)

[强化学习基础 Ⅳ: State-of-the-art 强化学习经典算法汇总 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/137208923)

## DQN

[[model-free\] 经典强化学习论文合集 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/89058164)

[（CODEs）AmazingAng/WTF-DeepRL: Deep RL algorithm in pytorch (github.com)](https://github.com/AmazingAng/WTF-DeepRL)

**Loss Function**：$L_i(\theta_i) = E_{(s,a,r,s')\in U(D)}[(r+\gamma\max_{a'}Q(s',a';\theta_i^-)-Q(s,a;\theta_i))^2]$ .

## Policy Gradient

![img](https://img-blog.csdnimg.cn/20191117155915950.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MjM4OTM0OQ==,size_16,color_FFFFFF,t_70)

策略梯度：$\nabla_{\theta}J(\pi_{\theta})=\mathbb{E}_{\tau \sim \pi_{\theta}}[\sum_{t=0}^T \nabla_{\theta}\log{\pi_{\theta}(A_t|S_t)}R(\tau)]$.

或者：$\nabla_{\theta}J(\pi_{\theta})=\mathbb{E}_{\tau \sim \pi_{\theta}}[\sum_{t=0}^T \nabla_{\theta}\log{\pi_{\theta}(A_t|S_t)}\Phi_t]$.
其中 $\Phi_t=\sum_{t'=t}^T(R(S_{t'},a_{t'},S_{t'+1})-b(S_t))$ ，$\Phi_t$也可被替换为其他形式。

### Actor-Critic(AC)

[深度强化学习8——Actor-Critic（AC、A2C、A3C）_probability distribution actor-critic (ac) agent-CSDN博客](https://blog.csdn.net/weixin_42389349/article/details/103109659)

将$\Phi_t$ 变为 $Q^w(S_t,A_t)$，即：$\nabla_{\theta}J(\pi_{\theta})=\mathbb{E}_{\tau \sim \pi_{\theta}}[\sum_{t=0}^T \nabla_{\theta}\log{\pi_{\theta}(A_t|S_t)}Q^w(S_t,A_t)]$.

### Advantage Actor-Critic(A2C)

用**优势函数** $A^{\pi_{\theta}}(S_t,A_t)=Q^{\pi_{\theta}}(S_t,A_t)-V^{\pi_{\theta}}(S_t)=R_t+\gamma V^{w}(S_{t+1})-V^{w}(S_{t})$  代替 $Q^w(S_t,A_t)$，
即：$\nabla_{\theta}J(\pi_{\theta})=\mathbb{E}_{\tau \sim \pi_{\theta}}[\sum_{t=0}^T \nabla_{\theta}\log{\pi_{\theta}(A_t|S_t)}A^{\pi_{\theta}}(S_t,A_t)]$.

### Asynchronous Advantage Actor-Critic  (A3C)

A3C是上一节中 A2C 的异步版本。

### Trust Region Policy Optimization (TRPO)

该算法类似于自然策略梯度法，对神经网络等大型非线性策略的优化是有效的。

### Proximal Policy Optimization Algorithms(PPO)

PPO 的优化目标与 TRPO 相同，但 PPO 用了一些相对简单的方法来求解。
