# REINFORCE

[策略梯度算法 (boyuai.com)](https://hrl.boyuai.com/chapter/2/策略梯度算法)

[深度强化学习-策略梯度算法(Reinforce)代码_策略梯度代码_indigo love的博客-CSDN博客](https://blog.csdn.net/weixin_46133643/article/details/122439616)

## introduction

策略梯度：$\nabla_{\theta}J(\pi_{\theta})=\mathbb{E}_{\tau \sim \pi_{\theta}}[\sum_{t=0}^T \nabla_{\theta}\log{\pi_{\theta}(A_t|S_t)}R(\tau)]$.

REINFORCE 算法便是采用了蒙特卡洛方法来估计 $R(\tau)$ ，即：$R(\tau)=\sum_{t'=t}^T{\gamma^{t'-t}r_{t'}}$ 。

![algorhtim](https://github.com/Huex123/RL-Learning/blob/main/Policy%20Gradient/REINFORCE/algorhtim.png)



## code

### architecture

- `reinforce.py`：reinforce模型
  - PolicyNet：策略神经网络
  - REINFORCE：智能体
- `train.py`：训练模型并保存PolicyNet参数`CartPole_REINFORCE_policy_net.pt`、train_rewards
- `play.py`：导入训练好的PolicyNet参数`CartPole_REINFORCE_policy_net.pt`，生成最终模型，进行游戏并导出游戏画面gif、play_rewards图像png。



### results

![REINFORCE_train_rewards1](https://github.com/Huex123/RL-Learning/blob/main/Policy%20Gradient/REINFORCE/REINFORCE_train_rewards1.png)

其中，该CartPole环境最大奖励为500，可看到，训练很有效果，但是REINFORCE 算法的梯度估计的方差很大，可能会造成一定程度上的不稳定。
