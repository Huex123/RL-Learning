# Trust Region Policy Optimization

文章：[pdf](https://arxiv.org/pdf/1502.05477.pdf)
参考：[翻译-CSDN](https://ananjiaoju.blog.csdn.net/article/details/125853594?spm=1001.2101.3001.6650.1&utm_medium=distribute.pc_relevant.none-task-blog-2~default~BlogCommendFromBaidu~Rate-1-125853594-blog-106960623.235^v38^pc_relevant_sort_base2&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2~default~BlogCommendFromBaidu~Rate-1-125853594-blog-106960623.235^v38^pc_relevant_sort_base2&utm_relevant_index=2)、[公式总结](https://blog.csdn.net/bbbeoy/article/details/106960623)、[TRPO-知乎](https://zhuanlan.zhihu.com/p/605886935)、

## 概要

通过对理论证明过程的几个近似，开发了一个实用的算法—信任区域策略优化(TRPO)。该算法类似于自然策略梯度法，对神经网络等大型非线性策略的优化是有效的。尽管它的近似偏离了理论，TRPO倾向于给出单调的改进，很少调整超参数。



![TRPO伪代码](.\algorithm.svg)





# CODE IMPLEMENTATION

[TRPO 算法 (boyuai.com)](https://hrl.boyuai.com/chapter/2/trpo算法)
[TensorLayer/examples/reinforcement_learning/tutorial_TRPO.py at master · tensorlayer/TensorLayer (github.com)](https://github.com/tensorlayer/TensorLayer/blob/master/examples/reinforcement_learning/tutorial_TRPO.py)

特点：近似求解、共轭梯度、线性搜索、广义优势估计(Generalized Advantage Estimation，GAE)

## architecture

- `TRPO.py`：TRPO模型
  - PolicyNet：策略网络
  - ValueNet：价值网络
  - TRPO：TRPO智能体
- `train.py`：训练模型并保存PolicyNet和ValueNet参数、train_rewards
- `play.py`：导入训练好的PolicyNet、ValueNet参数，生成最终模型，进行游戏并导出游戏画面gif、play_rewards图像png。

**步骤**：如上图伪代码所示，在每个episode中，收集轨迹放在GAE_Buffer中，当这个episode结束后，开始更新。
通过GAE采集数据和价值网络计算Advantage，即可求得 策略梯度$\hat{g}_k$；进行价值网络更新；不打算计算Hessian矩阵，但需要计算Hx，则先利用共轭梯度法计算x，再由公式 $Hx=\nabla_{\theta}((\nabla_{\theta}\overline{D}_{KL}(\theta || \theta_k))^Tx)$ 计算得到Hx，代入可进行策略更新。



## results

`CartPole-v1`环境最大奖励值为500。

训练`CartPole-v1`结果：

![TRPO_train_rewards1](D:\StudyNotes\Papers_Reproduction\RL\Policy Gradient\Trust Region Policy Optimization\CartPole\TRPO_train_rewards1.png)
