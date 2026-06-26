# 机器学习与深度学习深度入门课程 · 中英文术语对照表（GLOSSARY）

> 速查约定：每条格式为 **中文名 | English | 一句话精准解释（注所在章）**。
> 分组大致按课程主题；同一术语跨章出现时归入其最核心的主题组，必要时在解释里指出其它出场章。
> 缩写在领域内通用时随术语标注（如 ML、DL、ERM、MLE、MAP、SGD、PCA、SVD、CNN、RNN、LSTM、GRU、GAN、VAE、ELBO、MDP、RL、GNN、ReLU、BN、LN、KL、MSE、SVM、kNN、GBDT、RBF、AUC、MDL、ViT、LLM、RLHF）。

---

## 目录

1. [学习框架与泛化](#1-学习框架与泛化)
2. [线性模型与正则化](#2-线性模型与正则化)
3. [优化](#3-优化)
4. [经典机器学习方法](#4-经典机器学习方法)
5. [无监督学习与降维](#5-无监督学习与降维)
6. [神经网络基础](#6-神经网络基础)
7. [深度架构](#7-深度架构)
8. [训练深度网络的工程学](#8-训练深度网络的工程学)
9. [生成模型](#9-生成模型)
10. [自监督与强化学习](#10-自监督与强化学习)
11. [评估、流水线与负责任的 AI](#11-评估流水线与负责任的-ai)
12. [人物·文献·里程碑](#12-人物文献里程碑)

---

## 1. 学习框架与泛化

| 中文名 | English | 解释 |
|---|---|---|
| 机器学习 | Machine learning (ML) | 不写规则，而让机器从"输入-答案"数据里归纳出能推广的函数（第 1 章）。 |
| 归纳推断 | Inductive inference | 从有限具体例子归纳出能推广到新例子的一般规律，"学习"的本质（第 1 章）。 |
| 监督学习 | Supervised learning | 每个输入配正确标签，学 $\mathbf x\mapsto y$ 的映射，分回归与分类（第 1、2 章）。 |
| 无监督学习 | Unsupervised learning | 数据只有输入无标签，自己发现结构（聚类 / 降维 / 密度估计）（第 1、6 章）。 |
| 强化学习 | Reinforcement learning (RL) | 智能体在环境中靠奖励试错学策略、最大化长期回报（第 1、10 章）。 |
| 自监督学习 | Self-supervised learning (SSL) | 从数据自身结构自动造监督信号（如预测被遮的词），大模型引擎（第 1、10 章）。 |
| 半监督学习 | Semi-supervised learning | 少量带标签 + 大量无标签一起学，借无标签数据的结构补足监督（第 1 章）。 |
| 伪标签 | Pseudo-labeling | 用当前模型给无标签数据打标签再当真标签训，半监督常用招（第 1 章）。 |
| 回归 | Regression | 输出连续值的监督任务（预测房价 / 气温）（第 1、2 章）。 |
| 分类 | Classification | 输出离散类别的监督任务（垃圾邮件 / 猫狗）（第 1、2 章）。 |
| 泛化 | Generalization | 机器学习的中心问题：推广到没见过的新数据而非记住训练集（第 1、2、9 章）。 |
| 记忆 | Memorization | 死记训练样本（如查找表），训练误差零但泛化为零，是泛化的反面（第 1、2 章）。 |
| 训练误差 | Training error | 模型在见过的训练数据上的误差，低不代表好（第 1、2 章）。 |
| 测试误差 / 泛化误差 | Test / generalization error | 模型在没见过的新数据上的误差，机器学习真正在乎的量（第 1、2 章）。 |
| 泛化差距 | Generalization gap | 测试误差减训练误差，暗线一的量化身，随容量增大而张大（第 1、2 章）。 |
| 过拟合 | Overfitting | 把训练数据的噪声 / 偶然细节当规律学了，训练好测试差（第 1、2 章）。 |
| 欠拟合 | Underfitting | 模型太简单，连训练集真规律都没抓住，训练测试都差（第 1、2 章）。 |
| 容量 / 复杂度 | Capacity / complexity | 假设空间"能表达多少种函数"的丰富度，不等于参数个数（第 2 章）。 |
| 偏差-方差权衡 | Bias–variance tradeoff | 期望平方误差 $=\text{Bias}^2+\text{Var}+\sigma^2$，容量调控前两项此消彼长（第 2 章）。 |
| 偏差 | Bias | 模型平均预测偏离真值的系统性误差，来自假设空间受限（第 2 章）。 |
| 方差 | Variance | 换一批训练集预测会抖多少，来自对训练随机性敏感，过拟合化身（第 2 章）。 |
| 不可约噪声 | Irreducible noise ($\sigma^2$) | 数据本身相对 $\mathbf x$ 的随机性，任何模型都消不掉的误差地板（第 2 章）。 |
| 没有免费的午餐定理 | No free lunch theorem | 在所有问题上平均任两算法等价，意味着泛化必须靠归纳偏置（第 1、5 章）。 |
| 归纳偏置 | Inductive bias | 数据之外对"什么解更可能对"的假设，泛化的前提（暗线三）（第 1、5、8 章）。 |
| 表示 / 假设空间 | Representation / hypothesis space | 三件套之一：允许模型长什么样的函数集合，编码归纳偏置（第 1、2 章）。 |
| 目标 / 损失 | Objective / loss | 三件套之一：衡量"拟合得好不好"的标准（第 1、2 章）。 |
| 优化 / 推断 | Optimization / inference | 三件套之一：在假设空间里找目标最优模型的算法（第 1、4 章）。 |
| 不变性 | Invariance | 输入做某变换、输出不变（图平移仍判猫）；标量量该不变（第 1、8 章）。 |
| 等变性 | Equivariance | 输入做某变换、输出做相应变换（卷积特征随输入平移）（第 1、8 章）。 |
| 分布偏移 / 分布漂移 | Distribution shift | 未来数据分布变了，违反 i.i.d.，泛化保证失效的现实裂缝（第 2、11 章）。 |
| 数据生成分布 | Data-generating distribution ($\mathcal D$) | 定义在 $\mathcal X\times\mathcal Y$ 上、未知固定客观的联合分布，沉默的"出题人"（第 2 章）。 |
| 独立同分布 | i.i.d. | 样本独立且同分布，大数定律 / 一切泛化保证的命根子（第 2 章）。 |
| 有效样本数 | Effective sample size | 名义样本里真正独立的数目；相关样本越多边际信息越少（第 2 章）。 |
| 期望风险 / 真实风险 | Expected / true risk ($R(f)$) | 损失在 $\mathcal D$ 上的期望，真正想最小化但算不出的量（第 2 章）。 |
| 经验风险 | Empirical risk ($\hat R_S(f)$) | 训练集上的平均损失，能算但只是期望风险的估计（第 2 章）。 |
| 经验风险最小化 | Empirical risk minimization (ERM) | 用经验风险代替期望风险、在假设空间内最小化的核心原则（第 2 章）。 |
| 损失函数 | Loss function ($\ell$) | 把"错得多严重"编码成可优化标量，三件套"目标"的原子（第 2 章）。 |
| 平方损失 | Squared loss / L2 loss | $(\hat y-y)^2$，处处可导、给条件均值、对离群点敏感（第 2、3 章）。 |
| 绝对损失 | Absolute loss / L1 loss | $\lvert\hat y-y\rvert$，给条件中位数、对离群点稳健、零点不可导（第 2 章）。 |
| Huber 损失 | Huber loss | 误差小用平方、大用线性，光滑兼稳健的折中（第 2、3 章）。 |
| 分位数损失 | Quantile / pinball loss | 让回归预测条件 $\tau$ 分位数，可做预测区间（第 2 章）。 |
| 0-1 损失 | Zero-one loss | 错分率的损失化身，几乎处处梯度为零不可优化，故需替代损失（第 2 章）。 |
| 交叉熵损失 / 对数损失 | Cross-entropy / log loss | 真类负对数概率 $-\log\hat p_y$，0-1 损失的可导凸替代 = 负对数似然（第 2、3 章）。 |
| 替代损失 | Surrogate loss | 替代不可优化 0-1 损失的可导凸上界（交叉熵 / 合页）（第 2 章）。 |
| 合页损失 | Hinge loss | $\max(0,1-y\cdot s)$，SVM 用，分对且够远则零损失（第 2、5 章）。 |
| 大数定律 | Law of large numbers | 样本平均收敛到期望，ERM 大体可行的根据（第 2 章）。 |
| 一致收敛 | Uniform convergence | 对假设空间所有 $f$ 同时 $\hat R_S\approx R$，比逐点大数定律强、依赖容量（第 2 章）。 |
| 泛化界 | Generalization bound | 把"复杂度↑、数据需求↑、$1/\sqrt n$ 衰减"写成的不等式（第 2 章）。 |
| 集中不等式 | Concentration inequality | 如 Hoeffding，把样本均值偏离期望的概率压成指数小（第 2 章）。 |
| 并集界 | Union bound | 把多个"坏事件"概率相加，泛化界里复杂度惩罚的来源（第 2 章）。 |
| VC 维 | Vapnik–Chervonenkis dimension | 假设类能打散的最大点数，无限假设类的有效复杂度度量（第 2 章）。 |
| 打散 | Shatter | 假设类能实现一组点上全部 $2^m$ 种标签组合（第 2 章）。 |
| Rademacher 复杂度 | Rademacher complexity | 假设类拟合随机噪声标签的平均能力，现代泛化界主力（第 2 章）。 |
| 协变量漂移 | Covariate shift | $P(\mathbf x)$ 变而 $P(y\mid\mathbf x)$ 不变，原则上可重要性加权纠偏（第 2、11 章）。 |
| 标签 / 先验漂移 | Label / prior shift | $P(y)$ 变，常可调阈值 / 重标定解决（第 2、11 章）。 |
| 概念漂移 | Concept shift / drift | $P(y\mid\mathbf x)$ 本身变，最难救、需持续重训（第 2、11 章）。 |
| 双下降 | Double descent | 越过插值阈值后测试误差二次下降，推翻"参数多=过拟合"（第 2、9 章）。 |
| 插值阈值 | Interpolation threshold | 容量恰好能记住训练集（参数≈样本）处，测试误差尖峰（第 2、9 章）。 |
| 过参数化 | Overparameterization | 参数远多于样本，能完美插值但仍泛化好的现代现象（第 2、9 章）。 |
| 隐式正则 | Implicit regularization | 优化算法自带的偏好（如 SGD 偏好平坦 / 最小范数解）起正则作用（第 2、4、9 章）。 |
| 奥卡姆剃刀 | Occam's razor | 同等解释力下偏好更简单的假设，正则化的精神（第 2 章）。 |
| 最小描述长度 | Minimum description length (MDL) | 奥卡姆的信息论版：模型+残差总码长最短者最优（第 2、6 章）。 |
| 超参数 | Hyperparameter | 不由训练梯度学、控制模型族 / 学习过程的旋钮（学习率 / $\lambda$），用验证集选（第 2、9 章）。 |
| 参数 | Parameters | 模型从数据里学出来的量（权重 $W$、偏置 $b$）（第 2、3 章）。 |
| 训练 / 验证 / 测试集 | Training / validation / test set | 三分各司其职：拟合参数 / 选超参 / 一次性报泛化，互不串用（第 2 章）。 |
| 验证集 | Validation set | 调超参、选模型用，承担被选择污染的角色（第 2 章）。 |
| k 折交叉验证 | k-fold cross-validation | 轮流用 1 折验证、其余训练，数据少时更稳的超参选择（第 2 章）。 |
| 留一交叉验证 | Leave-one-out (LOOCV) | $k=n$ 的交叉验证，近无偏但方差大且贵（第 2 章）。 |
| 分层划分 | Stratified split | 保证各折 / 各份类别比例一致的划分（第 2 章）。 |
| 分组交叉验证 | Group k-fold | 同实体样本不跨折，防分组泄漏（第 2 章）。 |
| 数据泄漏 | Data leakage | 训练偷用了部署时拿不到的信息（未来 / 标签 / 划分前预处理），离线虚高（第 2、11 章）。 |
| 适应性过拟合 | Adaptive overfitting | 反复在测试集上看一眼调整，慢性把测试集变成验证集（第 2 章）。 |
| 判别式模型 | Discriminative model | 直接建模 $P(y\mid\mathbf x)$ 或决策边界（逻辑回归 / 神经网络）（第 2、5、10 章）。 |
| 生成式模型 | Generative model | 建模 $P(\mathbf x,y)$ 或 $P(\mathbf x)$ 再反推（朴素贝叶斯 / VAE / 扩散）（第 2、5、10 章）。 |
| 贝叶斯最优分类器 | Bayes optimal classifier | 取后验最大类，达到最低期望错误率（贝叶斯误差）（第 2 章）。 |
| 贝叶斯误差 | Bayes error / risk | 分类版不可约误差，由问题本身决定的天花板（第 2 章）。 |
| 概率视角 | Probabilistic view | 学习 = 在数据下做（近似）极大似然 / 贝叶斯推断，贯穿全课的镜头（第 1、2、3 章）。 |

---

## 2. 线性模型与正则化

| 中文名 | English | 解释 |
|---|---|---|
| 线性回归 | Linear regression | 假设 $\hat y=\mathbf w^\top\mathbf x+b$ + 平方损失 + 闭式解，一切线性层的原子（第 1、3 章）。 |
| 权重 | Weight ($\mathbf w$) | 线性模型里各特征的系数，可解释为特征的影响强度（第 3 章）。 |
| 偏置 / 截距 | Bias / intercept ($b$) | 线性模型的常数项，可吸收进增广权重，通常不被正则（第 3 章）。 |
| 仿射函数 | Affine function | 线性变换加平移 $\mathbf w^\top\mathbf x+b$，线性模型的假设空间（第 3、7 章）。 |
| 设计矩阵 | Design matrix ($X$) | 每行一个样本特征堆成的矩阵，最小二乘的输入（第 3 章）。 |
| 最小二乘 | Least squares | 最小化平方残差，几何上把 $\mathbf y$ 投影到特征列空间（第 3 章）。 |
| 正规方程 | Normal equations | $X^\top X\mathbf w=X^\top\mathbf y$，最小二乘闭式解、实战用 QR/SVD 代替（第 3、4 章）。 |
| 闭式解 | Closed-form solution | 一个公式直接算出最优、无需迭代，机器学习里的奢侈品（第 3、4 章）。 |
| 残差 | Residual | $\mathbf y-\hat{\mathbf y}$，最优最小二乘时正交于所有特征（第 3 章）。 |
| 帽子矩阵 / 投影矩阵 | Hat / projection matrix ($H$) | $X(X^\top X)^{-1}X^\top$，把 $\mathbf y$ 投影到列空间，幂等对称（第 3 章）。 |
| 凸函数 | Convex function | 碗状向上、局部最优即全局最优，Hessian 半正定（第 3、4 章）。 |
| 极大似然估计 | Maximum likelihood estimation (MLE) | 找让观测数据最可能的参数；平方损失 = 高斯噪声 MLE（第 2、3 章）。 |
| 高斯噪声 | Gaussian noise | $y=\mathbf w^\top\mathbf x+\varepsilon,\ \varepsilon\sim\mathcal N(0,\sigma^2)$，平方损失背后的假设（第 3 章）。 |
| 高斯-马尔可夫定理 | Gauss–Markov theorem | 噪声零均值同方差不相关时，最小二乘是最优线性无偏估计（BLUE）（第 3 章）。 |
| 无偏估计 | Unbiased estimator | 期望等于真值的估计；普通最小二乘无偏、岭回归故意有偏（第 3 章）。 |
| 异方差 | Heteroscedasticity | 噪声方差随样本变化，需加权最小二乘（第 3 章）。 |
| 加权最小二乘 | Weighted least squares (WLS) | 按 $1/\sigma_i^2$ 给样本加权，异方差下的高斯 MLE（第 3 章）。 |
| 正则化 | Regularization | 任何为减小泛化差距而非训练误差的修改，惩罚复杂度 = 注入先验（第 2、3、9 章）。 |
| 岭回归 | Ridge regression | 加 L2 惩罚 $\lambda\lVert\mathbf w\rVert_2^2$，闭式解 $(X^\top X+\lambda I)^{-1}X^\top\mathbf y$，治病态（第 3 章）。 |
| L2 正则 / 权重衰减 | L2 regularization / weight decay | 惩罚权重平方 = 高斯先验 MAP = 温柔收缩、偏好光滑（第 3、9 章）。 |
| L1 正则 / Lasso | L1 regularization / Lasso | 惩罚权重绝对值 = 拉普拉斯先验 MAP，尖角产生稀疏与特征选择（第 3 章）。 |
| 稀疏解 | Sparse solution | 很多权重恰为零，L1 / 拉普拉斯先验的产物，兼做特征选择（第 3 章）。 |
| 次梯度 | Subgradient | 不可导点（如 $\lvert w\rvert$ 在 0）的广义梯度，解释 L1 稀疏的机制（第 3 章）。 |
| 弹性网络 | Elastic net | L1+L2 混合，兼得稀疏与对相关特征的稳定（第 3 章）。 |
| 最大后验估计 | Maximum a posteriori (MAP) | 最大化似然×先验，L2↔高斯先验、L1↔拉普拉斯先验（第 2、3 章）。 |
| 正则强度 | Regularization strength ($\lambda$) | 调容量 / 偏差-方差的旋钮 = 噪声方差与先验方差之比（第 3 章）。 |
| 基函数 | Basis function | 先把 $\mathbf x$ 做非线性映射 $\boldsymbol\phi(\mathbf x)$ 再线性组合，让线性模型拟合非线性（第 3 章）。 |
| 多项式回归 | Polynomial regression | 取多项式基的线性模型，阶数是容量旋钮（第 3 章）。 |
| 龙格现象 | Runge phenomenon | 高阶多项式在区间边界剧烈振荡、外推危险（第 3 章）。 |
| 径向基函数 | Radial basis function (RBF) | 高斯"鼓包"基 $\exp(-\lVert\mathbf x-\mathbf c\rVert^2/2s^2)$，表达局部结构（第 3、5 章）。 |
| 样条 | Spline | 分段低次多项式基，兼顾光滑与局部、避高阶振荡（第 3 章）。 |
| 特征工程 | Feature engineering | 人手设计非线性 / 交互特征，表格数据上常比换模型有效（第 3、5、11 章）。 |
| 特征缩放 / 标准化 | Feature scaling / standardization | 各特征减均值除标准差，正则语义与优化收敛都靠它（统计量只从训练集算）（第 3、4 章）。 |
| 逻辑回归 | Logistic regression | 线性打分经 sigmoid 压成概率 + 交叉熵，决策边界是超平面（线性分类器）（第 3 章）。 |
| sigmoid 函数 | Sigmoid / logistic function | $1/(1+e^{-z})$ 把实数压到 $(0,1)$，导数 $\sigma(1-\sigma)$（第 3、7 章）。 |
| 对数几率 / logit | Log-odds / logit | $\log\frac{\hat p}{1-\hat p}=\mathbf w^\top\mathbf x+b$，逻辑回归假设它线性（第 3 章）。 |
| 决策边界 | Decision boundary | 分类器把空间分成各类的界面，逻辑回归的是超平面（第 3 章）。 |
| 超平面 | Hyperplane | $\mathbf w^\top\mathbf x+b=0$，线性分类器的平直分界面（第 3 章）。 |
| 线性分类器 | Linear classifier | 决策边界是超平面的分类器（逻辑回归 / 线性 SVM / 朴素贝叶斯）（第 3、5 章）。 |
| 伯努利分布 | Bernoulli distribution | 二分类标签的分布，逻辑回归交叉熵 = 其负对数似然（第 3 章）。 |
| 负对数似然 | Negative log-likelihood (NLL) | 损失的概率身份，交叉熵 = NLL = 极大似然（第 2、3 章）。 |
| softmax 回归 / 多项逻辑回归 | Softmax / multinomial logistic regression | 逻辑回归的多类推广，softmax 归一化成类别分布 + 类别交叉熵（第 3 章）。 |
| softmax 函数 | Softmax function | $e^{z_i}/\sum_k e^{z_k}$ 把打分变概率分布，"温和的 max"、平移不变（第 3、7 章）。 |
| 类别分布 | Categorical distribution | 多分类标签的分布，softmax 回归交叉熵 = 其 MLE（第 3 章）。 |
| 独热编码 | One-hot encoding | 类别编成第 k 位为 1 的向量，又大又稀疏、丢语义相似性（第 3、8 章）。 |
| 最大熵原理 | Maximum entropy principle | 给定约束下最不武断的分布，逼出 sigmoid / softmax（呼应信息论）（第 3 章）。 |
| 广义线性模型 | Generalized linear model (GLM) | 线性打分+指数族分布+链接函数+NLL 的统一配方（第 3 章）。 |
| 链接函数 | Link function | 把线性打分接到分布均值参数（恒等 / sigmoid / softmax）（第 3 章）。 |
| 指数族 | Exponential family | 高斯 / 伯努利 / 类别 / 泊松等一族分布，GLM 的核心（第 3 章）。 |
| 校准 | Calibration | 模型输出的概率是否对应真实频率，现代网络常过度自信（第 3、9 章）。 |
| 温度标定 | Temperature scaling | logits 除以温度 $T$ 再 softmax，事后改善概率校准（第 3 章）。 |
| 期望校准误差 | Expected calibration error (ECE) | 把样本按预测置信度分箱，量"置信度与实际准确率"差距的校准指标（第 3、11 章）。 |
| 感知机 | Perceptron | 最早的线性分类器，可分则收敛、否则震荡，解不了 XOR（第 3 章）。 |
| 最大间隔分类器 | Maximum margin classifier | 选离两类都最远的分界面，间隔大则泛化好，SVM 的核心（第 3、5 章）。 |
| 间隔 | Margin | 分界面到最近样本的距离，最大化它 = 最小化 $\lVert\mathbf w\rVert$ = L2（第 3、5 章）。 |
| 支持向量 | Support vector | 落在间隔边缘、决定边界的少数关键样本（第 3、5 章）。 |

---

## 3. 优化

| 中文名 | English | 解释 |
|---|---|---|
| 迭代优化 | Iterative optimization | 不靠闭式解、只用局部信息一步步逼近最优，机器学习的常态引擎（第 4 章）。 |
| 损失曲面 / 优化地形 | Loss surface / optimization landscape | 参数空间上方的"海拔地形"，学习 = 在上面找低谷（第 4 章）。 |
| 梯度 | Gradient ($\nabla\mathcal L$) | 各偏导组成的向量，指向函数值增长最快的方向（第 4 章）。 |
| 梯度下降 | Gradient descent (GD) | 沿负梯度迈步 $\theta\leftarrow\theta-\eta\nabla\mathcal L$，迭代下山（第 4 章）。 |
| 学习率 | Learning rate ($\eta$) | 每步迈多大，最重要最难调的超参，太大发散太小太慢（第 4 章）。 |
| 全批量梯度下降 | Full-batch gradient descent | 每步用全部样本算精确梯度，慢且占显存（第 4 章）。 |
| 随机梯度下降 | Stochastic gradient descent (SGD) | 用一小撮样本的无偏梯度估计更新，深度学习的命脉（第 4 章）。 |
| 小批量 | Mini-batch | 每步用的一小撮样本（如 32~1024），平衡噪声与硬件（第 4 章）。 |
| 批大小 | Batch size ($B$) | 小批量样本数，与学习率耦合、与泛化有张力（第 4、9 章）。 |
| epoch | Epoch / 轮次 | 把全部训练数据过一遍（第 4、7 章）。 |
| 计算 vs 统计权衡 | Computation vs statistics tradeoff | SGD 每步"糙但省"换全批量"准但贵"，固定预算下更快下山（第 4 章）。 |
| 平坦极小 | Flat minimum | 宽阔的谷底，参数扰动损失变化小，常泛化更好（SGD 偏好）（第 4、9 章）。 |
| 凸优化 | Convex optimization | 凸问题局部最优即全局，梯度下降有全局收敛保证（第 4 章）。 |
| 非凸 | Non-convex | 神经网络损失，地形坑洼、不保证全局最优（第 4 章）。 |
| 局部极小 | Local minimum | 四周都高但非全局最低，高维深网里其实不是主要障碍（第 4 章）。 |
| 鞍点 | Saddle point | 某些方向极小、某些极大，高维里压倒性多、是主要减速来源（第 4 章）。 |
| 平台 / 高原 | Plateau / flat region | 梯度近零的大片区域，优化几乎停滞（第 4 章）。 |
| 驻点 | Stationary point | 梯度为零点，可能是极小 / 极大 / 鞍点（第 4 章）。 |
| 优化误差 | Optimization error | 没把经验风险真优化到底带来的误差，ERM 之外的第三重（第 2、4 章）。 |
| 动量 | Momentum | 累积历史梯度成速度，沿峡谷加速、冲过小坑（重球类比）（第 4 章）。 |
| Nesterov 加速梯度 | Nesterov accelerated gradient (NAG) | 动量的"先前瞻再算梯度"变体，凸情形 $O(1/t^2)$ 加速（第 4 章）。 |
| AdaGrad | AdaGrad | 累积历史梯度平方做逐参数自适应步长，后期衰减过头（第 4 章）。 |
| RMSProp | RMSProp | 用梯度平方的指数移动平均，修掉 AdaGrad 衰减过头（第 4 章）。 |
| Adam | Adam (Adaptive Moment Estimation) | 动量+RMSProp 合体，常用默认优化器但非永远最优（第 4 章）。 |
| AdamW | AdamW | 把权重衰减与自适应缩放解耦的 Adam 变体，做法更干净（第 4、9 章）。 |
| 偏差校正 | Bias correction | Adam 中修正一二阶矩初值为零的偏差（第 4 章）。 |
| 学习率调度 | Learning rate schedule | 让 $\eta$ 随训练变化（衰减 / 预热）（第 4、9 章）。 |
| 预热 | Warmup | 训练初期把 $\eta$ 从小线性升到目标值，稳住早期、大模型必备（第 4、9 章）。 |
| 余弦退火 | Cosine annealing | $\eta$ 沿余弦曲线平滑降到接近零，后期精细收敛（第 4、9 章）。 |
| 学习率衰减 | Learning rate decay | 后期调小 $\eta$ 让 SGD 噪声球收缩、沉进谷底（第 4 章）。 |
| Robbins–Monro 条件 | Robbins–Monro condition | $\sum\eta_t=\infty,\sum\eta_t^2<\infty$ 保证 SGD 收敛的步长条件（第 4 章）。 |
| 条件数 | Condition number ($\kappa$) | Hessian 最大与最小特征值之比，大则病态、梯度下降之字形慢（第 4、9 章）。 |
| 病态 | Ill-conditioned | 损失曲面被拉成又长又窄峡谷，逼小学习率、收敛慢（第 4 章）。 |
| Hessian / 黑塞矩阵 | Hessian | 二阶偏导对称矩阵，描述损失曲面曲率，特征值=各主轴曲率（第 4 章）。 |
| 之字形 | Zig-zag | 病态曲面上梯度下降在峡谷两壁来回横跳的轨迹（第 4 章）。 |
| 预条件 / 预处理 | Preconditioning | 换坐标使 Hessian 接近单位阵、降条件数；自适应优化器是廉价对角版（第 4 章）。 |
| 白化 | Whitening | 用 PCA/ZCA 去相关再缩放使协方差近单位阵，治尺度+相关病态（第 4、6 章）。 |
| 牛顿法 | Newton's method | 更新乘 Hessian 逆，对条件数不敏感、收敛极快但深度学习不可行（第 4 章）。 |
| 拟牛顿法 | Quasi-Newton (BFGS / L-BFGS) | 用梯度差近似 Hessian 逆，中小光滑问题高效（第 4 章）。 |
| 一阶 / 二阶方法 | First / second-order method | 用梯度 / 用曲率（Hessian）的优化，深度学习主流是一阶+对角近似（第 4 章）。 |
| 自然梯度 | Natural gradient | 用 Fisher 度量定义最陡方向的优化，结构化近似二阶之一（第 4 章）。 |
| 梯度检查 | Gradient checking | 用中心差分数值梯度核对解析梯度，只用于调试（第 4、7 章）。 |
| 中心差分 | Central difference | $\frac{\mathcal L(\theta+\epsilon e)-\mathcal L(\theta-\epsilon e)}{2\epsilon}$，误差 $O(\epsilon^2)$ 的数值梯度（第 4 章）。 |
| 学习率范围测试 | LR range test | $\eta$ 指数增大画 loss 曲线，取陡降段略小于发散点（第 4 章）。 |
| 梯度裁剪 | Gradient clipping | 梯度范数超阈值就按比例缩回，挡偶发爆炸，序列模型 / RL 常备（第 4、9 章）。 |
| 数值稳定的损失 | Numerically stable loss | 用 log-sum-exp 等技巧防 softmax 溢出 / $\log 0$（第 4 章）。 |

---

## 4. 经典机器学习方法

| 中文名 | English | 解释 |
|---|---|---|
| 基线 | Baseline | 任何项目应先跑的简单参照模型，给"相对提升"一个分母（第 5、11 章）。 |
| k 近邻 | k-nearest neighbors (kNN) | 找最近 $k$ 个邻居投票 / 取平均，非参数、懒惰学习（第 5 章）。 |
| 非参数方法 | Non-parametric method | 不假设固定函数形式、复杂度随数据增长（kNN）（第 5 章）。 |
| 懒惰学习 | Lazy learning | 训练几乎不做事、把计算推迟到预测时（kNN）（第 5 章）。 |
| 急切学习 | Eager learning | 训练时辛苦优化、预测时一次前向（线性 / 神经网络）（第 5 章）。 |
| 距离度量 | Distance metric | kNN 里定义"什么叫近"的函数，决定一切（欧氏 / 曼哈顿 / 余弦 / 马氏）（第 5 章）。 |
| 马氏距离 | Mahalanobis distance | 用协方差逆"拉正"各方向尺度的距离 = 先白化再算欧氏（第 5 章）。 |
| 度量学习 | Metric learning | 从数据学一个最好的距离 / 嵌入，让同类近异类远（第 5 章）。 |
| 距离加权投票 | Distance-weighted voting | 近邻按距离倒数加权，让近的说话更响，连向 RBF 核（第 5 章）。 |
| Voronoi 图 | Voronoi diagram | 1-NN 把空间按"离哪个训练点最近"切成的拼块，决策边界（第 5 章）。 |
| 维数灾难 | Curse of dimensionality | 高维下点彼此近乎等距、"最近邻"失意义，kNN 失效根源（第 5、6 章）。 |
| 流形假设 | Manifold hypothesis | 高维真实数据常集中在低维弯曲流形上，降维 / kNN 复活的根据（第 5、6 章）。 |
| 近似最近邻 | Approximate nearest neighbor (ANN) | 用 HNSW / LSH 等在高维高效近似检索，向量数据库 / RAG 召回核心（第 5 章）。 |
| 决策树 | Decision tree | 递归把特征空间切成轴对齐盒子、给分段常数，可解释（第 5 章）。 |
| 分段常数函数 | Piecewise-constant function | 决策树表达的函数形态，边界与坐标轴平行（第 5 章）。 |
| 信息增益 | Information gain | 父节点熵减子节点熵加权平均 = 特征与标签互信息，分裂准则（第 5 章）。 |
| 基尼不纯度 | Gini impurity | $1-\sum p_k^2$，CART 的分裂准则，熵的低阶近似（第 5 章）。 |
| 剪枝 | Pruning | 限制深度 / 剪掉弱分支控制树容量 = 正则化（预 / 后剪枝）（第 5 章）。 |
| 代价复杂度剪枝 | Cost-complexity pruning | 给目标加叶子数惩罚 $\alpha$，本质是正则化（第 5 章）。 |
| 特征重要性 | Feature importance | 树 / 森林给出的特征贡献度，偏向高基数特征、≠ 因果（第 5 章）。 |
| 置换重要性 | Permutation importance | 打乱某特征看性能下降多少，比不纯度重要性更稳健（第 5 章）。 |
| 集成学习 | Ensemble learning | 把许多"还行"的模型组合成强模型（bagging / boosting / stacking）（第 5 章）。 |
| Bagging | Bagging (bootstrap aggregating) | 自助重采样训多个高方差模型再平均，纯降方差（第 5 章）。 |
| Bootstrap 重采样 | Bootstrap resampling | 有放回抽样造略不同的训练集，bagging 的基础（第 5 章）。 |
| 袋外样本 / 误差 | Out-of-bag (OOB) | 每棵树没抽到的约 37% 样本，可做免费验证估计（第 5 章）。 |
| 随机森林 | Random forest | bagging+每次分裂随机选特征降树间相关，开箱即用、鲁棒（第 5 章）。 |
| Boosting | Boosting | 串行训弱模型逐步纠错、降偏差（第 5 章）。 |
| 梯度提升 | Gradient boosting | 新弱学习器拟合负梯度（伪残差）= 函数空间里的梯度下降（第 5 章）。 |
| 伪残差 | Pseudo-residual | 损失对当前模型输出的负梯度，新树要拟合的目标（第 5 章）。 |
| 梯度提升决策树 | Gradient-boosted decision trees (GBDT) | 用浅树做弱学习器的梯度提升，表格数据之王（XGBoost/LightGBM）（第 5、11 章）。 |
| 早停 | Early stopping | 验证误差不再降就停、回退最优 checkpoint，≈ 隐式 L2（第 5、9 章）。 |
| 孔多塞陪审团定理 | Condorcet's jury theorem | 多个独立、略好于随机的分类器多数投票，集体正确率趋于 1（第 5 章）。 |
| 支持向量机 | Support vector machine (SVM) | 最大间隔+核技巧的分类器，解只由支持向量定（第 5 章）。 |
| 软间隔 | Soft margin | 允许少量样本越界、付松弛代价 = hinge 损失 + L2 正则（第 5 章）。 |
| 凸二次规划 | Convex quadratic programming (QP) | SVM 原问题的形式，有唯一全局最优（第 5 章）。 |
| 对偶问题 | Dual problem | SVM 的等价形式，数据只以两两内积出现，核技巧的入口（第 5 章）。 |
| 核技巧 | Kernel trick | 用核函数隐式在高维做内积而不显式映射，让线性方法非线性化（第 5 章）。 |
| 核函数 | Kernel function ($K$) | $\langle\phi(\mathbf x),\phi(\mathbf z)\rangle$，某特征空间的内积（线性 / 多项式 / RBF）（第 5 章）。 |
| Mercer 定理 | Mercer's theorem | 对称半正定函数必对应某特征空间内积、可合法当核（第 5 章）。 |
| 朴素贝叶斯 | Naive Bayes | 生成式分类器，假设给定类别下特征条件独立，极快、抗高维（第 5 章）。 |
| 条件独立 | Conditional independence | 给定类别后特征互不相关，朴素贝叶斯的"朴素"假设（第 5 章）。 |
| 拉普拉斯平滑 | Laplace / additive smoothing | 给计数加小常数，防朴素贝叶斯零概率一票否决（第 5 章）。 |
| 表格数据 | Tabular data | 行是样本、列是独立语义特征的结构化数据，GBDT 主场（第 5、11 章）。 |

---

## 5. 无监督学习与降维

| 中文名 | English | 解释 |
|---|---|---|
| 聚类 | Clustering | 把相似样本分到同一组的无监督任务（第 6 章）。 |
| 降维 | Dimensionality reduction | 把高维数据压到低维同时保留信息 / 结构（第 6 章）。 |
| 密度估计 | Density estimation | 估计数据分布 $p(\mathbf x)$ 本身，异常检测 / 采样的底座（第 6 章）。 |
| 表示学习 | Representation learning | 学一种让下游任务变简单的数据表示，深度学习核心优势（第 6、8 章）。 |
| 内部 / 外部指标 | Internal / external metrics | 聚类评估：只用数据本身 / 借外部标签事后验证（轮廓系数 / ARI）（第 6 章）。 |
| 轮廓系数 | Silhouette | 比较簇内与最近邻簇距离的聚类内部指标，$\in[-1,1]$（第 6 章）。 |
| K-means | K-means | 最小化簇内平方和，预设 $k$、球状簇假设（第 6 章）。 |
| 簇内平方和 / 惯性 | Within-cluster sum of squares (WCSS) / inertia | 各点到所属簇心距离平方和，K-means 目标（第 6 章）。 |
| 簇心 / 质心 | Centroid | 簇内点的均值，K-means 的代表点（第 6 章）。 |
| Lloyd 算法 | Lloyd's algorithm | K-means 的交替最小化（最近邻指派+均值更新），单调下降到局部最优（第 6 章）。 |
| 交替最小化 / 坐标下降 | Alternating minimization / coordinate descent | 固定一组变量优化另一组、来回交替（Lloyd / EM）（第 6 章）。 |
| k-means++ | k-means++ | 让初始簇心彼此散开的聪明播种法，缓解初始化敏感（第 6 章）。 |
| 肘部法则 | Elbow method | 看 WCSS-k 曲线拐点选簇数 $k$ 的启发式（第 6 章）。 |
| 矢量量化 | Vector quantization | 用 $k$ 个簇心（码本）近似代表数据集，K-means 的压缩身份（第 6 章）。 |
| 硬指派 / 软指派 | Hard / soft assignment | 非此即彼归一簇 / 以概率属于各簇（K-means vs GMM）（第 6 章）。 |
| 层次聚类 | Hierarchical clustering | 自底向上凝聚建簇的树，不预设 $k$、给多粒度结构（第 6 章）。 |
| 连接方式 | Linkage | 层次聚类里簇间距离的定义（单 / 全 / 平均 / Ward）（第 6 章）。 |
| DBSCAN | DBSCAN | 基于密度的聚类，簇=高密度区，能发现任意形状、识别离群点（第 6 章）。 |
| 高斯混合模型 | Gaussian mixture model (GMM) | $k$ 个多元高斯的概率混合，软指派、拟合椭球簇、是密度模型（第 6 章）。 |
| 混合权重 | Mixing weights ($\pi_j$) | GMM 里选各分量的先验概率（第 6 章）。 |
| 隐变量 | Latent variable | 未观测的内在变量（GMM 的"来自哪簇"、VAE 的 $\mathbf z$）（第 6、10 章）。 |
| EM 算法 | Expectation-Maximization (EM) | 含隐变量极大似然的通用框架：E 步算后验、M 步加权 MLE（第 6 章）。 |
| 责任 | Responsibility ($\gamma_{ij}$) | E 步算的点属于各簇的后验概率（软指派）（第 6 章）。 |
| 证据下界 | Evidence Lower BOund (ELBO) | 对数似然的下界，EM / VAE 通过抬高它来优化（第 6、10 章）。 |
| 后验坍缩 | Posterior collapse | 解码器过强时模型忽略隐变量、$q$ 退化成先验（第 6、10 章）。 |
| 主成分分析 | Principal component analysis (PCA) | 投影到方差最大 = 重构误差最小的正交方向 = 协方差特征分解 = SVD（第 6 章）。 |
| 最大方差 | Maximum variance | PCA 的一种刻画：找让投影方差最大的方向（第 6 章）。 |
| 最小重构误差 | Minimum reconstruction error | PCA 的等价刻画：找重建数据丢失最少的子空间（第 6 章）。 |
| 主成分 | Principal component | 数据方差最大的正交方向 = 协方差特征向量 = 右奇异向量（第 6 章）。 |
| 中心化 | Centering | PCA 第一步：每维减均值（第 6 章）。 |
| 解释方差比 | Explained variance ratio | 各主成分捕获的总方差比例，选保留维数用（第 6 章）。 |
| 碎石图 | Scree plot | 特征值随序号衰减曲线，找拐点定主成分个数（第 6 章）。 |
| 概率 PCA | Probabilistic PCA (PPCA) | PCA 的生成模型化身，PCA 是其零噪声极限（连向 VAE）（第 6 章）。 |
| 线性判别分析 | Linear discriminant analysis (LDA) | 用类别信息找最能分开类别的方向，PCA 的监督对应物（第 6 章）。 |
| 非线性降维 | Nonlinear dimensionality reduction | 能展开弯曲流形的降维（t-SNE / UMAP / 自编码器）（第 6 章）。 |
| t-SNE | t-distributed Stochastic Neighbor Embedding | 保局部邻域、最小化邻居概率 KL 的可视化降维，簇间距离不可读（第 6 章）。 |
| 困惑度 (t-SNE) | Perplexity | t-SNE 自适应带宽超参，约等于每点有效邻居数（第 6 章）。 |
| UMAP | UMAP | 基于流形 / 拓扑的非线性降维，比 t-SNE 快、保更多全局结构（第 6 章）。 |
| 自编码器 | Autoencoder | 编码器压到瓶颈 + 解码器重建、最小化重构误差的非线性降维（第 6 章）。 |
| 瓶颈 | Bottleneck | 自编码器中维度远小于输入的潜码层，逼模型抓本质结构（第 6 章）。 |
| 信息瓶颈 | Information bottleneck | 限制表示携带的信息量、只留有用部分，瓶颈 / VAE 的 KL 项（第 6、10 章）。 |
| 去噪自编码器 | Denoising autoencoder (DAE) | 破坏输入、重建干净，逼模型学真结构，自监督雏形（第 6 章）。 |
| 稀疏自编码器 | Sparse autoencoder | 罚隐层激活不稀疏，逼出可解释 / 解耦特征（第 6 章）。 |
| 解耦表示 | Disentangled representation | 不同维对应不同独立变化因子，纯无监督下一般无法保证（第 6 章）。 |
| 直方图 | Histogram | 最朴素的密度估计，对 bin 敏感、不连续、高维爆炸（第 6 章）。 |
| 核密度估计 | Kernel density estimation (KDE) | 每数据点放核相加得光滑密度，带宽是偏差-方差旋钮、只适合低维（第 6 章）。 |
| 带宽 | Bandwidth ($h$) | KDE 里核的宽窄，小则过拟合尖刺、大则糊平（第 6 章）。 |
| 异常检测 | Anomaly detection | 把 $p(\mathbf x)$ 低（似然低）的样本判为异常（第 6、11 章）。 |
| 分布假说 | Distributional hypothesis | 一个词的意义由它出现的上下文决定，词嵌入的根据（第 6、8 章）。 |

---

## 6. 神经网络基础

| 中文名 | English | 解释 |
|---|---|---|
| 神经网络 | Neural network | "线性变换+非线性激活"交替堆叠，自动学层层递进特征（第 7 章）。 |
| 深度学习 | Deep learning (DL) | 用深层神经网络从原始数据端到端学表示的机器学习（第 1、7 章）。 |
| 塌缩 | Collapse | 纯线性叠多层精确等于一层（$W_2W_1$），层数白加（第 7 章）。 |
| 激活函数 | Activation function ($\phi$) | 逐元素非线性，网络唯一非线性来源、打破塌缩（第 7 章）。 |
| 多层感知机 | Multilayer perceptron (MLP) | 全连接前馈网络，所有深度架构的最小公共子集（第 7 章）。 |
| 神经元 | Neuron | 加权和+偏置+激活 = 一个广义线性单元（第 7 章）。 |
| 权重矩阵 | Weight matrix ($W$) | 一层神经元权重堆成的矩阵，$W\mathbf x+\mathbf b$（第 7 章）。 |
| 隐藏层 | Hidden layer | 既非输入也非输出的中间层，模型自学的表示（第 7 章）。 |
| 宽度 / 深度 | Width / depth | 一层的单元数 / 网络层数（第 7 章）。 |
| 前向传播 | Forward pass / propagation | 从输入一层层算到输出，一串矩阵乘+激活（第 7 章）。 |
| 预激活 / 激活 | Pre-activation / activation | 线性变换结果 $\mathbf z$ / 过非线性后 $\mathbf a$，反向传播要区分（第 7 章）。 |
| tanh | Hyperbolic tangent | 把实数压到 $(-1,1)$ 的激活，零中心但仍饱和（第 7 章）。 |
| 饱和 | Saturation | sigmoid/tanh 两端导数趋零，导致梯度消失（第 7 章）。 |
| ReLU | Rectified linear unit | $\max(0,z)$，正半轴不饱和、计算廉、现代默认激活（第 7 章）。 |
| 死亡 ReLU | Dying ReLU | 神经元恒输出 0、梯度恒零、永久死亡（第 7 章）。 |
| LeakyReLU | LeakyReLU | 负半轴给小斜率 $\alpha$，让死神经元仍有微弱梯度（第 7 章）。 |
| GELU | Gaussian error linear unit | $z\Phi(z)$，ReLU 的平滑版，Transformer 事实标准激活（第 7 章）。 |
| SiLU / Swish | SiLU / Swish | $z\sigma(z)$，平滑非单调激活，性能近 GELU（第 7 章）。 |
| 通用逼近定理 | Universal approximation theorem | 足够宽的单隐层 MLP 能逼近紧集上任意连续函数（第 7 章）。 |
| 存在性定理 | Existence theorem | 只承诺"存在"好网络，不保证可学习 / 不告诉多大 / 不保证泛化（第 7 章）。 |
| 可学习性 | Learnability | 优化器能否真找到那个好解，与"能逼近"是两件事（第 7、9 章）。 |
| 分段线性函数 | Piecewise-linear function | ReLU 网络把空间剖成多面体、每块仿射的函数形态（第 7 章）。 |
| 组合性 | Compositionality | 数据有层次结构（边缘→部件→物体），深度的归纳偏置（第 7、8 章）。 |
| 反向传播 | Backpropagation (backprop) | 链式法则在计算图上的系统化高效组织，算精确梯度（第 4、7 章）。 |
| 链式法则 | Chain rule | 复合函数导数是各层导数乘积，反向传播的数学核心（第 4、7 章）。 |
| 误差信号 / 灵敏度 | Error signal / sensitivity ($\boldsymbol\delta$) | $\partial\mathcal L/\partial\mathbf z^{(\ell)}$，从输出端逆流而上的反传量（第 7 章）。 |
| 计算图 | Computational graph | 节点是中间变量、边是基本运算的有向图，自动微分的载体（第 7 章）。 |
| 自动微分 | Automatic differentiation (autodiff) | 对任意计算图自动求精确导数的技术（第 7 章）。 |
| 反向模式自动微分 | Reverse-mode autodiff | 正向缓存、反向传梯度，标量输出对海量输入求导的高效模式 = 反向传播（第 7 章）。 |
| 向量-雅可比积 | Vector-Jacobian product (VJP) | 把流入梯度乘节点局部雅可比传给上游，反向模式的原子操作（第 7 章）。 |
| 雅可比矩阵 | Jacobian matrix | 多元映射的一阶导矩阵，反向传播是雅可比连乘（第 7、9 章）。 |
| 梯度消失 | Vanishing gradient | 雅可比连乘因子持续 <1，浅层梯度指数衰减、学不动（第 7、8 章）。 |
| 梯度爆炸 | Exploding gradient | 连乘因子持续 >1，梯度指数放大、发散 / NaN（第 7、8 章）。 |
| 初始化 | Initialization | 权重初值；要随机打破对称、并让方差逐层守恒（第 7 章）。 |
| 对称性（初始化） | Symmetry breaking | 全 0 初始化致命，必须随机打破让神经元学不同东西（第 7 章）。 |
| Xavier / Glorot 初始化 | Xavier / Glorot initialization | 方差 $\approx 2/(n_\text{in}+n_\text{out})$，兼顾前后向，配 tanh/sigmoid（第 7 章）。 |
| He / Kaiming 初始化 | He / Kaiming initialization | 方差 $\approx 2/n_\text{in}$，补偿 ReLU 砍半方差，ReLU 默认搭配（第 7 章）。 |
| 方差传播 | Variance propagation | 追踪激活 / 梯度方差如何被层乘法和激活改变，初始化的分析（第 7 章）。 |
| 训练循环 | Training loop | 前向→反向→优化器更新→循环，把全课串成一台机器（第 7 章）。 |
| teacher forcing | Teacher forcing | 训练时喂真实前缀预测下一元素，自回归 / seq2seq 用（第 8、10 章）。 |

---

## 7. 深度架构

| 中文名 | English | 解释 |
|---|---|---|
| 架构即归纳偏置 | Architecture as inductive bias | 用局部连接+权重共享把数据对称性焊进连接模式（本章主旨）（第 8 章）。 |
| 局部连接 | Local connectivity | 隐藏单元只看输入一小块，编码"局部性"先验（第 8 章）。 |
| 权重共享 | Weight sharing / parameter tying | 不同位置用同一套权重，编码对称、省参数、提样本效率（第 8 章）。 |
| 几何深度学习 | Geometric deep learning | 把对称性 / 等变性作为设计原则的统一纲领（第 8 章）。 |
| 卷积神经网络 | Convolutional neural network (CNN) | 为网格数据而生，卷积 = 局部连接+权重共享、平移等变（第 8 章）。 |
| 卷积 | Convolution | 小核沿输入滑动做局部内积，深度学习里实为互相关（第 8 章）。 |
| 互相关 | Cross-correlation | 不翻核的"卷积"，框架里的卷积实为此（第 8 章）。 |
| 卷积核 / 滤波器 | Kernel / filter | 一小段 / 小块可学权重，扫过输入抽局部模式（第 8 章）。 |
| 通道 | Channel | 输入 / 特征图的"深度"维（如 RGB），每个核输出一个通道（第 8 章）。 |
| 特征图 | Feature map | 一个卷积核扫全图得到的输出，对应一种局部模式（第 8 章）。 |
| 步幅 | Stride | 核每次滑动几格，>1 做下采样（第 8 章）。 |
| 填充 | Padding | 边缘补零以控制输出尺寸（same padding）（第 8 章）。 |
| 感受野 | Receptive field | 深层神经元最终"看到"输入多大一块，随深度增长（第 8 章）。 |
| 层级特征 | Hierarchical features | 浅层学边缘、深层学物体，局部连接+深度自然涌现（第 8 章）。 |
| 1×1 卷积 | Pointwise convolution | 逐像素跨通道线性组合，做通道融合 / 升降维（第 8 章）。 |
| 空洞 / 膨胀卷积 | Dilated / atrous convolution | 核元素间插空隙，不增参数地扩大感受野（第 8 章）。 |
| 转置卷积 | Transposed convolution | 用卷积矩阵转置做前向把特征图放大，非真正反卷积（第 8 章）。 |
| 池化 | Pooling | 窗口取最大 / 平均，下采样并带来近似局部平移不变（第 8 章）。 |
| 循环神经网络 | Recurrent neural network (RNN) | 用随时间更新的隐状态处理变长序列，编码时序先验（第 8 章）。 |
| 隐状态 | Hidden state ($\mathbf h_t$) | RNN 随时间更新的"记忆"，压缩到 $t$ 为止的历史（第 8 章）。 |
| 时间反向传播 | Backpropagation through time (BPTT) | 把 RNN 按时间展开成深网再反向传播（第 8 章）。 |
| 长程依赖 | Long-range dependency | 远处元素的影响，朴素 RNN 因梯度消失学不到（第 8 章）。 |
| 长短期记忆 | Long Short-Term Memory (LSTM) | 加性细胞状态+门控让信息近乎无损沿时间流动，缓解梯度消失（第 8 章）。 |
| 细胞状态 | Cell state ($\mathbf c_t$) | LSTM 的加性"信息高速公路"（第 8 章）。 |
| 门控 | Gating | sigmoid 软开关控制信息流（遗忘 / 输入 / 输出门）（第 8 章）。 |
| 门控循环单元 | Gated Recurrent Unit (GRU) | LSTM 的精简版，更少门、参数少、常效果相当（第 8 章）。 |
| 双向 RNN | Bidirectional RNN | 正反两条 RNN 拼接，每位置看双侧上下文（第 8 章）。 |
| 编码器-解码器 / seq2seq | Encoder–decoder / seq2seq | 编码器压源序列成上下文向量、解码器生成目标序列（第 8 章）。 |
| 上下文向量 | Context vector | seq2seq 编码器输出的固定长度表示，是其瓶颈（第 8 章）。 |
| 注意力 | Attention | 按内容相关性动态加权聚合所有位置信息，最初为对齐而生（第 8 章）。 |
| 自注意力 | Self-attention | $\mathrm{softmax}(QK^\top/\sqrt{d_k})V$，序列对自己做加权重组（第 8 章）。 |
| Query / Key / Value | Query / Key / Value (Q/K/V) | token 的三种线性投影角色（想找什么 / 被匹配 / 实际内容）（第 8 章）。 |
| 缩放点积注意力 | Scaled dot-product attention | 除 $\sqrt{d_k}$ 防 softmax 饱和的注意力打分（有方差分析支撑）（第 8 章）。 |
| 注意力权重 | Attention weights ($A$) | 每行 softmax 后的离散概率分布，决定从哪取信息（第 8 章）。 |
| 多头注意力 | Multi-head attention | 切成多头各学一种关系再拼接融合（第 8 章）。 |
| 交叉注意力 | Cross-attention | 解码器 Query 查编码器 Key/Value，回看源序列（第 8 章）。 |
| 置换等变 | Permutation-equivariant | 自注意力打乱输入则输出相应打乱、不依赖位置（第 8 章）。 |
| 位置编码 | Positional encoding | 把序号信息注入 token，弥补自注意力对顺序视而不见（第 8 章）。 |
| Transformer | Transformer | 自注意力+残差+LayerNorm+FFN 堆叠，长程+并行+弱偏置统治 NLP（第 8 章）。 |
| 前馈子层 | Feed-forward network (FFN) | Transformer 里逐位置作用的 MLP，在每 token 内部加工信息（第 8 章）。 |
| 因果掩码 | Causal mask | 每 token 只看自己及左侧，自回归生成的要求（漏了=数据泄漏）（第 8 章）。 |
| 纯编码器 / 纯解码器 | Encoder-only / decoder-only | 双向看全文（BERT 类）/ 因果只看左侧（GPT 类）的 Transformer 范式（第 8 章）。 |
| Vision Transformer | Vision Transformer (ViT) | 图像切块当 token 喂 Transformer，大数据上能胜 CNN（第 8 章）。 |
| 复杂度 $O(N^2)$ | $O(N^2)$ complexity | 注意力相似度矩阵随序列长度平方增长的代价（第 8 章）。 |
| FlashAttention | FlashAttention | IO 感知的分块注意力实现，绕开 $O(N^2)$ 显存（第 8 章）。 |
| 嵌入 | Embedding | 把离散符号映成稠密低维可学向量，取代独热（第 8 章）。 |
| 查找表 | Lookup table | 嵌入层的实现：给 ID 取对应行向量（第 8 章）。 |
| 上下文嵌入 | Contextual embedding | 词向量由整句动态算出，解一词多义（Transformer 编码器）（第 8 章）。 |
| 静态嵌入 | Static embedding | 每词固定向量（word2vec 类），一词多义被压成折中（第 8 章）。 |
| 余弦相似度 | Cosine similarity | 归一化内积，衡量两嵌入方向接近度（第 8 章）。 |
| 图神经网络 | Graph neural network (GNN) | 为图数据而生，消息传递+置换不变聚合+权重共享（第 8 章）。 |
| 消息传递 | Message passing | 节点反复从邻居收集消息、聚合、更新表示（第 8 章）。 |
| 置换不变聚合 | Permutation-invariant aggregation | 用 sum/mean/max 汇总邻居消息（邻居无序）（第 8 章）。 |
| 图卷积网络 | Graph convolutional network (GCN) | 邻域归一化加权和+共享权重的 GNN 代表（第 8 章）。 |
| 图注意力网络 | Graph attention network (GAT) | 用注意力权重做邻居聚合的 GNN，连起 GNN 与注意力（第 8 章）。 |
| 过平滑 | Over-smoothing | 深 GNN 反复聚合使节点表示趋同，故通常很浅（第 8 章）。 |
| 残差连接 | Residual / skip connection | $\mathbf y=\mathbf x+F(\mathbf x)$，雅可比含 $I$ 给梯度修高速公路（第 8、9 章）。 |

---

## 8. 训练深度网络的工程学

| 中文名 | English | 解释 |
|---|---|---|
| 三道关卡 | Three gates | 能逼近≠训得动≠泛化好，深度学习要逐关打通（第 9 章）。 |
| 损失曲线 | Loss curve | 训练 / 验证损失随步数变化，诊断卡在哪关的仪表盘（第 9 章）。 |
| Dropout | Dropout | 训练时随机置零神经元（推理关闭），集成 / 抑制共适应的正则（第 9 章）。 |
| inverted dropout | Inverted dropout | 训练时除以保留概率，使推理无需缩放（第 9 章）。 |
| 共适应 | Co-adaptation | 神经元抱团死记的脆弱协作，Dropout 抑制它（第 9 章）。 |
| 数据增强 | Data augmentation | 对样本施加不改标签的变换造数据 = 把不变性当先验注入（第 9 章）。 |
| Mixup / CutMix | Mixup / CutMix | 线性混合两样本及标签，注入"样本间行为线性平滑"先验（第 9 章）。 |
| 标签平滑 | Label smoothing | 把独热目标改软（$1-\varepsilon$），防过度自信、改善校准（第 9 章）。 |
| 归一化 | Normalization | 把内部激活减均值除标准差再用可学 $\gamma,\beta$ 还原，稳住激活统计（第 9 章）。 |
| 内部协变量偏移 | Internal covariate shift | 层激活分布随训练漂移，归一化的最初（被质疑的）动机（第 9 章）。 |
| 批归一化 | Batch Normalization (BN) | 沿 batch 维统计，依赖 batch、训练/推理行为不同、CNN 常用（第 9 章）。 |
| 滑动平均统计 | Running mean / variance | BN 训练时维护、推理时用的全局统计量（第 9 章）。 |
| 层归一化 | Layer Normalization (LN) | 沿特征维统计，不依赖 batch、适配变长序列、Transformer 标配（第 9 章）。 |
| GroupNorm / InstanceNorm / RMSNorm | GroupNorm / InstanceNorm / RMSNorm | 归一化家族变体，分歧只在沿哪个维度统计（第 9 章）。 |
| 退化问题 | Degradation problem | 单纯堆深反而训练误差升高（非过拟合），残差连接解决（第 8、9 章）。 |
| 梯度高速公路 | Gradient highway | 残差的 $+\mathbf x$ 恒等通路让梯度不被层层缩放地回流（第 8、9 章）。 |
| pre-norm / post-norm | Pre-norm / post-norm | 归一化在残差分支内 / 外，pre-norm 更稳、大模型主流（第 9 章）。 |
| 混合精度训练 | Mixed precision training | 大部分用 16 位算、关键处保 32 位，提速省显存（第 9 章）。 |
| FP16 / BF16 | FP16 / bfloat16 | 16 位浮点：FP16 量程窄精度中、BF16 量程同 FP32 精度粗（第 9 章）。 |
| 损失缩放 | Loss scaling | 损失乘大常数撑起 FP16 梯度量级、防小梯度下溢（第 9 章）。 |
| 主权重副本 | Master weights | 保留 FP32 参数副本累加更新，防长期累加被舍入吞掉（第 9 章）。 |
| 线性缩放规则 | Linear scaling rule | batch 翻倍学习率大致翻倍的经验法则（有边界）（第 9 章）。 |
| 梯度累积 | Gradient accumulation | 连跑几个小 batch 累加梯度再更新，等效大 batch（第 9 章）。 |
| 网格搜索 | Grid search | 遍历超参候选组合，组合数随维度指数爆炸（第 9 章）。 |
| 随机搜索 | Random search | 随机采样超参，同预算下常胜网格（重要维采样更密）（第 9 章）。 |
| 贝叶斯优化 | Bayesian optimization | 用代理模型主动选最可能有收益的超参点试，样本高效（第 9 章）。 |
| 采集函数 | Acquisition function | 贝叶斯优化里权衡探索-利用选下一点（如期望改进 EI）（第 9 章）。 |
| Hyperband / 连续减半 | Hyperband / successive halving | 给多组超参少量预算先训、淘汰垫底者集中资源给幸存者（第 9 章）。 |
| 可复现性 | Reproducibility | 固定种子、记录环境、多种子报均值±方差（第 9 章）。 |
| 可复现性危机 | Reproducibility crisis | 大量已发表 ML 结果难独立重现（cherry-picking / 泄漏 / 基线不公）（第 9 章）。 |
| 迁移学习 | Transfer learning | 在数据丰富的源任务预训练学通用表示，再迁移到目标任务（第 9 章）。 |
| 预训练 | Pretraining | 在大规模数据上先学通用表示这一昂贵步骤（第 9、10 章）。 |
| 微调 | Fine-tuning | 解冻预训练模型用小学习率续训以适应目标任务（第 9 章）。 |
| 特征提取 / 线性探测 | Feature extraction / linear probe | 冻结骨干、只训顶上小分类头（第 9、10 章）。 |
| 逐步解冻 | Gradual unfreezing | 先训顶层再逐层解冻的折中微调（第 9 章）。 |
| 灾难性遗忘 | Catastrophic forgetting | 微调时大学习率把预训练学的通用知识冲掉（第 9 章）。 |
| 参数高效微调 | Parameter-efficient fine-tuning (PEFT) | 只训极少量参数（LoRA / adapter / prompt tuning）（第 9、10 章）。 |
| LoRA | Low-Rank Adaptation (LoRA) | 冻结原权重、只训低秩增量 $\Delta W=BA$，呼应低秩 / SVD（第 9 章）。 |
| 负迁移 | Negative transfer | 源与目标差异太大时预训练反而不如随机初始化（第 9 章）。 |
| 领域偏移 | Domain shift | 预训练数据分布与目标差很远时迁移收益大减（第 9 章）。 |
| 先过拟合一个小 batch | Overfit a small batch | 调试头号招式：记不住几个样本则是优化 / 实现 bug（第 9 章）。 |
| 少样本 | Few-shot | 只需几十几百样本微调就达高水平的迁移能力（第 9、10 章）。 |
| 知识蒸馏 | Knowledge distillation | 让小学生模型拟合大教师模型的软标签 / logits，压缩模型并保留性能（第 9 章）。 |
| 泛化之谜 | Generalization mystery | 过参数化网络违背经典理论却泛化好，理论严重滞后（第 9 章）。 |
| 彩票假设 | Lottery ticket hypothesis | 大网络里藏着单独训练就同样好的"中奖"子网络（第 9 章）。 |
| 最小范数解 | Minimum-norm solution | 欠定系统里范数最小的特解，SGD 隐式偏好它（呼应伪逆 / SVD）（第 9 章）。 |

---

## 9. 生成模型

| 中文名 | English | 解释 |
|---|---|---|
| 采样器 | Sampler | 能反复输出合法 $\mathbf x$ 的随机过程，生成模型的本质（第 10 章）。 |
| 最优传输 | Optimal transport | 把简单分布"搬运"到复杂数据分布的数学工具（第 10 章）。 |
| 显式 / 隐式似然 | Explicit / implicit likelihood | 写得出密度（AR/VAE/扩散）/ 只会采样不写密度（GAN）（第 10 章）。 |
| 自回归模型 | Autoregressive model (AR) | 用链式法则把联合分布拆成一串条件，逐元素生成、精确似然（第 10 章）。 |
| 链式法则（概率） | Chain rule of probability | $P(\mathbf x)=\prod_t P(x_t\mid\mathbf x_{<t})$，自回归的精确分解（第 10 章）。 |
| 大语言模型 | Large language model (LLM) | 自回归+Transformer 解码器在海量文本上预测下一词，涌现语言能力（第 8、10 章）。 |
| 温度（采样） | Temperature ($\tau$) | logits 除以 $\tau$ 调采样随机性 / 多样性（第 10 章）。 |
| 集束搜索 | Beam search | 自回归解码时并行保留 top-$k$ 条候选序列、近似找高概率整句（第 10 章）。 |
| 核采样 / Top-p | Nucleus / top-p sampling | 只在累积概率达 $p$ 的最小词集里采样，比 top-$k$ 更自适应（第 10 章）。 |
| 曝光偏差 | Exposure bias | 训练喂真实前缀、生成喂自产前缀，误差沿序列累积（第 10 章）。 |
| 变分自编码器 | Variational autoencoder (VAE) | 隐变量模型，优化 ELBO+重参数化，隐空间规整、采样快、样本偏糊（第 10 章）。 |
| 隐变量模型 | Latent variable model | 假设观测背后有低维隐变量 $\mathbf z$ 生成数据（第 10 章）。 |
| 重参数化技巧 | Reparameterization trick | $\mathbf z=\boldsymbol\mu+\boldsymbol\sigma\odot\boldsymbol\epsilon$ 把随机性挪到外面、让梯度可穿过采样（第 10 章）。 |
| 变分推断 | Variational inference | 用可优化的 $q$ 逼近难算后验、最大化 ELBO（第 6、10 章）。 |
| 率失真 | Rate–distortion | 码率（KL）与失真（重构）的权衡，VAE 的信息论身份（第 10 章）。 |
| 生成对抗网络 | Generative adversarial network (GAN) | 生成器 vs 判别器的 min-max 博弈，隐式定义分布（第 10 章）。 |
| 生成器 / 判别器 | Generator / discriminator | GAN 里造假者 $G$ vs 验钞员 $D$（第 10 章）。 |
| min-max 博弈 | Min-max game | GAN 的对抗目标，$G$ 最小化、$D$ 最大化同一目标（第 10 章）。 |
| 纳什均衡 | Nash equilibrium | GAN 理论平衡点：$p_G=p_\text{data}$、$D$ 蒙 1/2（第 10 章）。 |
| Jensen–Shannon 散度 | Jensen–Shannon divergence | 固定最优判别器后 GAN 实际最小化的分布距离（第 10 章）。 |
| 模式崩溃 | Mode collapse | GAN 只生成少数能骗过 $D$ 的样本、丢失多样性（第 10 章）。 |
| 扩散模型 | Diffusion model | 前向加噪 + 学反向去噪 / 得分，图像 / 分子 / 蛋白生成 SOTA、采样慢（第 10 章）。 |
| 前向 / 反向过程 | Forward / reverse process | 固定加噪把数据揉成噪声 / 学去噪从噪声生成数据（第 10 章）。 |
| 去噪 | Denoising | 扩散网络预测每步掺入的噪声，训练即回归损失（第 10 章）。 |
| 得分匹配 | Score matching | 学数据对数密度梯度 $\nabla_\mathbf x\log p(\mathbf x)$（得分），与去噪等价（第 10 章）。 |
| 得分 | Score | $\nabla_\mathbf x\log p$，指向数据更可能的方向（第 10 章）。 |
| 朗之万动力学 | Langevin dynamics | 沿得分爬+加噪的采样，本质 MCMC，扩散生成的根（第 10 章）。 |
| U-Net | U-Net | 带跳连的编码-解码卷积网，扩散模型预测噪声的常用骨干（第 10 章）。 |
| 无分类器引导 | Classifier-free guidance | 混合有 / 无条件预测放大条件信号，提条件扩散的可控性与保真（第 10 章）。 |
| 结构生成 | Structure generation | 用扩散在原子 / 主链坐标空间生成新分子 / 蛋白结构（第 10、11 章）。 |
| FID | Fréchet Inception Distance | 图像生成质量指标，有盲区（第 10 章）。 |
| 困惑度 (语言) | Perplexity | $\exp(\text{交叉熵})$，语言模型对语言的平均不确定性，越低越好（第 10、11 章）。 |
| 补全 / 修复 | Completion / inpainting | 给定部分数据按条件分布采样剩余部分（第 10 章）。 |

---

## 10. 自监督与强化学习

| 中文名 | English | 解释 |
|---|---|---|
| 掩码预测 | Masked prediction | 遮住输入一部分让模型填回，自监督主力（第 10 章）。 |
| 掩码语言模型 | Masked language model (MLM) / BERT | 双向预测被遮的词，学理解型表示（第 10 章）。 |
| 掩码自编码器 | Masked autoencoder (MAE) | 遮住大部分图块让模型重建，逼理解全局结构（第 10 章）。 |
| 对比学习 | Contrastive learning | 学表示空间让相似样本近、不相似远（第 10 章）。 |
| 正 / 负样本 | Positive / negative pairs | 同一对象的不同增强视角 / 不同对象，对比学习的监督信号（第 10 章）。 |
| InfoNCE | InfoNCE / NT-Xent | 在候选里认出正样本的对比损失，是互信息的下界（第 10 章）。 |
| 互信息 | Mutual information | 两变量共享的信息量，对比学习最大化它的下界（第 10 章）。 |
| 捷径学习 | Shortcut learning | 模型抄与真意图无关的捷径完成任务、学到没用表示（第 10、11 章）。 |
| 基础模型 | Foundation model | 在超大无标签数据上自监督预训练的通用大模型（第 9、10 章）。 |
| 预训练+适配 | Pre-train then adapt | 一次大规模预训练、再微调 / 提示适配到下游的新范式（第 10 章）。 |
| 提示 | Prompting | 不改参数、只在输入里用自然语言指示模型做什么（第 10 章）。 |
| 上下文学习 | In-context learning (ICL) | 从输入示例"现学"任务、不改权重，少样本能力（第 10 章）。 |
| 思维链 | Chain-of-thought (CoT) | 提示模型先写出中间推理步骤再给答案，提升多步推理（第 10 章）。 |
| 检索增强生成 | Retrieval-augmented generation (RAG) | 先检索相关文档再喂给模型生成，补外部知识 / 减幻觉（连向 ANN）（第 10 章）。 |
| 多模态 | Multimodal | 把图像 / 文本 / 音频嵌到同一空间或同一序列框架（如 CLIP）（第 10 章）。 |
| 零样本分类 | Zero-shot classification | 没在特定标签集训练过、靠图文同空间比相似度分类（第 10 章）。 |
| 缩放定律 | Scaling laws | 模型 / 数据 / 算力一起增大、损失沿幂律平滑下降的经验趋势（第 10 章）。 |
| 能力涌现 | Emergent abilities | 某些能力越过规模门槛"突然"出现（是否度量假象有争议）（第 10 章）。 |
| 幻觉 | Hallucination | 大模型自信地胡说，能力≠理解的体现（第 10 章）。 |
| 智能体 / 环境 | Agent / environment | RL 里行动的主体 / 给出奖励与状态的世界（第 10 章）。 |
| 奖励 | Reward | RL 里"这一步做得好不好"的反馈信号，常延迟稀疏（第 10 章）。 |
| 信用分配 | Credit assignment | 把延迟总奖励分摊回沿途各动作的核心难题（第 1、10 章）。 |
| 探索 vs 利用 | Exploration vs exploitation | 试没试过的动作 vs 用已知最好动作的两难（第 10 章）。 |
| 马尔可夫决策过程 | Markov decision process (MDP) | 状态 / 动作 / 转移 / 奖励 / 折扣五元组，RL 的数学骨架（第 10 章）。 |
| 马尔可夫性 | Markov property | 下一步只依赖当前状态与动作、与更早历史无关（第 10 章）。 |
| 策略 | Policy ($\pi$) | 每个状态下选各动作的概率，RL 要学的行为（第 10 章）。 |
| 回报 | Return ($G_t$) | 折扣累积奖励 $\sum\gamma^k R_{t+k}$，RL 要最大化的量（第 10 章）。 |
| 折扣因子 | Discount factor ($\gamma$) | 未来奖励的贴现率，$<1$ 保证无穷和收敛（第 10 章）。 |
| 价值函数 | Value function ($V^\pi,Q^\pi$) | 从某状态 / 状态-动作起的期望回报（第 10 章）。 |
| 贝尔曼方程 | Bellman equation | 价值=即时奖励+折扣后未来价值的递归，RL 引擎（第 10 章）。 |
| 贝尔曼最优方程 | Bellman optimality equation | 把期望换成取 $\max$ 动作，解出最优价值 $Q^*$（第 10 章）。 |
| 价值迭代 | Value iteration | 环境已知时反复套贝尔曼方程更新价值的动态规划（第 10 章）。 |
| 自举 | Bootstrapping | 用下一步的估计更新这一步的估计（TD / Actor-Critic）（第 10 章）。 |
| Q-learning | Q-learning | 学 $Q^*$、off-policy、TD、更新含 $\max$、适合离散动作（第 10 章）。 |
| 时序差分 | Temporal difference (TD) | 不等一局结束、用下一步估计更新当前估计（第 10 章）。 |
| TD 误差 | TD error ($\delta_t$) | $r+\gamma V(s')-V(s)$，目标减当前，价值学习与优势的信号（第 10 章）。 |
| off-policy / on-policy | Off-policy / on-policy | 学的目标策略≠ / =实际采数据的策略（Q-learning vs 策略梯度）（第 10 章）。 |
| 深度 Q 网络 | Deep Q-Network (DQN) | 用神经网络近似 $Q$ 的 Q-learning（第 10 章）。 |
| 策略梯度 | Policy gradient | 直接参数化策略、梯度上升期望回报、on-policy、连续动作（第 10 章）。 |
| 策略梯度定理 | Policy gradient theorem | $\nabla J=\mathbb E[\sum\nabla\log\pi\cdot G_t]$，让高回报动作概率升（第 10 章）。 |
| Actor–Critic | Actor–Critic | 演员（策略）+评论家（价值估计），用优势 / TD 误差降方差（第 10 章）。 |
| 基线 | Baseline | 减去只依赖状态的项，不引偏只降策略梯度方差（第 10 章）。 |
| 优势函数 | Advantage function ($A=Q-V$) | 动作比该状态平均水平好多少，只奖励"超出预期"（第 10 章）。 |
| 得分函数 | Score function | $\nabla_\theta\log\pi_\theta$，与 MLE 得分、扩散得分同源（第 10 章）。 |
| $\epsilon$-贪婪 | $\epsilon$-greedy | 以 $\epsilon$ 随机探索、否则选最优，最朴素的探索策略（第 10 章）。 |
| 人类反馈强化学习 | RLHF | 用偏好训奖励模型、再用 RL 微调语言模型做对齐（第 10 章）。 |
| 奖励模型 | Reward model | 预测"人类更喜欢哪个回答"的模型，RLHF 的奖励来源（第 10 章）。 |
| 对齐 | Alignment | 让大模型输出符合人类偏好（有用 / 安全 / 少胡说）（第 10 章）。 |
| 奖励钻空子 | Reward hacking | 智能体钻奖励设计漏洞刷分而非达成真目标（第 10 章）。 |
| 模仿学习 | Imitation learning | 借用监督招式从专家示范学策略（第 1 章）。 |
| 网格世界 | Grid world | 让 RL 概念可视化的玩具环境（第 10 章）。 |

---

## 11. 评估、流水线与负责任的 AI

| 中文名 | English | 解释 |
|---|---|---|
| 端到端流水线 | End-to-end pipeline | 问题定义→数据→特征→模型→训练→评估→部署→监控的环（第 11 章）。 |
| 问题定义 | Problem framing | 想清"是不是 ML 问题、目标 / 标签 / 成功标准 / 约束"，回报最高的一段（第 11 章）。 |
| 探索性数据分析 | Exploratory data analysis (EDA) | 画分布、看相关、查不平衡 / 泄漏的数据理解步骤（第 11 章）。 |
| 强基线 | Strong baseline | 平凡基线（猜多数类 / 均值）+简单模型基线，给提升一个分母（第 11 章）。 |
| 复杂度 / 收益比 | Complexity-to-benefit ratio | 每加复杂度要问换来的提升是否值（含训练 / 运维 / 可解释 / 风险）（第 11 章）。 |
| 错误分析 | Error analysis | 人工翻看错样本归类，比再调参更能指明下一步（第 11 章）。 |
| 准确率 | Accuracy | 预测对的比例，类别不平衡时极具误导性（第 11 章）。 |
| 类别不平衡 | Class imbalance | 某类极少，准确率失效、需看精确率 / 召回 / PR-AUC（第 2、11 章）。 |
| 混淆矩阵 | Confusion matrix | TP/FP/TN/FN 四格，各分类指标的来源（第 11 章）。 |
| 精确率 | Precision | $TP/(TP+FP)$，"判为阳性里真阳性多少"，关心误报代价（第 11 章）。 |
| 召回率 | Recall | $TP/(TP+FN)$，"真阳性里抓到多少"，关心漏报代价（第 11 章）。 |
| F1 分数 | F1 score | 精确率与召回率的调和平均，逼两者都不太差（第 11 章）。 |
| ROC 曲线与 AUC | ROC curve & AUC | 扫阈值的真阳率 vs 假阳率曲线下面积 = 排序正确率（第 11 章）。 |
| PR-AUC | Precision-Recall AUC | PR 曲线下面积，极度不平衡时比 ROC-AUC 更可信（随基率变）（第 11 章）。 |
| 决定系数 | Coefficient of determination ($R^2$) | 模型比"永远预测均值"基线好多少，可为负（第 11 章）。 |
| 分群评估 | Slice-based evaluation | 按子群体分别看指标，防整体高分掩盖少数群灾难（第 11 章）。 |
| MLOps | MLOps | 把 ML 系统当需长期运维的软件，版本化 / 监控 / 重训（第 11 章）。 |
| 训练 / 服务一致性 | Training-serving skew | 训练与线上特征计算逻辑必须一致，否则模型莫名变差（第 11 章）。 |
| 特征存储 | Feature store | 让线下训练与线上服务共享同一份特征定义，消灭 skew（第 11 章）。 |
| 影子 / 灰度 / 回滚 | Shadow / canary / rollback | 新模型零风险对比 / 小流量验证 / 出事退回的渐进发布（第 11 章）。 |
| 监控 | Monitoring | 监控输入 / 预测分布漂移与线上指标，及早预警（第 11 章）。 |
| 反馈回路偏差 | Feedback loop bias | 系统只收集自己推荐内容的反馈、数据越来越窄（信息茧房）（第 11 章）。 |
| 负责任的 AI | Responsible AI | 直面分布漂移 / 偏见 / 隐私 / 鲁棒 / 能耗等部署的内在要求（第 11 章）。 |
| 虚假相关 | Spurious correlation | 数据采集伪影而非真因果的相关性（"有雪=狼"），分布一变就崩（第 11 章）。 |
| 偏见与公平性 | Bias & fairness | 模型忠实学并放大历史数据偏见，还披"客观"外衣（第 11 章）。 |
| 公平性不可能定理 | Fairness impossibility | 基率不同时多种公平定义一般无法同时满足，需价值权衡（第 11 章）。 |
| 可解释性 | Interpretability / explainability | 解释"模型为什么这么判"，高风险领域的刚需（第 11 章）。 |
| 事后解释 | Post-hoc explanation | 用局部可加模型近似黑盒（如 SHAP），可能不忠实（第 11 章）。 |
| 隐私 | Privacy | 模型可能记住并泄露训练数据敏感信息（第 11 章）。 |
| 成员推断攻击 | Membership inference attack | 靠训练样本损失更低判断某人是否在训练集，根在过拟合（第 11 章）。 |
| 差分隐私 | Differential privacy | 训练注入受控噪声、从数学上限制单样本对输出的影响（第 11 章）。 |
| 联邦学习 | Federated learning | 数据不出本地、只传模型更新（第 11 章）。 |
| 对抗样本 | Adversarial example | 人眼难察的微小扰动让模型高置信度判错，源于高维线性性（第 11 章）。 |
| 鲁棒性 | Robustness | 模型对扰动 / 分布外输入的稳健程度（第 11 章）。 |
| 相关 ≠ 因果 | Correlation ≠ causation | 模型学的是相关、不等于因果，贯穿负责任 AI 的红线（第 1、11 章）。 |
| 矩阵分解 | Matrix factorization | 把交互矩阵近似成低秩之积、用户 / 物品得嵌入，推荐核心（第 8、11 章）。 |

---

## 12. 人物·文献·里程碑

| 中文名 | English | 解释 |
|---|---|---|
| 感知机（Rosenblatt） | Perceptron (Rosenblatt) | 最早可学习线性分类器，点燃第一波神经网络热情（第 1、3 章）。 |
| Minsky & Papert《Perceptrons》 | Minsky & Papert | 指出感知机解不了 XOR，催生第一次 AI 寒冬（第 3 章）。 |
| 统计学习 | Statistical learning | SVM / 核 / 集成 / 图模型，把学习建在概率与优化上（第 1 章）。 |
| 没有免费的午餐（Wolpert） | No free lunch | 所有问题平均任两算法等价，泛化必须靠归纳偏置（第 1、5 章）。 |
| Tibshirani（Lasso, 1996） | Tibshirani | L1 稀疏与特征选择的提出者（第 3 章）。 |
| Zou & Hastie（弹性网络） | Zou & Hastie | 弹性网络解决纯 Lasso 在相关特征上的不稳（第 3 章）。 |
| Cover–Hart 定理 | Cover–Hart theorem | 1-NN 渐近错误率不超过两倍贝叶斯误差（第 5 章）。 |
| Bergstra & Bengio（随机搜索） | Bergstra & Bengio | 论证随机搜索同预算下常胜网格搜索（第 9 章）。 |
| Breiman（随机森林） | Breiman | 随机森林的提出者（第 5 章）。 |
| Freund & Schapire（AdaBoost） | Freund & Schapire | AdaBoost / boosting 的源头（第 5 章）。 |
| Friedman（梯度提升） | Friedman | 梯度提升（函数空间梯度下降）的提出者（第 5 章）。 |
| Chen & Guestrin（XGBoost） | Chen & Guestrin | 高效 GBDT 工业实现 XGBoost 的作者（第 5 章）。 |
| Cortes & Vapnik（SVM） | Cortes & Vapnik | 支持向量机（最大间隔+核）的提出者（第 5 章）。 |
| Ng & Jordan（生成 vs 判别） | Ng & Jordan | "少数据生成式快、多数据判别式准"的经典比较（第 5 章）。 |
| Grinsztajn 等（表格 GBDT） | Grinsztajn et al. | 系统比较"表格上 GBDT 是否仍胜深度模型"（第 5 章）。 |
| LeNet（LeCun） | LeNet (LeCun) | 最早成功的 CNN（手写数字），确立卷积+池化+全连接范式（第 8 章）。 |
| AlexNet（Krizhevsky 等, 2012） | AlexNet | 在 ImageNet 夺冠、引爆深度学习浪潮的深度 CNN（第 1、8 章）。 |
| VGG | VGG | 全用 $3\times3$ 小核靠堆深度，揭示"更深=更强"（第 8 章）。 |
| ResNet（He 等, 2015） | ResNet | 残差连接破解深网络训练，几乎一切现代深架构的地基（第 8、9 章）。 |
| Transformer（Vaswani 等, 2017） | Transformer ("Attention Is All You Need") | 自注意力为核心、甩掉 RNN，统治 NLP 并扩散到视觉与科学（第 8 章）。 |
| BERT | BERT | 掩码语言模型双向预训练，理解型表示的代表（第 10 章）。 |
| GPT 系列 | GPT family | 纯解码器自回归语言模型，生成式大模型代表（第 10 章）。 |
| word2vec | word2vec | 词嵌入代表，"国王−男人+女人≈女王"涌现语义几何（第 8 章）。 |
| CLIP | CLIP | 图文对比学习对齐到同一空间，零样本分类（第 10 章）。 |
| Bahdanau 等（注意力起源） | Bahdanau et al. | 最早把注意力作为 seq2seq 翻译的对齐补丁（第 8 章）。 |
| Goodfellow 等（GAN） | Goodfellow et al. | 生成对抗网络的提出者（第 10 章）。 |
| Kingma & Welling（VAE） | Kingma & Welling | 变分自编码器（ELBO+重参数化）的提出者（第 10 章）。 |
| Ho 等（DDPM 扩散） | Ho et al. (DDPM) | 去噪扩散概率模型，现代扩散生成的代表（第 10 章）。 |
| Frankle & Carbin（彩票假设） | Frankle & Carbin | 大网络里藏中奖子网络的彩票假设（第 9 章）。 |
| Zhang 等（拟合随机标签） | Zhang et al. | 深网络能记住随机标签，揭示泛化之谜（第 2、9 章）。 |
| Belkin 等（双下降） | Belkin et al. | 把经典 U 形与过参数化二次下降统一的双下降工作（第 2 章）。 |
| Kingma & Ba（Adam） | Kingma & Ba | Adam 优化器的提出者（第 4 章）。 |
| Loshchilov & Hutter（AdamW） | Loshchilov & Hutter | 解耦权重衰减的 AdamW（第 9 章）。 |
| Dauphin 等（鞍点） | Dauphin et al. | "高维非凸里鞍点是主要障碍"的代表性论述（第 4 章）。 |
| AlphaFold（Jumper 等, 2021） | AlphaFold | Transformer+几何深度学习预测蛋白结构，科学旗舰应用（第 8、11 章）。 |
| RFdiffusion（分子 / 蛋白生成） | RFdiffusion | SE(3) 等变扩散从头设计蛋白，生成模型科学应用（第 10、11 章）。 |
| Sutton & Barto《强化学习》 | Sutton & Barto, *RL: An Introduction* | RL 标准教材（MDP / 价值 / 策略 / TD）（第 10、11 章）。 |
| Hastie/Tibshirani/Friedman《ESL》 | *The Elements of Statistical Learning* | 统计学习圣经，经典方法 / 偏差-方差 / 集成的权威（第 2、5 章）。 |
| Bishop《PRML》 | *Pattern Recognition and Machine Learning* | 以概率视角统贯监督 / 无监督的经典（第 2、3 章）。 |
| Murphy《Probabilistic ML》 | Murphy, *Probabilistic Machine Learning* | 现代全面的概率机器学习教材（全课）（第 1、11 章）。 |
| Goodfellow/Bengio/Courville《Deep Learning》 | *Deep Learning* | 深度学习标准底本（第 7–10 章）（第 1、11 章）。 |

---

> **一句话收束**：这张表是全课术语的索引而非替代——遇到不熟的词先在此定位中英名与所在章，再回正文精读。把"中文直觉 + 英文术语 + 所在章"三者绑定，并随时用三条暗线（泛化 / 表示+目标+优化 / 归纳偏置）去归位每个新概念，你读机器学习论文与代码的速度与深度都会上一个台阶。
