# 第 7 章 点估计：极大似然、充分统计量、Fisher 信息与 EM

> **本章定位**：前六章我们做的是「正问题」——给定一个分布（或一个随机模型），算它的期望、它的尾巴、它在高维如何耦合、它的样本平均会收敛到哪、波动多大。但真实研究永远是**反问题**：手里只有数据（一个 MSA、一段轨迹、一批突变计数），背后那个「真分布」的参数 $\theta$ 是**未知的**，得从数据里把它**估出来**。本章就是频率派统计推断（frequentist inference）的主干，专讲「怎样从数据估参数、估得好不好、能估多好」。我们先立评价标准——偏差、方差、均方误差、一致性（§7.1），它正是机器学习里「偏差-方差权衡」的统计学母体；再讲两条造估计量的通用流水线：朴素但稳的**矩估计**（§7.2）与几乎统治了现代统计与 ML 的**极大似然估计（MLE）**（§7.3）。MLE 这一节是全章重心：它的解法、它的**不变性**、它与信息论课程 ch2 §2.8「MLE = 最小化交叉熵 = 最小化到经验分布的 KL」的精确同义，以及它的渐近三件套（一致、渐近正态、渐近有效）连同诚实的适用边界。然后是两个把「估计」抬到理论高度的概念：**充分统计量**（§7.4，数据中关于 $\theta$ 的全部信息的浓缩，紧扣第 4 章 §4.5 指数族）与 **Fisher 信息 + Cramér–Rao 下界**（§7.5，告诉你「无偏估计的方差有一条物理下限」，也回答了「需要多少条同源序列」这个折叠研究的实际问题）。最后两节面向真实算法：隐变量模型的 MLE 引擎 **EM 算法**（§7.6，profile HMM 的 Baum–Welch、高斯混合都是它），与正则化/伪似然预告（§7.7，把岭/lasso 焊到第 8 章贝叶斯 MAP 上，把**伪似然**指向 plmDCA——因为 Potts 的配分函数算不动）。读完本章，你会拥有「从数据反推参数」的整套频率派武器，并清楚每件武器的脾气与失效边界。这是通往第 8 章贝叶斯、第 9 章检验的必经之路。

---

## 7.1 估计量与它的评价：偏差、方差、MSE、一致性

### 7.1.1 什么是估计量

设我们有来自某个分布 $p_\theta$ 的独立同分布（independent and identically distributed, iid）样本 $X_1,\dots,X_n$，其中参数 $\theta\in\Theta$ 是**未知**的真值。一个 **估计量（estimator）** 就是一个把数据映成参数估计的函数

$$\hat\theta=\hat\theta(X_1,\dots,X_n).$$

请把记号读清楚：$\theta$ 是一个**固定但未知**的数（频率派的根本立场——参数不是随机的，第 8 章贝叶斯才让它随机）；而 $\hat\theta$ 是数据的函数，数据是随机的，所以 $\hat\theta$ 是一个**随机变量**，它有自己的分布，叫**抽样分布（sampling distribution）**。把这两件事分清，是理解全章的前提：我们要研究的是「一个随机的估计 $\hat\theta$ 围绕一个不动的真值 $\theta$ 如何分布」。

> **CS 读者陷阱（估计量 vs 估计值）**：「estimator」是函数（算法），「estimate」是把具体数据代进去得到的那个数。样本均值 $\bar X=\frac1n\sum X_i$ 是估计量；对你这批数据它等于 $3.17$，$3.17$ 是估计值。评价一个估计量好不好，问的永远是它的**抽样分布**（在所有可能的数据集上的表现），而不是某一次的具体数字——这跟你评价一个随机算法看它的期望/方差、而非某一次运行，是同一个思维。

### 7.1.2 偏差、方差与均方误差

评价 $\hat\theta$ 的两把尺子：

**偏差（bias）** 衡量「平均而言，估计偏离真值多少」：

$$\mathrm{Bias}(\hat\theta)=\mathbb E_\theta[\hat\theta]-\theta.$$

若 $\mathrm{Bias}(\hat\theta)=0$ 对一切 $\theta$ 成立，称 $\hat\theta$ **无偏（unbiased）**。这里的期望 $\mathbb E_\theta$ 是对抽样分布取的——「在真值是 $\theta$ 的前提下，把所有可能的 $n$ 样本数据集都试一遍，$\hat\theta$ 的平均」。

**方差（variance）** $\mathrm{Var}_\theta(\hat\theta)$ 衡量「估计随数据抖动多厉害」（即抽样分布的散布）。

把两者合成一把综合尺子——**均方误差（mean squared error, MSE）**：

$$\mathrm{MSE}(\hat\theta)=\mathbb E_\theta\big[(\hat\theta-\theta)^2\big].$$

它有一个干净到必须背下来的分解：

> **定理（偏差–方差分解）。** 对任意估计量，
> $$\boxed{\ \mathrm{MSE}(\hat\theta)=\underbrace{\big(\mathrm{Bias}(\hat\theta)\big)^2}_{\text{系统性偏离}}+\underbrace{\mathrm{Var}(\hat\theta)}_{\text{随机抖动}}.\ }$$

一行证明：记 $m=\mathbb E[\hat\theta]$，则 $\mathbb E[(\hat\theta-\theta)^2]=\mathbb E[(\hat\theta-m+m-\theta)^2]=\mathbb E[(\hat\theta-m)^2]+2(m-\theta)\underbrace{\mathbb E[\hat\theta-m]}_{=0}+(m-\theta)^2=\mathrm{Var}(\hat\theta)+\mathrm{Bias}^2$。交叉项因 $\mathbb E[\hat\theta-m]=0$ 消失。

这个分解的份量怎么强调都不过分：它说**误差有两个来源**——一个是「打靶总往左偏」（偏差），一个是「手抖打散」（方差）。无偏不等于好：一个无偏但方差巨大的估计量，可能远不如一个略有偏但方差很小的估计量。这就引出权衡。

### 7.1.3 偏差–方差权衡（直接连机器学习）

既然 $\mathrm{MSE}=\mathrm{Bias}^2+\mathrm{Var}$，我们常常**故意引入一点偏差去换取方差的大幅下降**，使总 MSE 更小。这正是统计学里「收缩估计」（shrinkage）与机器学习里「正则化」（regularization）的全部动机，也是 §7.7 岭回归/lasso 的理论根。

> **与你的研究的连接（伪计数是一次偏差换方差）**：拿到一个 MSA 的某一列，纯频率估计 $\hat p(a)=\dfrac{\#a}{n}$ 是无偏的（§7.3 会证它就是 MLE），但当序列条数 $n$ 很小（浅 MSA），它方差巨大、还会给没观测到的氨基酸赋概率 0。加**伪计数（pseudocount）** $\hat p(a)=\dfrac{\#a+\alpha}{n+20\alpha}$ 引入了偏差（把估计往均匀分布拉），却显著压低了方差、消灭了零概率灾难。当 $n$ 大时这点偏差可忽略；当 $n$ 小时它救命。这一权衡的贝叶斯解释（伪计数 = Dirichlet 先验）见第 4 章 §4.6 与第 8 章——但你现在已经能从**纯频率派的 MSE 分解**理解它为什么对：用一点偏差买来更小的方差。信息论课程 ch2 §2.8 末尾的「零频率需平滑」也是这件事的信息论侧写。

> **给数学/CS 读者的视角（这就是欠拟合/过拟合）**：把「偏差大」读成「模型太简单、欠拟合（underfitting）」，把「方差大」读成「模型太复杂、过拟合（overfitting）」，偏差–方差分解就是「测试误差 = 欠拟合 + 过拟合 + 不可约噪声」的统计学母体。深度学习里调正则强度、early stopping、dropout，本质都在这条曲线上找 MSE 最小点。本章把这条 ML 直觉**追溯到它的源头**：它不是 ML 发明的，是 1940 年代点估计理论就有的恒等式。

### 7.1.4 一致性：样本越多，估计越准

偏差/方差/MSE 是「固定 $n$」的评价。还有一把「$n\to\infty$」的尺子：

**一致性（consistency）**：称 $\hat\theta_n$ 是 $\theta$ 的**一致估计**，若 $\hat\theta_n\xrightarrow{P}\theta$（依概率收敛，第 6 章），即数据越多、估计越逼近真值。一个充分条件极好用：

$$\mathrm{MSE}(\hat\theta_n)\to0\ \Longrightarrow\ \hat\theta_n\xrightarrow{L^2}\theta\ \Longrightarrow\ \hat\theta_n\xrightarrow{P}\theta\quad(\text{一致}).$$

也就是说，**偏差与方差都随 $n$ 趋于 0**，就一致。样本均值 $\bar X$ 估 $\mu$ 是一致的，因为它无偏（偏差 0）且 $\mathrm{Var}(\bar X)=\sigma^2/n\to0$——这正是第 6 章大数定律的内容。

> **诚实的边界（一致 ≠ 无偏，无偏 ≠ 一致）**：这两个概念**互不蕴含**。(1) 样本方差的 MLE $\hat\sigma^2=\frac1n\sum(X_i-\bar X)^2$ **有偏**（$\mathbb E[\hat\sigma^2]=\frac{n-1}{n}\sigma^2<\sigma^2$，系统性低估），但**一致**（偏差 $\sim\sigma^2/n\to0$）。这就是为什么有「除以 $n-1$」的无偏版本 $S^2$——它修掉了那点偏差（第 4 章 §4.2 自由度那条注解，本章 §7.3.4 会推）。(2) 反过来，一个无偏估计若方差不随 $n$ 缩小（比如永远只用第一个样本 $X_1$ 估 $\mu$），它无偏但**不一致**。所以「无偏」是个常被高估的性质——很多优秀估计量（包括正则化估计、收缩估计）都是有偏的。**别把无偏当成金标准**。

> **要点**：估计量 $\hat\theta$ 是数据的随机函数，评价它看它的抽样分布。三把尺子：偏差（系统偏离）、方差（随机抖动）、二者经 $\mathrm{MSE}=\mathrm{Bias}^2+\mathrm{Var}$ 合成。这条分解是 ML 偏差–方差权衡/正则化的母体——常用一点偏差换大幅降方差（伪计数即一例）。$n\to\infty$ 看一致性，MSE→0 即一致；无偏与一致互不蕴含，无偏不是金标准。（§7.1）

---

## 7.2 矩估计法：最朴素的流水线

### 7.2.1 思想

**矩估计法（method of moments, MoM）** 是历史上最早的系统造估计量方法（K. Pearson, 1894），思路朴素到一句话：**让理论矩等于样本矩，解出参数**。

设分布有 $k$ 个参数 $\theta=(\theta_1,\dots,\theta_k)$。理论的前 $k$ 阶矩 $\mu_j(\theta)=\mathbb E_\theta[X^j]$ 是 $\theta$ 的函数。样本矩 $\hat\mu_j=\frac1n\sum_i X_i^j$ 是数据算得的数。令它们相等

$$\mu_1(\theta)=\hat\mu_1,\quad \mu_2(\theta)=\hat\mu_2,\quad\dots,\quad \mu_k(\theta)=\hat\mu_k,$$

解这个 $k$ 元方程组，得到的解 $\hat\theta_{\mathrm{MoM}}$ 就是矩估计。背后的合法性来自大数定律：样本矩依概率收敛到理论矩（第 6 章），所以只要 $\theta\mapsto(\mu_1,\dots,\mu_k)$ 的逆连续，矩估计就**一致**。

### 7.2.2 例子

**例 1（正态 $\mathcal N(\mu,\sigma^2)$）**。两个参数，用前两阶矩：$\mu_1=\mu$、$\mu_2=\sigma^2+\mu^2$。令等于样本矩，解得

$$\hat\mu=\bar X,\qquad \hat\sigma^2=\hat\mu_2-\bar X^2=\tfrac1n\textstyle\sum_i X_i^2-\bar X^2=\tfrac1n\sum_i(X_i-\bar X)^2.$$

矩估计给出的方差正好是那个**有偏**的 $1/n$ 版本（§7.1.4）。

**例 2（Gamma $\mathrm{Gamma}(k,\theta)$，形状–尺度参数化）**。理论矩 $\mu_1=k\theta$、$\mathrm{Var}=k\theta^2$。由样本均值 $\bar X$ 与样本二阶中心矩 $\hat s^2=\frac1n\sum_i(X_i-\bar X)^2$（与 §7.2.1 样本矩定义一致的「除以 $n$」版本，**不是** §7.1.4 那个除以 $n-1$ 的无偏 $S^2$）反解：$\hat\theta=\hat s^2/\bar X$、$\hat k=\bar X/\hat\theta=\bar X^2/\hat s^2$。（这里 Gamma 的**尺度参数** $\theta$ 与全章「被估参数」的通用符号 $\theta$ 同字母只是传统，此处特指尺度。）这是矩估计的典型甜区——Gamma 的 MLE 要解含 digamma 函数的超越方程，矩估计却给出闭式，常被拿来当 MLE 数值迭代的**初值**。

> **与你的研究的连接（von Mises 的 $\kappa$）**：第 4 章 §4.3.3 估二面角分布 von Mises 的集中度 $\kappa$ 时，用的「平均合向量长度 $\bar R$ 解 $A(\kappa)=I_1(\kappa)/I_0(\kappa)=\bar R$」本质上就是矩估计/MLE（对 von Mises 这个指数族，二者重合，因为充分统计量 $(\cos\theta,\sin\theta)$ 的矩匹配 = MLE，§7.4）。这解释了「为什么圆周统计的核心量是合向量长度」：它是 von Mises 的充分统计量的样本矩。注意这个 $\kappa$ 估计在**小样本下系统性上偏**（见 §7.3.5 的小样本边界），实践常用 Best–Fisher 等**偏差修正**——这把本节与 §7.3.5 的内部线索接上。

### 7.2.3 优缺点

| | 矩估计 MoM | 极大似然 MLE |
|---|---|---|
| 计算 | 常有闭式、极简单 | 常需数值优化 |
| 一致性 | 通常一致 | 正则条件下一致 |
| 效率（方差） | **未必有效**，可能远大于 CR 下界 | 渐近有效（达 CR 下界，§7.5） |
| 是否用满信息 | 只用前 $k$ 阶矩，**丢弃高阶信息** | 用满整个似然 |
| 取值合法性 | 可能跑到参数空间外（如负方差） | 通常留在合法域内 |

> **诚实的边界（矩估计简单但常不够好）**：矩估计的最大问题是**统计效率低**——它只榨取了前几阶矩，扔掉了数据里其余的信息，所以方差通常大于 MLE，达不到 §7.5 的 Cramér–Rao 下界。它还可能给出**越界**的估计（比如某些情形下估出负的方差或大于 1 的概率）。在重尾分布里更糟：若理论矩根本不存在（柯西分布连一阶矩都没有，第 3 章），矩估计直接失效。**所以矩估计的现代角色主要是：(a) 给 MLE 的数值迭代提供一个好初值；(b) 在 MLE 没有闭式且你只要个粗估时救急。** 它是入门概念，不是主力武器。主力是下一节的 MLE。

> **要点**：矩估计 = 令理论矩等于样本矩、解参数。优点是简单、常有闭式（适合做 MLE 初值）；缺点是只用低阶矩、统计效率低、可能越界、对不存在矩的分布失效。一致但通常不有效——它是脚手架，MLE 才是主梁。（§7.2）

---

## 7.3 极大似然估计（MLE）：本章的主梁

### 7.3.1 似然与对数似然

给定 iid 数据 $x_1,\dots,x_n$，把它们的联合密度（或概率）**当成参数 $\theta$ 的函数**，就是 **似然函数（likelihood function）**：

$$L(\theta)=L(\theta\mid x_1,\dots,x_n)=\prod_{i=1}^n p_\theta(x_i).$$

注意视角的翻转：在第 2–6 章里 $p_\theta(x)$ 是「固定 $\theta$、把 $x$ 当变量」的密度；现在我们**固定数据、把 $\theta$ 当变量**。$L(\theta)$ 回答的是「在参数为 $\theta$ 的世界里，看到这批数据的可能性有多大」。

**极大似然估计（maximum likelihood estimation, MLE）** 就是选那个让数据「最不意外」的参数：

$$\hat\theta_{\mathrm{MLE}}=\arg\max_{\theta\in\Theta} L(\theta).$$

因为乘积难求导、且会下溢，几乎总是取对数，最大化 **对数似然（log-likelihood）**：

$$\ell(\theta)=\log L(\theta)=\sum_{i=1}^n \log p_\theta(x_i).$$

$\log$ 单调，所以 $\arg\max$ 不变。求解通常是令 **得分函数（score function）** $s(\theta)=\nabla_\theta\ell(\theta)=\sum_i \nabla_\theta\log p_\theta(x_i)$ 等于零，并检验是极大（Hessian 负定）。

> **CS 读者陷阱（似然不是概率分布）**：$L(\theta)$ 是 $\theta$ 的函数，但**它对 $\theta$ 积分不等于 1**，它不是 $\theta$ 的概率密度。「在 $\theta$ 上对似然归一化」这件事只有在你乘上先验、转到贝叶斯（第 8 章）后才合法。频率派里 $\theta$ 没有概率，似然只是一个「拟合优度」的曲面，MLE 就是去找它的峰。把似然的峰高之比当成「$\theta$ 取值之比的概率」是越界的。**似然比**（两个 $\theta$ 值、或两个假设下的 $L$ 之比）本身是个合法量，但它是**检验统计量**（似然比检验，第 9 章），**不是 $\theta$ 的概率**；只有乘上先验、在 $\theta$ 上**积分**（得到边缘似然 → 贝叶斯因子/后验，第 8 章），才谈得上「$\theta$ 的概率」。别把「比两点的似然」和「对似然归一化成 $\theta$ 的密度」混为一谈。

### 7.3.2 三个标准例子

**例 1（伯努利 / 二项）**。$X_i\sim\mathrm{Bernoulli}(p)$，$\sum x_i=k$ 个成功。

$$\ell(p)=k\log p+(n-k)\log(1-p),\quad \ell'(p)=\frac{k}{p}-\frac{n-k}{1-p}=0\ \Rightarrow\ \boxed{\hat p=\frac{k}{n}}.$$

成功比例。把它推到多项分布（multinomial），$X$ 取 20 类、第 $a$ 类计数 $n_a$，MLE 是

$$\hat p(a)=\frac{n_a}{n}\quad(\text{经验频率}).$$

**这就是「MSA 某列氨基酸频率 = MLE」的全部出处**（信息论 ch2 §2.8、§7.1.3）。

**例 2（正态 $\mathcal N(\mu,\sigma^2)$，两参数）**。

$$\ell(\mu,\sigma^2)=-\frac n2\log(2\pi\sigma^2)-\frac1{2\sigma^2}\sum_i(x_i-\mu)^2.$$

对 $\mu$ 求偏导令零得 $\hat\mu=\bar X$；代回再对 $\sigma^2$ 求导令零得 $\hat\sigma^2=\frac1n\sum_i(x_i-\bar X)^2$。与矩估计一致（§7.2.2），方差是有偏的 $1/n$ 版本。

**例 3（泊松 $\mathrm{Poisson}(\lambda)$）**。$\ell(\lambda)=\sum_i(x_i\log\lambda-\lambda)-\sum\log x_i!$，令导数零得 $\hat\lambda=\bar X$。

> **给数学/CS 读者的视角（MLE 在指数族里就是矩匹配）**：第 4 章 §4.5 已经点过，对指数族 $p_\theta(x)=h(x)\exp(\theta^\top T(x)-A(\theta))$，对数似然是 $\theta^\top\big(\sum_i T(x_i)\big)-nA(\theta)+\text{const}$，求导令零得
> $$\nabla A(\hat\theta)=\frac1n\sum_i T(x_i),\quad\text{即}\quad \mathbb E_{\hat\theta}[T(X)]=\overline{T}.$$
> **MLE = 让模型的充分统计量期望等于数据的经验充分统计量 = 矩匹配（moment matching）**。又因 $A$ 凸（§7.5 会看到 $\nabla^2A=\mathrm{Cov}(T)=$ Fisher 信息），这是个**凸优化**，解唯一（极小参数化下）。上面三个例子全是它的特例：伯努利的 $T=x$，矩匹配就是 $\hat p=\bar x$；正态的 $T=(x,x^2)$，矩匹配就是均值方差对上。这条「MLE=矩匹配」是把 §7.2 矩估计与 MLE 在指数族上**统一**起来的桥——但注意只在指数族里二者重合，一般分布它们不同。

### 7.3.3 MLE 与信息论：最小化交叉熵 = 最小化到经验分布的 KL

这是本章与信息论课程最重要的一次握手，必须说精确。把对数似然除以 $n$（不改变 $\arg\max$）：

$$\frac1n\ell(\theta)=\frac1n\sum_{i=1}^n\log p_\theta(x_i)=\sum_{x}\hat p_n(x)\log p_\theta(x)=-H(\hat p_n,p_\theta),$$

其中 $\hat p_n$ 是数据的**经验分布**（empirical distribution，信息论 ch2 §2.8），$H(\hat p_n,p_\theta)=-\sum_x\hat p_n(x)\log p_\theta(x)$ 是经验分布对模型的**交叉熵（cross-entropy）**。于是

$$\arg\max_\theta\ \frac1n\ell(\theta)=\arg\min_\theta\ H(\hat p_n,p_\theta).$$

再用 KL 与交叉熵的关系 $H(\hat p_n,p_\theta)=H(\hat p_n)+D_{\mathrm{KL}}(\hat p_n\,\|\,p_\theta)$，而 $H(\hat p_n)$ 与 $\theta$ 无关：

> **定理（MLE 的信息论身份）。** （离散情形最干净；连续情形用微分交叉熵/微分 KL、把求和换成积分、经验分布取 $\frac1n\sum_i\delta_{x_i}$ 的弱意义，结论同样成立。）
> $$\boxed{\ \text{极大似然}\ \Longleftrightarrow\ \text{最小化交叉熵}\ H(\hat p_n,p_\theta)\ \Longleftrightarrow\ \text{最小化 KL}\ D_{\mathrm{KL}}(\hat p_n\,\|\,p_\theta).\ }$$
> 三者只差一个与 $\theta$ 无关的常数 $H(\hat p_n)$。MLE 就是**让模型分布 $p_\theta$ 去逼近经验分布 $\hat p_n$**（在 KL 意义下）。完整推导与前/反向 KL 的几何见信息论课程 ch4 §4.7（§4.7.2 从经验分布出发逐步推导、§4.7.3 前向 KL 覆盖 vs 反向 KL 寻峰）。

这与信息论课程 ch2 §2.8.2 的「三句话同义反复」**逐字呼应**——那里从信息论侧讲，这里从估计论侧讲，是同一枚硬币的两面。（本课后文大量用连续模型——正态、von Mises、Potts 的连续近似——做 MLE，这条身份对它们一样成立，只是把离散求和读成积分。）它也解释了深度学习里「交叉熵损失（cross-entropy loss）」的来历：最小化交叉熵损失 = 做 MLE。蛋白质语言模型（如 ESM）训练时逐残基的交叉熵损失，正是在对氨基酸的条件分布做 MLE。

> **与你的研究的连接（为什么频率就是答案）**：上面这条让「MSA 列频率 = MLE」有了第二重解释——经验分布 $\hat p_n$ 本身就是「在所有可能的 $p_\theta$ 里离它 KL 最近（其实是零）的那个」。当你的模型族足够丰富（能取到任意离散分布）时，MLE 就退化成经验分布本身；当模型族有约束（比如 Potts 模型只能匹配单点+成对频率），MLE 给的是「在约束下离经验分布 KL 最近」的那个，这正是最大熵/矩匹配（信息论 ch10、本课 §7.4）。

### 7.3.4 不变性（invariance）：MLE 的独门好性质

> **定理（MLE 的不变性）。** 若 $\hat\theta$ 是 $\theta$ 的 MLE，$g$ 是任意函数，则 $g(\hat\theta)$ 是 $g(\theta)$ 的 MLE。

一句话证明（$g$ 可逆时）：$\eta=g(\theta)$ 是重参数化，似然作为曲面没变，只是横轴换了刻度，峰还在同一点对应的 $g(\hat\theta)$。$g$ 不可逆时用「诱导似然」 $L^*(\eta)=\max_{\theta:g(\theta)=\eta}L(\theta)$ 也能证。

这条性质极其好用，也是 MLE 区别于很多其它估计法的招牌。例如：你估了正态的方差 $\hat\sigma^2$，那么标准差的 MLE 就**直接是** $\hat\sigma=\sqrt{\hat\sigma^2}$，不用重做优化；你估了二项的 $\hat p$，那么 odds 比 $\frac{p}{1-p}$ 的 MLE 就是 $\frac{\hat p}{1-\hat p}$。

> **诚实的边界（不变性不保无偏）**：不变性只对 MLE 本身成立，**不保留无偏性**。即使 $\hat\theta$ 无偏，$g(\hat\theta)$ 一般**有偏**（由 Jensen 不等式，第 3 章：对非线性 $g$，$\mathbb E[g(\hat\theta)]\ne g(\mathbb E[\hat\theta])$）。典型例子：$\hat\sigma^2$ 的无偏版本 $S^2$ 开根号后 $\sqrt{S^2}$ **不是** $\sigma$ 的无偏估计（向下偏）。**偏的方向由 $g$ 的凸性定（Jensen，第 3 章）**：$g$ 凸则 $\mathbb E[g(\hat\theta)]\ge g(\theta)$（**向上偏**，如 odds $g(p)=p/(1-p)$，见练习 3）、$g$ 凹则 $\mathbb E[g(\hat\theta)]\le g(\theta)$（**向下偏**，如 $\sqrt{\cdot}$ 估 $\sigma$）。所以「不变性」是说「MLE 这个操作可以套函数」，不是说「好性质都跟着套」。这也是为什么很多场合宁要 MLE 的方便，也不强求无偏。

### 7.3.5 MLE 的渐近性质：一致、渐近正态、渐近有效

MLE 真正的统治力来自它的**大样本（渐近）行为**。在一组**正则性条件（regularity conditions）** 下（见下方边界），有：

> **定理（MLE 渐近三件套）。** 设正则性条件成立、$\theta_0$ 是真值，则
> 1. **一致性**：$\hat\theta_n\xrightarrow{P}\theta_0$。
> 2. **渐近正态（asymptotic normality）**：
> $$\sqrt n\,(\hat\theta_n-\theta_0)\ \xrightarrow{d}\ \mathcal N\big(0,\ I_1(\theta_0)^{-1}\big),$$
> 其中 $I_1(\theta_0)$ 是**单个样本**的 Fisher 信息（§7.5）。等价地 $\hat\theta_n\approx\mathcal N\big(\theta_0,\ \frac1{nI_1(\theta_0)}\big)=\mathcal N\big(\theta_0, I_n(\theta_0)^{-1}\big)$。
> 3. **渐近有效（asymptotic efficiency）**：它的渐近方差 $I_1^{-1}/n$ **达到** Cramér–Rao 下界（§7.5）——没有别的（正则）一致估计量能渐近地比它方差更小。

直觉证明骨架（标量情形）：得分函数在真值处一阶 Taylor 展开 $0=s(\hat\theta)\approx s(\theta_0)+s'(\theta_0)(\hat\theta-\theta_0)$，整理得 $\sqrt n(\hat\theta-\theta_0)\approx\frac{\frac1{\sqrt n}s(\theta_0)}{-\frac1n s'(\theta_0)}$。分子里 $\frac1{\sqrt n}s(\theta_0)=\frac1{\sqrt n}\sum_i\partial_\theta\log p_{\theta_0}(x_i)$ 是 iid 零均值项之和（得分的期望为零，§7.5），由 CLT（第 6 章）收敛到 $\mathcal N(0,I_1)$；分母 $-\frac1n s'(\theta_0)\xrightarrow{P}I_1$（大数定律 + Fisher 信息的二阶导定义）。商即 $\mathcal N(0,I_1^{-1})$。**这就是为什么 Fisher 信息 $I_1$ 同时出现在分子的方差和分母里——它两头都是它**，最后净留下 $I_1^{-1}$。（这只是**能记住的骨架**：它默认了 $\hat\theta\to\theta_0$、余项可忽略、$-\frac1n s'\to I_1$。严格版需要先**独立证一致性**，再对 $s'(\cdot)$ 在 $\hat\theta$ 与 $\theta_0$ 之间用**中值定理**、并控制三阶导余项一致有界——细节见 §7.10 的 van der Vaart。）

这条定理是频率派统计的皇冠：它说「样本足够多时，MLE 是渐近最优的、且它的误差是高斯的、误差大小由 Fisher 信息精确刻画」。第 9 章构造置信区间、做 Wald 检验，全靠这条正态近似。

> **诚实的边界（渐近 ≠ 现实；正则条件不是摆设）**：这套漂亮结果有一长串前提，违反任何一条都可能翻车——
> 1. **正则性条件**：真值 $\theta_0$ 在参数空间**内部**（不在边界）；密度对 $\theta$ 光滑可微、支撑集不依赖 $\theta$（均匀分布 $\mathrm{Unif}(0,\theta)$ 违反这条，其 MLE $\hat\theta=\max_i X_i$ 收敛快得多、分布不是正态而是指数型）；Fisher 信息存在且非零；可微与积分可交换。
> 2. **边界参数**：若真值在参数空间边界（如方差为 0、混合权重为 0、$\kappa$ 在 0），渐近正态**失效**，分布会被边界「截断」成半正态之类。这在折叠里很常见：估某个耦合是否为零、某个混合分量是否存在，都踩边界。
> 3. **小样本不一定好**：MLE 的优良性是**渐近**的。小 $n$ 下 MLE 可能有可观偏差（正态方差的 $\frac{n-1}{n}$、von Mises 的 $\kappa$ 上偏都是例子），甚至不存在或不唯一。「MLE 最优」是关于 $n\to\infty$ 的承诺，不是对你手里那 30 条序列的承诺。
> 4. **模型设定错误（misspecification）**：若真分布根本不在你的模型族里，MLE 收敛到的是「KL 意义下离真分布最近的那个模型参数」（伪真值 pseudo-true value），渐近方差也要换成更复杂的「三明治」形式。**MLE 只在模型正确时才渐近有效。**

> **要点**：MLE = 最大化对数似然 = 最小化交叉熵 = 最小化到经验分布的 KL（信息论 ch2 §2.8）。求解令得分为零；指数族里 MLE = 矩匹配（§7.3.2）。招牌性质：**不变性** $g(\hat\theta)$ 是 $g(\theta)$ 的 MLE（但不保无偏）。渐近三件套：一致、渐近正态 $\sqrt n(\hat\theta-\theta_0)\to\mathcal N(0,I_1^{-1})$、渐近有效（达 CR 下界）。代价是一串正则条件——边界参数、小样本、模型设错都会让光环失效。（§7.3）

---

## 7.4 充分统计量与因子分解；Rao–Blackwell

### 7.4.1 充分统计量：把数据无损压缩成对 θ 有用的部分

**充分统计量（sufficient statistic）** 是这一思想的精确化：**一个统计量 $T(X)$ 充分，如果给定 $T(X)$ 之后，数据 $X$ 的剩余部分不再含任何关于 $\theta$ 的信息。** 形式定义：$T$ 充分当且仅当条件分布 $p_\theta(x\mid T(x)=t)$ **不依赖 $\theta$**。

直觉：充分统计量是「为了估 $\theta$，数据里你真正需要保留的全部」。一旦算出 $T$，原始数据就可以扔了——丢掉它们不损失任何关于 $\theta$ 的信息。例如估正态的 $(\mu,\sigma^2)$，你不需要记住 $n$ 个原始数据，只需 $\big(\sum x_i,\sum x_i^2\big)$（或等价地样本均值与样本方差）——这两个数就是充分统计量。

直接验证条件分布太麻烦，幸好有一个判据：

> **定理（Fisher–Neyman 因子分解定理）。** $T(X)$ 是 $\theta$ 的充分统计量，当且仅当联合密度能写成
> $$p_\theta(x)=g\big(T(x),\theta\big)\cdot h(x),$$
> 其中第一个因子只通过 $T(x)$ 依赖数据并含 $\theta$，第二个因子 $h(x)$ 完全不含 $\theta$。

> **诚实的边界（测度论细节）**：严格陈述需要「密度关于某个共同的 $\sigma$-有限测度几乎处处成立」（离散用计数测度、连续用 Lebesgue 测度），$h(x)$ 可吸收与 $\theta$ 无关的归一化；而把充分性定义成「给定 $T$ 的条件分布不依赖 $\theta$」时，连续情形还需要**正则条件分布存在**。本章在「够用」层面陈述（沿用 ch2 标题「严格而够用的地基」），完整测度论处理见 Lehmann–Casella。

用法极简单：写出似然，看 $\theta$ 是通过数据的什么组合 $T(x)$ 进来的，那个 $T(x)$ 就充分。例如伯努利 $L(p)=p^{\sum x_i}(1-p)^{n-\sum x_i}$，$p$ 只通过 $\sum x_i$ 进来，故 $T=\sum x_i$（成功次数）充分——估 $p$ 只需知道总共成功几次，谁在第几次成功无关紧要。

### 7.4.2 与指数族的关系：T(x) 现成就在那

第 4 章 §4.5 的指数族 $p_\theta(x)=h(x)\exp(\theta^\top T(x)-A(\theta))$ 看一眼就懂了：把 $n$ 个 iid 样本相乘，

$$\prod_i p_\theta(x_i)=\Big(\prod_i h(x_i)\Big)\exp\Big(\theta^\top\underbrace{\textstyle\sum_i T(x_i)}_{=:\,T_n(x)}-nA(\theta)\Big).$$

按因子分解定理，**指数族的充分统计量就是 $T_n(x)=\sum_i T(x_i)$**——那个写在指数里的 $T$ 直接现成。这就是为什么第 4 章 §4.5 那张表把 $T(x)$ 叫充分统计量：正态的 $(x,x^2)$、伯努利的 $x$、von Mises 的 $(\cos\theta,\sin\theta)$、Potts 的「单点+成对计数」，都是各自分布「需要记住的全部」。充分统计量的维数（不随 $n$ 增长）正是指数族的标志——这与一般分布形成鲜明对比。

> **诚实的边界（充分统计量维度不随 n 长，这是指数族的特权）**：有一条深刻的定理（Pitman–Koopman–Darmois）：在光滑、支撑集不依赖 $\theta$ 等正则条件下，**唯一一类「存在维度固定（不随样本量 $n$ 增长）的充分统计量」的分布族，就是指数族**。换句话说，非指数族（如均匀分布 $\mathrm{Unif}(0,\theta)$、$t$ 分布、混合分布）要么充分统计量随 $n$ 膨胀、要么没有低维充分统计量——你没法把数据无损压缩成几个数。$\mathrm{Unif}(0,\theta)$ 是个边界例子（它支撑依赖 $\theta$，逃过了定理前提），其充分统计量是 $\max_i x_i$，恰好低维——这正是它 MLE 也反常（§7.3.5）的同一个根源。**所以「能不能把 MSA 无损压成频率表」这件事，等价于问「模型是不是指数族」**——Potts 是，所以你只需统计单点+成对频率；一旦模型不是指数族，这种无损压缩就没了。

### 7.4.3 Rao–Blackwell：用充分统计量改良估计量

充分统计量不只是「能扔数据」，它还能**主动改良**任何估计量：

> **定理（Rao–Blackwell）。** 设 $\tilde\theta$ 是 $\theta$ 的任一无偏估计量，$T$ 是充分统计量。定义改良估计量 $\hat\theta=\mathbb E[\tilde\theta\mid T]$（对 $T$ 取条件期望）。则
> 1. $\hat\theta$ 仍无偏：$\mathbb E[\hat\theta]=\mathbb E[\mathbb E[\tilde\theta\mid T]]=\mathbb E[\tilde\theta]=\theta$（全期望公式）；
> 2. $\hat\theta$ 方差不增：$\mathrm{Var}(\hat\theta)\le\mathrm{Var}(\tilde\theta)$。

证明用第 3 章 §3.4.3 的全方差公式 $\mathrm{Var}(\tilde\theta)=\mathbb E[\mathrm{Var}(\tilde\theta\mid T)]+\mathrm{Var}(\mathbb E[\tilde\theta\mid T])=\underbrace{\mathbb E[\mathrm{Var}(\tilde\theta\mid T)]}_{\ge0}+\mathrm{Var}(\hat\theta)\ \ge\ \mathrm{Var}(\hat\theta)$。等号当且仅当 $\tilde\theta$ 本来就是 $T$ 的函数。注意第 1 条用到 $T$ 充分——否则 $\mathbb E[\tilde\theta\mid T]$ 可能还依赖 $\theta$、不是个合法的估计量。

含义：**任何无偏估计量，只要对充分统计量取条件期望，方差就只会变小或不变**。所以「最优的无偏估计量」一定是充分统计量的函数——这把寻找最小方差无偏估计（minimum-variance unbiased estimator, MVUE）的范围一举缩到「充分统计量的函数」里（配合 Lehmann–Scheffé 定理与完备性可定出唯一 MVUE，此处不展开）。

> **给数学/CS 读者的视角（条件期望是「最优投影」）**：Rao–Blackwell 本质是「$\mathbb E[\cdot\mid T]$ 是在 $L^2$ 里到『$T$-可测函数子空间』的正交投影」（第 3 章 §3.4.4「条件期望作为均方最优预测（$L^2$ 投影）」那条几何）。投影不增长度（方差是 $L^2$ 范数），所以方差不增。这与「最小二乘 = 投影」「卡尔曼滤波 = 条件期望」是同一个几何（在联合高斯下它退化为第 5 章 §5.3.4 的线性条件均值）。把估计量「投影到充分统计量上」=「扔掉与 $\theta$ 无关的随机噪声」。

> **要点**：充分统计量 $T(X)$ = 数据中关于 $\theta$ 的全部信息的无损浓缩（给定 $T$ 后剩余数据与 $\theta$ 独立）。判据是 Fisher–Neyman 因子分解 $p_\theta=g(T,\theta)h(x)$。指数族的 $T$ 现成写在指数里、且维度不随 $n$ 增长（Pitman–Koopman–Darmois：这是指数族的特权）。Rao–Blackwell：对 $T$ 取条件期望可无偏地降方差，故最优无偏估计必是 $T$ 的函数。（§7.4）

---

## 7.5 Fisher 信息与 Cramér–Rao 下界

### 7.5.1 得分函数与 Fisher 信息

**得分函数（score）** 是对数似然对参数的梯度：$s(\theta)=\partial_\theta\log p_\theta(X)$（单样本，标量参数）。它有个基本恒等式：

> **得分的期望为零**：$\mathbb E_\theta[s(\theta)]=\mathbb E_\theta[\partial_\theta\log p_\theta(X)]=\int\frac{\partial_\theta p_\theta}{p_\theta}p_\theta\,\mathrm dx=\partial_\theta\int p_\theta\,\mathrm dx=\partial_\theta 1=0$（用了「积分微分可交换」的正则条件）。

既然得分均值为零，它的散布（方差）就是一个有意义的量——**Fisher 信息（Fisher information）**：

$$\boxed{\ I(\theta)=\mathrm{Var}_\theta\big(s(\theta)\big)=\mathbb E_\theta\big[(\partial_\theta\log p_\theta(X))^2\big].\ }$$

在正则条件下它还有一个**更常用的等价形式**（负二阶导的期望，即对数似然曲面在真值处的平均曲率）：

$$I(\theta)=-\mathbb E_\theta\big[\partial_\theta^2\log p_\theta(X)\big].$$

两者相等的一行推导：对 $\mathbb E[s]=0$ 再求一次导，$\partial_\theta\int(\partial_\theta\log p_\theta)p_\theta=\int(\partial_\theta^2\log p_\theta)p_\theta+\int(\partial_\theta\log p_\theta)(\partial_\theta p_\theta)=\mathbb E[\partial_\theta^2\log p_\theta]+\mathbb E[(\partial_\theta\log p_\theta)^2]=0$，移项即得。此处再次用到「积分微分可交换」、且**支撑集不依赖 $\theta$**（否则求导会漏掉边界项）——这正是 §7.3.5/§7.5.3 里 $\mathrm{Unif}(0,\theta)$ 让 Fisher 信息/CRLB 失效的**同一条正则条件**。

**直觉**：Fisher 信息度量「对数似然曲面在真值处有多尖」。曲面越尖（曲率 $-\partial_\theta^2\ell$ 越大），数据对 $\theta$ 的位置越敏感、越「认得出」真值，信息越多、能估得越准；曲面越平，$\theta$ 移一点似然几乎不变，数据「说不清」$\theta$ 是多少，信息越少。

> **给数学/CS 读者的视角（Fisher 信息 = 局部 KL 曲率 = 信息几何度规）**：把模型分布 $p_\theta$ 和邻近的 $p_{\theta+\delta}$ 用 KL 散度比较，做二阶 Taylor 展开会得到 $D_{\mathrm{KL}}(p_\theta\,\|\,p_{\theta+\delta})\approx\frac12 I(\theta)\delta^2$（一阶项为零，因为 $\theta$ 是 KL 的极小）。所以 **Fisher 信息正是参数空间上「KL 散度诱导的局部度规」**——它把参数空间变成一个黎曼流形，这就是 **信息几何（information geometry）** 的起点。在指数族里它正好是 $I(\theta)=\nabla^2 A(\theta)=\mathrm{Cov}_\theta(T)$（第 4 章 §4.5.2 的 $A$ 的 Hessian），把「$A$ 凸」「充分统计量协方差」「Fisher 信息」「KL 曲率」四件事缝成一件。自然梯度（natural gradient）、实验设计（D-/A-最优）也都建在 $I(\theta)$ 上。信息论课程 ch10 的最大熵–指数族框架与这里是同一片地。

### 7.5.2 可加性与多样本

Fisher 信息对独立样本**可加**：$n$ 个 iid 样本的总信息

$$I_n(\theta)=n\,I_1(\theta).$$

证明一行：总得分是 $n$ 个**独立同分布**的单样本得分之和，故总方差 $=n\times$ 单样本得分方差 $=nI_1$（**独立**给「可加」、**同分布**给「相等」，两者缺一不可）。这条可加性是 §7.3.5 里渐近方差 $\frac1{nI_1}$ 的来源——**样本越多、信息线性累加、估计方差线性下降**。它也立刻给出「需要多少数据」的标尺：要把估计的标准差减半，得把样本量翻 4 倍（因为 $\mathrm{sd}\propto 1/\sqrt n$）。

### 7.5.3 Cramér–Rao 下界：无偏估计方差的物理下限

> **定理（Cramér–Rao 下界, CRLB）。** 在正则条件下，任何**无偏**估计量 $\hat\theta$（基于 $n$ 个 iid 样本）满足
> $$\boxed{\ \mathrm{Var}_\theta(\hat\theta)\ \ge\ \frac1{I_n(\theta)}=\frac1{n\,I_1(\theta)}.\ }$$
> 更一般地，对（可能有偏的）估计量，在同款正则条件下，下界是 $\dfrac{[1+b'(\theta)]^2}{I_n(\theta)}$，其中 $b(\theta)=\mathrm{Bias}(\hat\theta)$。注意分子 $[1+b'(\theta)]^2$：当估计量**向真值收缩**（$b'(\theta)<0$）时分子 $<1$，整个下界因此**低于**无偏的 $1/I_n$——这正是有偏估计（岭、收缩）方差能低于无偏 CRLB 的机制来源，与下方诚实边界 (1) **一致而非矛盾**。

证明骨架：得分 $s$ 与 $\hat\theta$ 的协方差 $\mathrm{Cov}(\hat\theta,s)=\mathbb E[\hat\theta\,s]-\underbrace{\mathbb E[\hat\theta]\mathbb E[s]}_{=0}=\partial_\theta\mathbb E[\hat\theta]=\partial_\theta\theta=1$（无偏时），再用 Cauchy–Schwarz $1=\mathrm{Cov}(\hat\theta,s)^2\le\mathrm{Var}(\hat\theta)\mathrm{Var}(s)=\mathrm{Var}(\hat\theta)\cdot I_n$，移项即得。

**这是一条信息论式的不可能定理**：它说不管你多聪明地设计无偏估计量，方差都不可能低过 $1/I_n(\theta)$——信息量给方差设了一个不可逾越的地板。达到下界的（无偏）估计量称为**有效（efficient）**。一个估计量的**效率（efficiency）** 定义为 $\mathrm{eff}(\hat\theta)=\frac{1/I_n}{\mathrm{Var}(\hat\theta)}\in(0,1]$，等于 1 即有效。§7.3.5 说 MLE **渐近**有效，正是说它在大样本下方差渐近地贴着这条地板。

> **诚实的边界（CRLB 不是万能下界）**：(1) CRLB 是**无偏**估计的下界。**有偏估计量可以方差更小**——这不矛盾，因为它牺牲了无偏（再看一眼 §7.1 的 MSE 分解：有偏估计可能 MSE 更小）。岭回归就是故意有偏、方差远低于无偏最小二乘的例子。(2) CRLB 要正则条件（与 MLE 同款）：支撑集不依赖 $\theta$ 时才成立，$\mathrm{Unif}(0,\theta)$ 又一次违反，它的 MLE 方差实际上**低于**形式上套出来的 CRLB（因为前提不满足，公式根本不适用）。(3) 下界**未必能达到**：很多模型不存在达到 CRLB 的无偏估计量，此时 MLE 也只是**渐近**贴近、有限样本下达不到。**所以「MLE 最优」要全部限定词：无偏类内、正则条件下、渐近意义上。**

> **与你的研究的连接（需要多少条同源序列？）**：这是 Fisher 信息在折叠研究里最实在的应用。估 MSA 某列氨基酸 $a$ 的频率 $p$，单样本 Fisher 信息是 $I_1(p)=\frac1{p(1-p)}$（伯努利），故 CRLB 给 $\mathrm{Var}(\hat p)\ge\frac{p(1-p)}{n}$——要把 $\hat p$ 的标准差压到 $0.02$，需 $n\gtrsim\frac{p(1-p)}{0.02^2}$，对 $p\approx0.5$ 约 625 条**有效独立**序列。关键词「有效独立」：MSA 里的序列由共同祖先沿系统发育树相关（非 iid），真实有效样本数 $n_{\mathrm{eff}}$ 远小于序列条数（信息论 ch2 §2.4.3、§2.7 Hoeffding 那条已点过），所以 Fisher 信息累加 $I_n=n_{\mathrm{eff}}I_1$ 里的 $n$ 要换成 $n_{\mathrm{eff}}$——这就是「浅 MSA 共进化信号不可靠」的**定量根源**，也是 DCA/plmDCA 要做**序列重加权（reweighting）** 和正则化的理由（折叠课程第 5 章、§7.7）。对 Potts 模型估每对耦合，要估的参数有 $O(L^2\cdot 20^2)$ 个，而有效独立序列数 $n_{\mathrm{eff}}$ 大致固定（受 MSA 深度与系统发育冗余限制），于是「待估参数数 $O(L^2\cdot 20^2)$ vs $n_{\mathrm{eff}}$」之比直接决定能不能估准——这是 AlphaFold 之前接触预测精度受限于 MSA 深度的统计学解释。

> **要点**：得分 $s=\partial_\theta\log p_\theta$ 均值为零，其方差 = Fisher 信息 $I(\theta)=\mathbb E[s^2]=-\mathbb E[\partial_\theta^2\log p_\theta]$ = 对数似然曲面的平均曲率 = 局部 KL 度规（信息几何）。指数族里 $I=\nabla^2A=\mathrm{Cov}(T)$。对 iid 可加 $I_n=nI_1$。**Cramér–Rao 下界** $\mathrm{Var}(\hat\theta)\ge1/I_n$ 给无偏估计方差一条地板，MLE 渐近达到它。但下界只对无偏类、正则条件下成立，有偏估计可更小。它直接回答「需要多少有效独立序列」。（§7.5）

---

## 7.6 EM 算法：隐变量模型的 MLE 引擎

### 7.6.1 问题：隐变量让似然算不动

前面 MLE 的例子里似然都好写、好求导。但真实模型常有**隐变量（latent / hidden variable）** $Z$——观测到 $X$，但生成机制里还有看不见的 $Z$，似然要把 $Z$ 边缘掉：

$$\ell(\theta)=\sum_i\log p_\theta(x_i)=\sum_i\log\sum_{z}p_\theta(x_i,z).$$

那个 **$\log\sum$**（对数套着求和/积分）是万恶之源：它不再拆成各项之和，求导后耦合在一起，没有闭式解。两个折叠/生信的核心例子：

- **高斯混合模型（Gaussian mixture model, GMM）**：$Z$ 是「这一帧属于哪个构象盆地」的隐标签，$X$ 是观测坐标。第 5 章 §5.6 说真实构象分布多峰、要用 GMM，正是这里。
- **profile 隐马尔可夫模型（profile HMM）**：$Z$ 是「每个残基比对到模型的哪个 match/insert/delete 状态」的隐路径，$X$ 是观测序列。HMMER、Pfam 用它做远缘同源检测与多序列比对，其参数估计正是下面要讲的 EM 特例 **Baum–Welch**。

### 7.6.2 ELBO：用 Jensen 给 log∑ 造一个可优化的下界

对任意一个关于隐变量的分布 $q(z)$（待定），用 $\log$ 的凹性 + Jensen 不等式（第 3 章）把 $\log\sum$ 压成 $\sum\log$：

$$\log p_\theta(x)=\log\sum_z q(z)\frac{p_\theta(x,z)}{q(z)}\ \ge\ \sum_z q(z)\log\frac{p_\theta(x,z)}{q(z)}\ =:\ \mathcal F(q,\theta).$$

$\mathcal F(q,\theta)$ 叫 **证据下界（evidence lower bound, ELBO）**。一个关键恒等式把「真似然」与「下界」的差精确写成 KL：

$$\boxed{\ \log p_\theta(x)=\mathcal F(q,\theta)+D_{\mathrm{KL}}\big(q(z)\ \big\|\ p_\theta(z\mid x)\big).\ }$$

因为 KL $\ge0$，确认了 $\mathcal F\le\log p_\theta(x)$，且**等号当且仅当 $q(z)=p_\theta(z\mid x)$**（隐变量的后验）。这给出一个交替上升的策略：固定 $\theta$ 把 $q$ 调到使下界最紧（E 步），再固定 $q$ 把 $\theta$ 推高下界（M 步）。

### 7.6.3 算法与单调上升

> **EM 算法（expectation–maximization）**。从初值 $\theta^{(0)}$ 开始，重复：
> - **E 步（expectation）**：用当前参数算隐变量后验 $q^{(t)}(z)=p_{\theta^{(t)}}(z\mid x)$。这让 ELBO 在 $\theta^{(t)}$ 处**贴紧**真似然（KL=0）。
> - **M 步（maximization）**：最大化期望完全数据对数似然
> $$\theta^{(t+1)}=\arg\max_\theta\ Q(\theta\mid\theta^{(t)}),\quad Q(\theta\mid\theta^{(t)})=\mathbb E_{q^{(t)}}\big[\log p_\theta(x,Z)\big].$$
> （$\mathcal F$ 中与 $\theta$ 无关的 $-\sum q\log q$ 项可丢，故 M 步只需最大化 $Q$。）

> **定理（EM 单调上升）。** 每次迭代真似然不减：$\log p_{\theta^{(t+1)}}(x)\ge\log p_{\theta^{(t)}}(x)$。

证明三行：
$$\log p_{\theta^{(t+1)}}\ \overset{(a)}{\ge}\ \mathcal F(q^{(t)},\theta^{(t+1)})\ \overset{(b)}{\ge}\ \mathcal F(q^{(t)},\theta^{(t)})\ \overset{(c)}{=}\ \log p_{\theta^{(t)}}.$$
(a) ELBO 永远是下界；(b) M 步定义就是在 $\theta$ 上最大化 $\mathcal F(q^{(t)},\cdot)$，所以 $\theta^{(t+1)}$ 处不小于 $\theta^{(t)}$ 处；(c) E 步让 $q^{(t)}=p_{\theta^{(t)}}(z\mid x)$，KL=0，下界贴紧。**单调上升的序列若有上界则收敛**；当似然**有上界**（离散模型自动满足；连续模型需对方差加下界或正则化/先验把似然封顶）时，EM 收敛到驻点。但要小心：对**连续**密度对数似然**不一定有上界**——$p_\theta(x)$ 可以 $>1$（高斯在小方差时密度峰值远大于 1，故 $\log p_\theta(x)$ 可为正，这正是第 4 章 §4.2.1 反复强调的「pdf 可 $>1$ ⇒ log-pdf 可 $>0$」），而无约束高斯混合的对数似然甚至可发散到 $+\infty$（某分量 $\sigma^2\to0$ 贴住单点，见 §7.6.4 退化解），此时单调上升的序列被吸入退化尖峰、**并不收敛**。所以「单调有界故收敛」必须以「似然有上界」为前提，这正是连续混合模型**必须防塌缩**（加方差下界/正则化）的另一面。

这个「造下界—顶上去—重造更高的下界—再顶」的图景，正是变分推断（variational inference）、EM、甚至扩散模型 ELBO 训练的**共同骨架**，与信息论里的「最大熵/KL 几何」是一家——把它看熟，现代生成模型的训练目标你一眼就能认出。

```
   logp(x) ──真似然（想最大化，但 log∑ 算不动）
      │
   ┌──┴──────────────── E 步：q←p(z|x)，下界贴紧真似然（KL=0）
   │  ╱ ELBO F(q,θ)  ← 此刻 F = logp
   │ ╱                 M 步：固定 q，把 θ 推到 F 的峰 → logp 必然↑
   │╱
   └────────────────────► θ
      θ⁽ᵗ⁾    θ⁽ᵗ⁺¹⁾      （单调爬升，收敛到驻点）
```

### 7.6.4 具体例子：两分量高斯混合

数据 $x_1,\dots,x_n\in\mathbb R$，模型 $p(x)=\pi\,\mathcal N(x\mid\mu_1,\sigma_1^2)+(1-\pi)\,\mathcal N(x\mid\mu_2,\sigma_2^2)$，隐变量 $z_i\in\{1,2\}$ 是分量标签。

- **E 步（算责任 responsibility）**：$\gamma_i=p(z_i{=}1\mid x_i)=\dfrac{\pi\,\mathcal N(x_i\mid\mu_1,\sigma_1^2)}{\pi\,\mathcal N(x_i\mid\mu_1,\sigma_1^2)+(1-\pi)\,\mathcal N(x_i\mid\mu_2,\sigma_2^2)}$。这是「软分配」——每个点按概率归属两个分量。
- **M 步（加权 MLE）**：
$$\mu_1=\frac{\sum_i\gamma_i x_i}{\sum_i\gamma_i},\quad \sigma_1^2=\frac{\sum_i\gamma_i(x_i-\mu_1)^2}{\sum_i\gamma_i},\quad \pi=\frac1n\sum_i\gamma_i,$$
分量 2 用 $1-\gamma_i$ 对称地算。**记软计数（soft count）$N_1=\sum_i\gamma_i$、$N_2=\sum_i(1-\gamma_i)$**，则 $\mu_k,\sigma_k^2$ 就是以责任为权的加权均值/方差、$\pi=N_1/n$，而 $N_1+N_2=n$ 是一个顺手的**归一检验**（实现练习 5 时最容易在这一步的归一上犯错）。M 步就是「用责任当权重做普通高斯 MLE」——隐变量一旦（软）知道，似然又拆开了。

> **与你的研究的连接（构象聚类、二面角混合、profile HMM）**：(1) 把 MD 轨迹的某个内坐标（如一个关键二面角、或一对残基距离）用 GMM/von Mises 混合（第 4 章 §4.3.4）拟合，EM 自动把帧软分配到「折叠态/未折叠态/中间态」各个盆地，分量均值就是盆地中心、权重就是占据概率——这是第 5 章 §5.6「单高斯不够、要混合」的算法落地。(2) **profile HMM 的 Baum–Welch 就是 EM**：E 步用前向–后向算法（forward–backward）算每个残基处于各隐状态的后验概率（责任），M 步用这些后验当权重重估发射概率（每个位置的氨基酸分布）与转移概率。这是 Pfam/HMMER 建库、远缘同源检测、做 MSA 的核心引擎（折叠课程第 5 章）。(3) 共进化里的某些隐变量模型、序列权重的迭代估计也带 EM 味道。一句话：**凡是「数据 + 看不见的归属/路径」的 MLE，引擎多半是 EM。**

> **诚实的边界（EM 只保证爬到局部最优，且会塌缩）**：(1) EM 单调上升，但只收敛到似然的**驻点（局部极大或鞍点）**，**不保证全局最优**——混合模型的似然曲面常有很多局部峰。实践必须**多个随机初值跑、取似然最高的**，或用 k-means 给好初值。(2) GMM 的似然在「某分量方差 $\to0$、均值贴住单个数据点」时**发散到 $+\infty$**（退化解 degenerate solution），EM 可能爬向这种无意义的尖峰——要加方差下界/正则化/贝叶斯先验（第 8 章）防塌缩。**正因如此，§7.6.3 的「单调有界故收敛」需要「似然有上界」这一前提——无约束 GMM 不满足它，必须先加方差下界/先验把似然封顶，单调上升才真正给出收敛。**(3) **分量数 $K$ 是超参数**，EM 不会替你选——要用 BIC/交叉验证/狄利克雷过程等另选（第 8 章模型比较）。(4) 收敛可能很慢（似然曲面平坦时）。所以 EM 是稳健好用的「下界爬升器」，但它的解依赖初值、需要防退化、模型阶数要另定——别把「单调上升」误读成「找到了真参数」。

> **要点**：隐变量让似然变成算不动的 $\log\sum$。EM 用 Jensen 造 ELBO 下界 $\mathcal F(q,\theta)$，恒等式 $\log p=\mathcal F+D_{\mathrm{KL}}(q\|p(z\mid x))$ 表明 $q=$ 后验时下界贴紧。E 步置 $q=$ 隐变量后验（算责任）、M 步最大化期望完全似然 $Q$（加权 MLE），保证真似然**单调不减**（似然有上界时收敛到驻点；连续混合须加方差下界封顶才有上界）。GMM、profile HMM 的 Baum–Welch 都是它。诚实边界：只到局部最优、依赖初值、会塌缩、阶数要另定。（§7.6）

---

## 7.7 惩罚似然 / 正则化与伪似然：通往贝叶斯与 DCA

### 7.7.1 惩罚似然 = MAP：把岭与 lasso 焊到先验上

纯 MLE 在参数多、数据少时会过拟合（方差爆炸，§7.1）。**惩罚似然 / 正则化（penalized likelihood / regularization）** 在对数似然上减一个惩罚项 $J(\theta)$：

$$\hat\theta=\arg\max_\theta\ \big[\ell(\theta)-\lambda J(\theta)\big].$$

两个最常见的惩罚有干净的贝叶斯身份（这条把频率派正则化与第 8 章贝叶斯 **最大后验估计（maximum a posteriori, MAP）** 完全焊死）：

> **定理（正则化 = 高斯/拉普拉斯先验的 MAP）。** MAP 估计 $\arg\max_\theta[\log p(x\mid\theta)+\log p(\theta)]$ 中，
> - 取**高斯先验** $\theta\sim\mathcal N(0,\tau^2 I)$，则 $\log p(\theta)=-\frac1{2\tau^2}\|\theta\|_2^2+\text{const}$，惩罚是 $\|\theta\|_2^2$ ——这就是 **岭回归 / Tikhonov（ridge / $L_2$）**，$\lambda=\frac1{2\tau^2}$。
> - 取**拉普拉斯先验** $\theta_j\sim\mathrm{Laplace}(0,b)$，则 $\log p(\theta)=-\frac1b\|\theta\|_1+\text{const}$，惩罚是 $\|\theta\|_1$ ——这就是 **lasso（$L_1$）**，它会把许多系数压成**恰好为零**（稀疏），因为 $L_1$ 球的尖角。（「尖角」图像对应的是**等价的约束形式** $\min\ell\ \text{s.t.}\ \|\theta\|_1\le t$；在这里写的**惩罚形式**下，稀疏等价地来自 $|\cdot|$ 在 $0$ 处的**次梯度**非光滑，使 KKT 最优解可以恰好落在 $\theta_j=0$ 的坐标轴上。两种形式等价、图像不同。）

一句话总结这张「翻译表」：**频率派眼里的「正则强度 $\lambda$」就是贝叶斯眼里的「先验集中度（先验方差的倒数）」**。$\lambda\to0$（先验极宽、无信息）退回纯 MLE；$\lambda\to\infty$（先验极窄）把参数钉死在先验中心。第 1 章/§7.1 说的「偏差换方差」在这里有了精确旋钮：$\lambda$ 越大、偏差越大、方差越小。

> **与你的研究的连接（伪计数 = Dirichlet 先验的 MAP，再次相遇）**：§7.1.3 的 MSA 伪计数 $\hat p(a)=\frac{n_a+\alpha}{n+20\alpha}$ 此刻露出第三重身份——它正是「多项似然 + Dirichlet 先验 $\mathrm{Dir}(\alpha,\dots,\alpha)$」的 MAP/后验（第 4 章 §4.6、第 8 章）。所以「加伪计数」= 「正则化频率估计」= 「贝叶斯 MAP」，三种语言说同一件事。在 DCA/plmDCA 里，对 Potts 模型的耦合参数 $J_{ij}$ 加 $L_2$ 惩罚（高斯先验）是标准操作，它既防过拟合（耦合参数 $O(L^2\cdot20^2)$ 个、有效序列才几百几千），又让优化数值稳定——这正是第 5 章 §5.4.4「$n<d$ 必须正则化」在离散 Potts 上的体现。

### 7.7.2 伪似然：当配分函数算不动时

有一类模型，真似然根本写不出来——不是不知道形式，而是**归一化常数（配分函数 partition function）算不动**。最重要的例子就是折叠研究的核心：**Potts 模型**（第 4 章 §4.5、第 5 章 §5.4、信息论 ch10 §10.3）

$$p(\mathbf a)=\frac1Z\exp\Big(\sum_i h_i(a_i)+\sum_{i<j}J_{ij}(a_i,a_j)\Big),\quad Z=\sum_{\mathbf a\in\{1,\dots,20\}^L}\exp(\cdots).$$

那个 $Z$ 是对 $20^L$ 个序列构型求和——$L=100$ 就是 $20^{100}$ 项，**计算上 #P-难，永远算不动**（第 4 章 §4.5.2 的诚实边界）。没有 $Z$ 就没有似然，MLE 直接卡死。

**伪似然（pseudo-likelihood）**（Besag, 1975）是绕过 $Z$ 的经典招数：不去算联合似然，而是把每个位点**在给定其余所有位点下的条件似然**乘起来当替代目标：

$$\ell_{\mathrm{PL}}(\theta)=\sum_{\text{序列}}\sum_{i=1}^L\log p_\theta\big(a_i\ \big|\ \mathbf a_{\setminus i}\big).$$

妙处在于：**每个条件分布 $p_\theta(a_i\mid\mathbf a_{\setminus i})$ 的归一化只对那一个位点的 20 种氨基酸求和**（一个 softmax），$Z$ 被彻底消掉了——条件分布里大配分函数约掉、只剩局部归一。于是伪似然可以梯度上升优化。这就是 **plmDCA（pseudo-likelihood maximization DCA）** 的全部内核——共进化接触预测里最准的经典方法之一（被 GREMLIN、CCMpred 等实现），AlphaFold 之前的接触预测主力，至今仍是 MSA 特征/基线。

> **定理（伪似然估计的性质，诚实陈述）。** 在模型设定正确时，伪似然估计是**一致的**（$n\to\infty$ 收敛到真参数）。但它**不是有效的**——它的渐近方差大于真 MLE（达不到 CRLB），因为它丢掉了位点之间的部分联合信息。这是「用统计效率换计算可行性」的经典交易：真 MLE 有效却算不动，伪似然次优但能算。

> **给数学/CS 读者的视角（伪似然 vs 对比散度 vs 变分）**：绕开配分函数 $Z$ 是无向图模型/能量模型（energy-based model）训练的永恒主题，伪似然只是其中一条路。另两条是：(a) **对比散度（contrastive divergence）/ 持续对比散度**——用 MCMC（第 10 章）近似 $\nabla_\theta\log Z=\mathbb E_\theta[T]$ 那个算不动的期望，玻尔兹曼机/RBM 用它；(b) **变分近似**——用可处理的 $q$ 给 $\log Z$ 造界（与 §7.6 的 ELBO 同源）。三者都在回答「指数族似然好写、$A(\theta)=\log Z$ 算不动」这个第 4 章 §4.5.2 留下的根本困难。伪似然胜在简单、凸（每个条件是 softmax 回归/logistic 回归，凸）、并行，所以成了 DCA 的工业标准。**记住这条主线：能量模型的训练 = 想办法搞定那个算不动的 $Z$**——这在折叠的生成模型（折叠课程第 8 章 AlphaFold3 的扩散式生成、第 9 章蛋白质设计的生成模型 RFdiffusion 等）里会反复出现。

> **要点**：惩罚似然 = MAP：$L_2$ 惩罚（岭）= 高斯先验、$L_1$ 惩罚（lasso，稀疏）= 拉普拉斯先验，正则强度 $\lambda$ = 先验集中度，把频率派正则化与第 8 章贝叶斯焊死（伪计数 = Dirichlet 先验 MAP）。**伪似然**用「各位点条件似然之积」绕过算不动的配分函数 $Z$，是 plmDCA 共进化接触预测的内核——一致但不有效，拿统计效率换计算可行。能量模型训练的核心难题永远是那个算不动的 $Z$。（§7.7）

---

## 7.8 本章小结

- **估计量是数据的随机函数**，看抽样分布评价它；三把尺子偏差/方差/MSE，核心恒等式 $\mathrm{MSE}=\mathrm{Bias}^2+\mathrm{Var}$ 是 ML 偏差–方差权衡/正则化的母体；$n\to\infty$ 看一致性（MSE→0 即一致），无偏与一致互不蕴含、无偏不是金标准。（§7.1）
- **矩估计** = 令理论矩等于样本矩、解参数：简单、常闭式（好当 MLE 初值），但只用低阶矩、效率低、可能越界、对无矩分布失效——是脚手架不是主梁。（§7.2）
- **MLE** = 最大化对数似然 = 最小化交叉熵 = 最小化到经验分布的 KL（与信息论 ch2 §2.8 逐字呼应）；指数族里 MLE = 矩匹配（充分统计量期望 = 经验充分统计量）。（§7.3）
- **MLE 的不变性**：$g(\hat\theta)$ 是 $g(\theta)$ 的 MLE（但不保无偏，Jensen）；**渐近三件套**：一致、$\sqrt n(\hat\theta-\theta_0)\to\mathcal N(0,I_1^{-1})$、渐近有效（达 CRLB）——代价是正则条件，边界参数/小样本/模型设错都会让光环失效。（§7.3）
- **充分统计量** $T(X)$ = 数据中关于 $\theta$ 的全部信息的无损浓缩，判据是 Fisher–Neyman 因子分解 $p_\theta=g(T,\theta)h(x)$；指数族的 $T$ 现成写在指数里且维度不随 $n$ 增长（Pitman–Koopman–Darmois 是指数族特权）。（§7.4）
- **Rao–Blackwell**：对充分统计量取条件期望可无偏地降方差（条件期望 = $L^2$ 投影），故最优无偏估计必是 $T$ 的函数。（§7.4）
- **Fisher 信息** $I(\theta)=\mathbb E[s^2]=-\mathbb E[\partial_\theta^2\log p_\theta]$ = 对数似然曲率 = 局部 KL 度规（信息几何）；指数族里 $=\nabla^2A=\mathrm{Cov}(T)$；对 iid 可加 $I_n=nI_1$。（§7.5）
- **Cramér–Rao 下界** $\mathrm{Var}(\hat\theta)\ge1/I_n$ 给无偏估计方差一条地板，MLE 渐近达到——但只对无偏类、正则条件下成立（有偏估计可更小），它直接回答「需要多少有效独立序列」。（§7.5）
- **EM 算法**：隐变量让似然成算不动的 $\log\sum$；用 Jensen 造 ELBO，E 步置 $q=$ 后验（算责任）、M 步最大化 $Q$（加权 MLE），真似然单调不减（似然有上界时收敛到**驻点**）；GMM、profile HMM 的 Baum–Welch 都是它；诚实边界是只到局部最优、依赖初值、会塌缩（连续混合须封顶似然才保证收敛）、阶数要另定。（§7.6）
- **惩罚似然 = MAP**（岭=高斯先验、lasso=拉普拉斯先验、伪计数=Dirichlet 先验），把频率派正则化焊到第 8 章贝叶斯上；**伪似然**用条件似然之积绕过算不动的配分函数 $Z$，是 plmDCA 接触预测内核——一致但不有效，拿效率换可行。（§7.7）

---

## 7.9 动手 / 思考小练习

**1（动手·编程，验证偏差–方差与一致性）。** 仅用 `numpy`：从 $\mathcal N(\mu{=}2,\sigma^2{=}9)$ 抽样。(a) 对 $n\in\{5,20,100,1000\}$，各重复 5000 次，每次算方差的两个估计 $\hat\sigma^2_{\mathrm{MLE}}=\frac1n\sum(x_i-\bar x)^2$ 与无偏 $S^2=\frac1{n-1}\sum(x_i-\bar x)^2$。(b) 用蒙特卡洛估各自的偏差、方差、MSE，验证 $\hat\sigma^2_{\mathrm{MLE}}$ 偏差约 $-\sigma^2/n$。(c) **思考**：哪个 MSE 更小？为什么无偏的不一定 MSE 最小？
（*提示*：(b) $\mathbb E[\hat\sigma^2_{\mathrm{MLE}}]=\frac{n-1}{n}\sigma^2$，偏差 $=-\sigma^2/n$，随 $n$→0 故一致（§7.1.4）。(c) 有趣的是 MLE 版本 MSE 常**更小**（甚至除以 $n+1$ 的版本 MSE 最小）——因为它方差略小，省下的方差超过了引入的偏差²。这是 §7.1.2 分解的活教材：无偏不等于 MSE 最优。）

**2（动手·编程，MLE = 最小化到经验分布的 KL，与研究相关）。** 仅用 `numpy`：造一个 $n{\times}L$ 的氨基酸 MSA（整数 0–19）。(a) 对某一列数 20 类计数，算频率估计 $\hat p(a)=n_a/n$，并验证它最大化对数似然 $\sum_a n_a\log p(a)$（在 $\sum p=1$ 约束下，用拉格朗日法手推一遍）。(b) 数值地：对该列经验分布 $\hat p$，在单纯形上随机取若干个 $q$，算 $D_{\mathrm{KL}}(\hat p\|q)$，确认 $q=\hat p$ 时 KL=0 且最小。(c) 加伪计数 $\hat p(a)=\frac{n_a+\alpha}{n+20\alpha}$，观察 $n$ 小时它如何把估计往均匀拉。
（*提示*：(a) 拉格朗日 $\sum_a n_a\log p_a+\lambda(1-\sum p_a)$ 求导得 $n_a/p_a=\lambda$、$p_a\propto n_a$，归一即 $\hat p(a)=n_a/n$（§7.3.2）。(b) 这是 §7.3.3 定理的数值确认：MLE = 最小化 $D_{\mathrm{KL}}(\hat p\|q)$，最小值 0 在 $q=\hat p$。(c) 小 $n$ 时伪计数主导、估计趋均匀（偏差换方差，§7.1.3）——这就是浅 MSA 必须平滑的理由。）

**3（思考·证明，MLE 不变性与无偏不传递）。** (a) 设 $\hat p=k/n$ 是二项 $p$ 的 MLE，写出 odds $\theta=p/(1-p)$ 的 MLE，并说明用了哪条定理。(b) 举例说明：即便 $\hat p$ 无偏，$\widehat{\theta}=\hat p/(1-\hat p)$ 一般**有偏**。
（*提示*：(a) 不变性（§7.3.4）：$\hat\theta=\hat p/(1-\hat p)=k/(n-k)$。(b) $g(p)=p/(1-p)$ 是凸函数，Jensen（第 3 章）给 $\mathbb E[g(\hat p)]\ge g(\mathbb E[\hat p])=g(p)=\theta$，故向上偏，等号仅当 $\hat p$ 退化。这说明不变性传递的是「MLE 身份」，不是无偏（§7.3.4 边界）。）

**4（思考·计算，Fisher 信息与「需要多少序列」，与研究相关）。** 对伯努利 $p$（某氨基酸出现与否）：(a) 推导单样本 Fisher 信息 $I_1(p)=\frac1{p(1-p)}$（两种定义各算一遍验证相等）。(b) 写出 CRLB，并算「要让 $\hat p$ 标准差 $\le0.02$、当 $p=0.3$」所需的有效独立序列数 $n_{\mathrm{eff}}$。(c) 解释为什么实际需要的序列**条数**远大于 $n_{\mathrm{eff}}$。
（*提示*：(a) $\log p_\theta=x\log p+(1-x)\log(1-p)$，$\partial_p^2=-\frac{x}{p^2}-\frac{1-x}{(1-p)^2}$，取 $-\mathbb E$ 得 $\frac1p+\frac1{1-p}=\frac1{p(1-p)}$（§7.5.1）。(b) CRLB $\mathrm{Var}\ge\frac{p(1-p)}{n}$，要 $\sqrt{\frac{0.21}{n}}\le0.02$ 得 $n_{\mathrm{eff}}\ge\frac{0.21}{0.0004}\approx525$。(c) MSA 序列由系统发育相关、非 iid，有效样本数被打折，故真实需要的条数远多于 525（§7.5.3 研究连接、信息论 ch2 §2.4.3）。）

**5（动手·编程，EM 拟合两分量高斯混合 = 构象盆地，与研究相关）。** 仅用 `numpy`：从 $0.4\,\mathcal N(-2,1)+0.6\,\mathcal N(3,0.5^2)$ 抽 1000 点（当作某二面角/距离的两个构象盆地）。(a) 实现 §7.6.4 的 EM（E 步算责任、M 步加权重估 $\pi,\mu,\sigma^2$）。(b) 画每次迭代的对数似然，确认**单调不减**。(c) 用 3 个不同随机初值跑，观察是否都收敛到同一解；构造一个会塌缩（某 $\sigma^2\to0$）的初值。
（*提示*：(b) §7.6.3 单调上升定理的数值确认。(c) 不同初值可能落到不同局部最优；若某分量初值正好压在单点上、$\sigma^2$ 初值极小，似然会向退化解发散——这就是 §7.6.4 边界说的塌缩，实践要加方差下界。这正是把 MD 帧软聚类到「折叠/未折叠」盆地的玩具版。）

**6（思考·辨析，伪似然为何能绕过配分函数，与研究相关）。** (a) 写出 Potts 模型某位点的条件分布 $p(a_i\mid\mathbf a_{\setminus i})$，说明为什么它的归一化只对 20 种氨基酸求和、与全局 $Z$ 无关。(b) 解释 plmDCA「一致但不有效」的含义，以及为什么实践仍选它而非真 MLE。(c) 这和第 5 章 §5.4 高斯图模型「直接看精度矩阵」是什么关系？
（*提示*：(a) $p(a_i{=}b\mid\mathbf a_{\setminus i})=\frac{\exp(h_i(b)+\sum_{j\ne i}J_{ij}(b,a_j))}{\sum_{b'=1}^{20}\exp(h_i(b')+\sum_{j\ne i}J_{ij}(b',a_j))}$——分子分母里全局 $Z$ 约掉，只剩对 $b'$ 的 20 项 softmax（§7.7.2），故可算。(b) 一致=$n$→∞ 收敛真参数，不有效=方差大于真 MLE（§7.5 CRLB 达不到）；选它因真 MLE 的 $Z$ 是 $20^L$ 项、#P-难、算不动（§7.7.2、第 4 章 §4.5.2）。(c) 二者都在「分离直接耦合 vs 间接相关」：GGM 是连续高斯版（精度矩阵有闭式逆），Potts/plmDCA 是离散版（须伪似然近似），第 5 章 §5.4 是离散 DCA 的「线性脚手架」。）

---

## 7.10 深入阅读指引

**点估计、MLE、充分性、CRLB 的标准教材（本章 §7.1–7.5）**
- **Casella & Berger, *Statistical Inference*（第 2 版），第 6–7 章「充分性原理 / 点估计」+ 第 10 章「渐近」**：本章 §7.1–7.5 的标准研究生参考，充分统计量、Fisher–Neyman、Rao–Blackwell、CRLB、MLE 渐近性质的权威而清晰的处理，定理陈述与本章记号一致。
- **Wasserman, *All of Statistics*（2004），第 9 章「参数推断」**：偏差/方差/MSE、矩估计、MLE、Fisher 信息、CRLB 一气呵成，写法极快、面向 CS/ML 读者，最适合作本章的精炼复习。
- **Lehmann & Casella, *Theory of Point Estimation*（第 2 版）**：点估计理论的圣经，充分性、完备性、UMVUE、CRLB 的彻底处理——想把 §7.4–7.5 学到底看它。
- **van der Vaart, *Asymptotic Statistics*（1998），MLE 渐近章节**：MLE 一致性/渐近正态/渐近有效的严格条件与证明，把 §7.3.5 的「正则条件」讲到精确——做理论方向必读。

**EM 与隐变量模型（§7.6）**
- **Dempster, Laird & Rubin, *"Maximum likelihood from incomplete data via the EM algorithm"*（JRSS-B, 1977）**：EM 的奠基论文，本章 §7.6 的源头。
- **Bishop, *Pattern Recognition and Machine Learning*（2006），第 9 章「混合模型与 EM」**：GMM 的 EM、ELBO 视角、与变分推断的连接讲得最清楚——对应 §7.6.2–7.6.4，强烈推荐。
- **Durbin, Eddy, Krogh & Mitchison, *Biological Sequence Analysis*（1998），HMM 与 profile HMM 章节**：Baum–Welch（序列上的 EM）、前向–后向、profile HMM 建库的权威生信教材——把 §7.6 的研究连接落到实处，做序列比对/同源检测必读。

**与蛋白质研究的桥梁（§7.5、§7.7）**
- **Ekeberg, Lövkvist, Lan, Weigt & Aurell, *"Improved contact prediction in proteins: using pseudolikelihood to infer Potts models"*（Phys. Rev. E, 2013）**：plmDCA 原始论文，把伪似然用于 Potts 参数估计做接触预测——本章 §7.7.2 的字面落地，必读。
- **Balakrishnan, Kamisetty, Carbonell, Lee & Langmead, *"Learning generative models for protein fold families (GREMLIN)"*（Proteins, 2011）**：另一条伪似然/正则化 Potts 路线——对应 §7.7。
- **信息论课程 ch2 §2.8、ch4 §4.7（交叉熵=极大似然=最小 KL 完整版，§4.7.2 逐步推导、§4.7.3 前/反向 KL）、ch4 §4.6（充分统计量的信息论刻画）、ch10 §10.3**：MLE=最小交叉熵=最小 KL 的信息论侧、充分统计量=信息守恒、最大熵=指数族、Potts 的最大熵推导——与本章 §7.3.3、§7.4、§7.7 互为表里，务必对照。
- **折叠课程第 5 章（经典方法 / 共进化 / DCA）、第 7 章（AlphaFold2，Evoformer 输入特征）**：本章 §7.5、§7.7 估计理论在蛋白质接触预测/MSA 特征上的应用全景。

> **下一章预告**：本章我们把频率派点估计讲透了，但它有两处「意犹未尽」：(1) 它只给一个**点**估计 $\hat\theta$，对「我对它有多确定」只能靠渐近正态近似挤出误差棒——当样本少、参数在边界、模型非正则时这套近似就不可靠；(2) 我们反复看到「伪计数 = 先验」「正则化 = MAP」这些贝叶斯影子，却一直没正面定义先验、后验。第 8 章《贝叶斯推断：共轭、层次模型与模型比较》就来把 $\theta$ 本身当成随机变量、给它一个**先验分布**，用贝叶斯定理算出**后验分布**——它不只给点估计，更给出**整条不确定性**（可信区间），且小样本下自然、不靠渐近近似。本章 §7.1 的伪计数、§7.7 的岭/lasso/伪计数 MAP，会在第 8 章升级为「共轭先验的后验更新」的完整图景；而 §7.6 的 ELBO 会重现为变分贝叶斯。带着「MLE = 无信息先验下后验的众数」这条桥进入第 8 章，你会发现频率派与贝叶斯不是对立，而是同一片连续光谱的两端。
