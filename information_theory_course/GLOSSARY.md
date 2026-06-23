# 信息论课程 · 中英文术语对照表（GLOSSARY）

> 速查约定：每条格式为 **中文名 | English | 一句话精准解释（必要时注所在章）**。
> 分组大致按课程主题；同一术语跨章出现时归入其最核心的主题组。
> 缩写在领域内通用时随术语标注（如 KL、MSA、DCA、AEP、DMC、SNR）。

---

## 目录

1. [基础概念与信息度量](#1-基础概念与信息度量)
2. [概率与数学工具](#2-概率与数学工具)
3. [熵及其家族](#3-熵及其家族)
4. [相对熵、互信息与统计推断](#4-相对熵互信息与统计推断)
5. [渐近均分性与典型集](#5-渐近均分性与典型集)
6. [无失真信源编码（压缩）](#6-无失真信源编码压缩)
7. [信道容量与有噪信道编码](#7-信道容量与有噪信道编码)
8. [微分熵与高斯信道](#8-微分熵与高斯信道)
9. [率失真理论（有损压缩）](#9-率失真理论有损压缩)
10. [最大熵、指数族与机器学习](#10-最大熵指数族与机器学习)
11. [生物信息学与蛋白质折叠应用](#11-生物信息学与蛋白质折叠应用)
12. [人物、里程碑与文献](#12-人物里程碑与文献)

---

## 1. 基础概念与信息度量

| 中文名 | English | 解释 |
|---|---|---|
| 信息论 | Information theory | 研究信息的度量、压缩与可靠传输极限的数学学科，1948 年由 Shannon 创立（第 1 章）。 |
| 信息 | Information | 不确定性的消解；越意外的事件发生时携带的信息越多（第 1 章）。 |
| 不确定性的消解 | Reduction of uncertainty | "信息"的本质刻画：被告知前后不确定性减少的那一部分（第 1 章）。 |
| 自信息 / 惊奇度 | Self-information / Surprisal | 单个结果的信息量 $I(p)=-\log_b p$，由四条公理唯一确定为对数（第 1、3 章）。 |
| 比特 | Bit / Shannon | 以 2 为底对数的信息单位，= 一次等概率二选一的不确定性（第 1 章）。 |
| 奈特 | Nat | 以 $e$ 为底（自然对数）的信息单位，ML 损失常用，$1$ nat $\approx1.4427$ bit（第 1、2 章）。 |
| 哈特利 / 迪特 | Hartley / Dit / Ban | 以 10 为底对数的历史信息单位（第 1、2 章）。 |
| 表示定理 | Representation theorem | 在几条温和公理下把某个量唯一确定为特定数学形式（如自信息=对数）的定理（第 1 章）。 |
| 语义 | Meaning / Semantics | 消息的"含义"；香农信息与之无关，只看统计结构（第 1 章）。 |
| 冗余 / 冗余度 | Redundancy | 信源中可预测、可压缩的部分；英语约有 70% 冗余（第 1、3 章）。 |
| 通信系统框图 | Communication system diagram | 信源→信源编码→信道编码→信道+噪声→译码→信宿的模块化模型（第 1 章）。 |
| 信源 | Source | 产生符号序列的概率模型，其压缩极限是熵 $H$（第 1、6 章）。 |
| 信宿 | Sink | 接收消息的终端（第 1 章）。 |
| 信源编码 | Source coding | 去冗余（压缩）的编码，极限是熵 $H$（第 1、6 章）。 |
| 信道编码 | Channel coding | 加冗余（纠错）的编码，极限是容量 $C$（第 1、7 章）。 |
| 三大基本问题 | Three fundamental problems | 数据压缩、可靠通信、推断与学习——信息论的骨架（第 1 章）。 |
| 学习=压缩=预测 | Learning = Compression = Prediction | 信息论的统一世界观：好模型必抓住数据规律、能压缩、能预测（第 1、6 章）。 |
| Kolmogorov 复杂度 | Kolmogorov complexity | 输出某具体串的最短程序长度 $K(x)$，不可计算；与香农熵在期望意义下握手（第 1、5、6 章）。 |
| 算法信息论 | Algorithmic information theory | 以 Kolmogorov 复杂度为核心、度量个体串信息的理论（第 1、5 章）。 |
| Solomonoff 归纳 | Solomonoff induction | 按程序长度（复杂度）加权所有能生成数据的程序的理想预测，不可计算（第 10 章）。 |

---

## 2. 概率与数学工具

| 中文名 | English | 解释 |
|---|---|---|
| 随机变量 | Random variable | 从样本空间到取值空间的可测函数；与其取值 $x$ 须严格区分（第 2 章）。 |
| 概率质量函数 | Probability mass function (pmf) | 离散分布 $p(x)=\Pr[X=x]$，非负且求和为 1（第 2 章）。 |
| 概率密度函数 | Probability density function (pdf) | 连续分布的密度 $f(x)$，非负且积分为 1，可大于 1（第 2 章）。 |
| 期望 | Expectation | 分布的加权平均 $\mathbb{E}[g(X)]$；线性、且无需独立（第 2 章）。 |
| 方差 | Variance | 离散程度 $\mathbb{E}[(X-\mathbb{E}X)^2]$；非线性，相加需不相关（第 2 章）。 |
| 联合分布 | Joint distribution | 多变量同时取值的分布 $p(x,y)$（第 2、3 章）。 |
| 边缘分布 | Marginal distribution | 对不关心的变量求和/积分掉所得的分布（第 2 章）。 |
| 边缘化 | Marginalization | 把变量"加掉/积掉"使其消失的操作（第 2 章）。 |
| 条件分布 | Conditional distribution | $p(x\mid y)=p(x,y)/p(y)$；对每个固定 $y$ 都是合法分布（第 2 章）。 |
| 链式法则（概率） | Chain rule of probability | $p(x_1,\dots,x_n)=\prod_i p(x_i\mid x_{<i})$，恒等式，无需独立（第 2 章）。 |
| 贝叶斯定理 | Bayes' theorem | 后验 ∝ 似然 × 先验；$p(x\mid y)=p(y\mid x)p(x)/p(y)$（第 2 章）。 |
| 全期望公式 | Law of total expectation | $\mathbb{E}[X]=\mathbb{E}[\mathbb{E}[X\mid Y]]$，又称塔性质/迭代期望（第 2 章）。 |
| 独立 | Independence | $p(x,y)=p(x)p(y)$；等价于互信息为 0（第 2、4 章）。 |
| 条件独立 | Conditional independence | 给定 $Y$ 后 $p(x,z\mid y)=p(x\mid y)p(z\mid y)$；与独立互不蕴含（第 2 章）。 |
| 马尔可夫链 | Markov chain | $X\to Y\to Z$ 即 $p(x,y,z)=p(x)p(y\mid x)p(z\mid y)$，等价于 $X\perp Z\mid Y$（第 2、4 章）。 |
| 凸函数 / 凹函数 | Convex / Concave function | 弦不低于/不高于曲线的函数；$-\log$ 凸、$\log$ 凹（第 2 章）。 |
| Jensen 不等式 | Jensen's inequality | 凸函数下"先平均再作用 ≤ 先作用再平均"；信息论第一工具（第 2 章）。 |
| 支撑线 | Supporting line | 凸函数在某点处不超过它的切线，用于一行证明 Jensen（第 2 章）。 |
| 弱大数定律 | Weak Law of Large Numbers (WLLN) | 样本均值依概率收敛到期望，支撑弱 AEP（第 2、5 章）。 |
| 强大数定律 | Strong Law of Large Numbers (SLLN) | 样本均值几乎必然收敛到期望，支撑强 AEP（第 2、5 章）。 |
| Markov 不等式 | Markov's inequality | $\Pr[X\ge a]\le\mathbb{E}[X]/a$，最基本的集中不等式（第 2 章）。 |
| Chebyshev 不等式 | Chebyshev's inequality | $\Pr[\lvert X-\mu\rvert\ge\varepsilon]\le\sigma^2/\varepsilon^2$，多项式尾界（第 2 章）。 |
| Hoeffding 不等式 | Hoeffding's inequality | 独立有界变量样本均值的指数尾界 $2e^{-2nt^2}$（第 2 章）。 |
| 经验分布 | Empirical distribution | 样本频率谱 $\hat p_n(x)$，随样本量收敛到真分布（第 2、4、11 章）。 |
| 经验频率 | Empirical frequency | 各符号在样本里出现的相对次数（第 2 章）。 |
| 类型 / 类型方法 | Type / Method of types | 序列的经验分布及精确计数其序列数的组合工具（第 2、5 章）。 |
| 极大似然估计 | Maximum likelihood estimation (MLE) | 选使数据似然最大的参数；离散下 = 匹配经验分布 = 最小化 KL（第 2、4、10 章）。 |
| 对数似然 | Log-likelihood | 似然取对数；负对数似然即交叉熵/码长（第 2、4 章）。 |
| 伪计数 / 拉普拉斯平滑 | Pseudocount / Laplace (add-one) smoothing | 给频率加先验计数避免零概率，等价于贝叶斯平滑（第 2、11 章）。 |
| 伯努利分布 | Bernoulli distribution | 二元 0/1 分布，其熵为二元熵函数 $H_b(p)$（第 2 章）。 |
| 范畴分布 | Categorical distribution | $m$ 元离散分布（如 4 碱基、20 氨基酸）（第 2 章）。 |
| 二项分布 | Binomial distribution | $n$ 次独立伯努利之和的分布（第 2 章）。 |
| 几何分布 | Geometric distribution | 首次成功的等待时间分布（第 2 章）。 |
| 泊松分布 | Poisson distribution | 稀有事件计数分布（第 2 章）。 |
| 高斯 / 正态分布 | Gaussian / Normal distribution | 固定方差下微分熵最大的连续分布（第 2、8 章）。 |
| 指数分布 | Exponential distribution | 正支撑、固定均值下微分熵最大的分布（第 2、8 章）。 |
| 均匀分布 | Uniform distribution | 无约束下熵最大的分布；离散熵上界 $\log\lvert\mathcal X\rvert$ 在此取到（第 2、3 章）。 |

---

## 3. 熵及其家族

| 中文名 | English | 解释 |
|---|---|---|
| 香农熵 | Shannon entropy | $H(X)=-\sum p\log p=\mathbb{E}[-\log p(X)]$，自信息的期望/平均惊奇（第 3 章）。 |
| 二元熵函数 | Binary entropy function | $H_b(p)=-p\log p-(1-p)\log(1-p)$，对称、严格凹、$p=1/2$ 取 1（第 2、3 章）。 |
| 联合熵 | Joint entropy | $H(X,Y)=-\sum p(x,y)\log p(x,y)$，复合变量的总不确定性（第 3 章）。 |
| 条件熵 | Conditional entropy | $H(Y\mid X)=\mathbb{E}[-\log p(Y\mid X)]$，已知 $X$ 后 $Y$ 的剩余不确定性（第 3 章）。 |
| 链式法则（熵） | Chain rule for entropy | $H(X,Y)=H(X)+H(Y\mid X)$，可推广到 $n$ 个变量之和（第 3 章）。 |
| Shannon–Khinchin 唯一性定理 | Shannon–Khinchin uniqueness theorem | 连续+单调+分组可加三公理唯一逼出 $-C\sum p\log p$（第 3 章）。 |
| 分组公理 / 可加性 | Grouping / Recursivity axiom | "组间不确定性 + 组内条件不确定性 = 总不确定性"，熵公理化核心（第 3 章）。 |
| Gibbs 不等式 | Gibbs' inequality | $\sum p\log(p/q)\ge0$，即 KL 非负，熵诸性质的统一源头（第 3、4 章）。 |
| 条件化不增熵 | Conditioning reduces entropy | $H(Y\mid X)\le H(Y)$（期望意义），差值即互信息（第 3 章）。 |
| 子可加性 | Subadditivity | $H(X_1,\dots,X_n)\le\sum H(X_i)$，独立时取等（第 3 章）。 |
| 熵的凹性 | Concavity of entropy | $H(p)$ 是概率向量的凹函数，使最大熵成为凸优化（第 3 章）。 |
| 最大熵原理 | Maximum entropy principle | 约束下选熵最大（最不偏不倚）的分布；均匀/高斯是其特例（第 3、8、10 章）。 |
| 随机过程 | Stochastic process | 符号间可相关的序列 $\{X_i\}$，如文本、DNA、蛋白序列（第 3、5 章）。 |
| 平稳过程 | Stationary process | 分布不随时间平移改变的过程，熵率良定义（第 3、5、6 章）。 |
| 熵率 | Entropy rate | 平稳信源每符号渐近不确定性 $\lim\frac1n H(X_1,\dots,X_n)$（第 3、5、6 章）。 |
| 平稳遍历过程 | Stationary ergodic process | 时间平均=系综平均的平稳过程，SMB 定理成立其上（第 5、6 章）。 |
| Rényi 熵 | Rényi entropy | 熵族 $H_\alpha=\frac{1}{1-\alpha}\log\sum p^\alpha$，$\alpha\to1$ 即香农熵（第 3 章）。 |
| Tsallis 熵 | Tsallis entropy | 换用非严格可加公理得到的另一族熵（第 3 章）。 |
| 碰撞熵 | Collision entropy | Rényi $\alpha=2$，$-\log\sum p^2$，关联逆 Simpson 指数（第 3 章）。 |
| 最小熵 | Min-entropy | Rényi $\alpha\to\infty$，$-\log\max_x p(x)$，密码学随机性标准（第 3 章）。 |
| Hartley 熵 | Hartley entropy | Rényi $\alpha=0$，$\log$ 支撑集大小（第 3 章）。 |
| 逆 Simpson 指数 | Inverse Simpson index | $1/\sum p^2$，生态/群体遗传中的"有效种类数"，= $2^{H_2}$（第 3 章）。 |
| 微分熵 | Differential entropy | 连续版熵 $h(X)=-\int f\log f$；可为负、依赖坐标，非信息位数（第 3、8 章）。 |

---

## 4. 相对熵、互信息与统计推断

| 中文名 | English | 解释 |
|---|---|---|
| 相对熵 / KL 散度 | Relative entropy / Kullback–Leibler (KL) divergence | $D(p\Vert q)=\mathbb{E}_p[\log(p/q)]$，用 $q$ 假装 $p$ 的额外比特代价；≥0、不对称、可为 ∞（第 4 章）。 |
| 交叉熵 | Cross-entropy | $H(p,q)=-\sum p\log q=H(p)+D(p\Vert q)$，用 $q$ 编码 $p$ 的平均码长（第 4、6 章）。 |
| 信息不等式 | Information inequality | KL 非负的别名，全章基石（第 4 章）。 |
| 绝对连续 | Absolute continuity ($p\ll q$) | $q$ 在 $p$ 有质量处不为零，KL 有限的前提（第 4 章）。 |
| 全变差距离 | Total variation distance | $\frac12\sum\lvert p-q\rvert$，真正的分布间距离/度量（第 4 章）。 |
| Pinsker 不等式 | Pinsker's inequality | KL 单向控制全变差：$\Vert p-q\Vert_{TV}\le\sqrt{D(p\Vert q)/2}$（第 4 章）。 |
| f-散度 | f-divergence | $D_f(p\Vert q)=\sum q\,f(p/q)$，KL/TV/χ²/Hellinger 的共同家族（第 4 章）。 |
| 互信息 | Mutual information | $I(X;Y)=D(p_{XY}\Vert p_Xp_Y)=H(Y)-H(Y\mid X)$，两变量共享的信息（第 4 章）。 |
| 信息维恩图 | Information Venn diagram | 把熵当面积、互信息当交集的助记图；三变量交集可负，勿当测度（第 4 章）。 |
| 条件互信息 | Conditional mutual information | $I(X;Y\mid Z)$，给定 $Z$ 后的互信息；与 $I(X;Y)$ 无固定大小关系（第 4 章）。 |
| 交互信息 | Interaction information / Co-information | 三变量"三重交集" $I(X;Y)-I(X;Y\mid Z)$，可为负（第 4 章）。 |
| 链式法则（互信息） | Chain rule for mutual information | $I(X_1,\dots,X_n;Y)=\sum_i I(X_i;Y\mid X_{<i})$（第 4 章）。 |
| 数据处理不等式 | Data Processing Inequality (DPI) | $X\to Y\to Z\Rightarrow I(X;Z)\le I(X;Y)$，后处理不增信息（第 4 章）。 |
| Fano 不等式 | Fano's inequality | 把残余不确定性 $H(X\mid Y)$ 翻译成错误率下界 $P_e$（第 4、7 章）。 |
| 充分统计量 | Sufficient statistic | $I(\theta;T)=I(\theta;X)$ 即 $T$ 无损概括数据；DPI 取等（第 4 章）。 |
| 极小充分统计量 | Minimal sufficient statistic | 不丢信息前提下对数据的最大压缩；信息瓶颈的理想终点（第 4 章）。 |
| 前向 KL | Forward KL ($D(p\Vert q_\theta)$) | 覆盖/矩匹配（mass-covering）；极大似然用之，倾向摊宽（第 4、10 章）。 |
| 反向 KL | Reverse KL ($D(q_\theta\Vert p)$) | 寻峰（mode-seeking/zero-forcing）；变分推断 ELBO 用之，倾向缩进单峰（第 4、10 章）。 |
| 矩匹配 | Moment matching | 让模型的充分统计量期望等于经验值；前向 KL 的解（第 4、10 章）。 |
| 信息投影 | Information projection | 在 KL 意义下让模型逼近目标分布的几何视角（第 4 章）。 |
| 标签平滑 | Label smoothing | 给 one-hot 目标掺均匀分布，防交叉熵爆炸、抑过度自信（第 4 章）。 |
| 知识蒸馏 | Knowledge distillation | 小模型拟合大模型软标签分布，传递类间关系信息（第 4 章）。 |

---

## 5. 渐近均分性与典型集

| 中文名 | English | 解释 |
|---|---|---|
| 渐近均分性 | Asymptotic Equipartition Property (AEP) | 信息论的大数定律：样本熵依概率收敛到 $H$（第 5 章）。 |
| 样本熵 / 经验熵率 | Sample entropy / Empirical entropy rate | $-\frac1n\log p(X^n)$，自信息的样本平均（第 5 章）。 |
| 典型集 | Typical set | 样本熵贴近 $H$ 的序列集 $A_\varepsilon^{(n)}$，约 $2^{nH}$ 条、近似等概率、占满概率（第 5 章）。 |
| 典型序列 | Typical sequence | 概率约 $2^{-nH}$ 的序列；注意典型 ≠ 最可能（第 5 章）。 |
| 渐近均分 | Equipartition | 概率近似均匀分摊在约 $2^{nH}$ 条典型序列上（第 5 章）。 |
| 高概率集合定理 | High-probability set theorem | 任何概率≥1−δ 的集合势都≥约 $2^{nH}$，证典型集指数最优（第 5 章）。 |
| 集中测度 | Concentration of measure | 高维质量集中在中等半径"薄壳"上的现象，典型集是其离散版（第 5 章）。 |
| 薄壳 | Thin shell | 典型集的几何形象：质量峰=条数×单条概率，落在中等频率处（第 5 章）。 |
| 联合典型集 | Jointly typical set | 序对要求边缘各典型且联合也典型的集合（第 5 章）。 |
| 联合 AEP | Joint AEP | 独立对落入联合典型集的概率 ≐ $2^{-nI(X;Y)}$，信道编码命根子（第 5 章）。 |
| 联合典型性译码 | Jointly typical decoding | 找与接收序列联合典型的唯一码字作译码结果（第 5、7 章）。 |
| 并集界 | Union bound | 多个事件并的概率不超过各概率之和，错误分析常用（第 5、7 章）。 |
| Shannon–McMillan–Breiman 定理 | Shannon–McMillan–Breiman theorem | AEP 推广到平稳遍历源、几乎必然收敛到熵率（第 5 章）。 |
| 遍历性 | Ergodicity | 时间平均=系综平均；SMB 定理的前提（第 5 章）。 |
| 有效序列空间 | Effective sequence space | 天然蛋白受约束、等价于 $2^{nH_{eff}}$ 的窄空间，远小于名义 $20^n$（第 5 章）。 |

---

## 6. 无失真信源编码（压缩）

| 中文名 | English | 解释 |
|---|---|---|
| 无失真 / 无损 | Lossless | 解码端精确无误恢复原序列（第 6 章）。 |
| 码字 | Codeword | 符号映成的 0/1 串，其长度记 $\ell(x)$（第 6 章）。 |
| 期望码长 | Expected code length | $L=\sum p(x)\ell(x)$，压缩追求最小化的目标（第 6 章）。 |
| 定长码 | Fixed-length / Block code | 每码字等长，可解但不利用分布不均（第 6 章）。 |
| 非奇异码 | Nonsingular code | 不同符号映到不同码字（第 6 章）。 |
| 唯一可译码 | Uniquely decodable code (UD) | 码字拼接后仍可无歧义还原（第 6 章）。 |
| 前缀码 / 瞬时码 | Prefix(-free) code / Instantaneous code | 无码字是另一码字前缀，即时可解、码字都在叶子（第 6 章）。 |
| 码树 | Code tree | 前缀码的 D 叉树表示，码长=深度（第 6 章）。 |
| 扩展 | Extension | 把符号序列映成码字拼接的映射 $C^*$（第 6 章）。 |
| Kraft 不等式 | Kraft inequality | $\sum D^{-\ell_i}\le1$ 是码长可凑成前缀码的充要条件（第 6 章）。 |
| McMillan 不等式 | McMillan inequality | 唯一可译码也满足 Kraft 界，故 UD 不比前缀码更短（第 6 章）。 |
| 信源编码下界 | Source coding lower bound | 任何 UD 码 $L\ge H$，由 Gibbs+Kraft 证（第 6 章）。 |
| 香农码 | Shannon code | 取 $\ell=\lceil-\log p\rceil$，达到 $H\le L<H+1$（第 6 章）。 |
| D-adic 分布 | D-adic distribution | 各概率均为 $D^{-\ell}$ 形式，码长下界取等的情形（第 6 章）。 |
| 香农信源编码定理 | Shannon's source coding theorem | 每符号码长可任意逼近熵率、不可低于它（第 6 章）。 |
| 块编码 | Block coding | 多符号打包编码，把 +1 开销摊薄为 $1/n$（第 6 章）。 |
| 典型集编码 | Typical set encoding | 给约 $2^{nH}$ 条典型序列编号、达到压缩极限的构造（第 5、6 章）。 |
| Huffman 编码 | Huffman coding | 自底向上贪心合并最小两权重的最优前缀码（第 6 章）。 |
| 交换论证 | Exchange argument | 证明 Huffman 最优性的"贪心+交换"手法（第 6 章）。 |
| Shannon–Fano–Elias 码 | Shannon–Fano–Elias code | 用累积分布中点定位符号的过渡码，算术编码的胚胎（第 6 章）。 |
| 算术编码 | Arithmetic coding | 把整条消息映成 $[0,1)$ 一子区间，突破整数比特枷锁、支持自适应模型（第 6 章）。 |
| 区间编码 | Range coding | 算术编码的工程近亲（第 6 章）。 |
| 非对称数字系统 | Asymmetric numeral systems (rANS) | 现代熵编码器（zstd 等）的主力，逼近 $\sum-\log p$（第 6 章）。 |
| 通用压缩 | Universal coding | 无需预知分布即对任意平稳遍历源渐近达熵率（第 6 章）。 |
| Lempel–Ziv | Lempel–Ziv (LZ77/LZ78/LZW) | 通用压缩算法族，边读边从数据学统计结构（第 6 章）。 |
| 滑动窗口 | Sliding window | LZ77 维护的最近字符缓冲，用"指回去"代替查字典（第 6 章）。 |
| DEFLATE | DEFLATE | gzip/zip/PNG 内核 = LZ77 + Huffman（第 6 章）。 |
| 最小描述长度 | Minimum description length (MDL) | 选"模型+用模型压数据"总码长最短的模型，量化奥卡姆剃刀（第 6、10 章）。 |
| 归一化压缩距离 | Normalized compression distance (NCD) | 用压缩长度估计序列相似度的无比对方法（第 6 章）。 |

---

## 7. 信道容量与有噪信道编码

| 中文名 | English | 解释 |
|---|---|---|
| 信道 | Channel | 把输入符号随机映成输出符号的对象，随机性即噪声（第 7 章）。 |
| 噪声 | Noise | 信道使同一输入可能输出不同结果的不确定性（第 7 章）。 |
| 离散无记忆信道 | Discrete Memoryless Channel (DMC) | 由转移矩阵刻画、各位独立作用 $p(y^n\mid x^n)=\prod p(y_i\mid x_i)$（第 7 章）。 |
| 转移概率 / 转移矩阵 | Transition probability / matrix | $p(y\mid x)$ 排成的行随机矩阵，完全刻画信道（第 7 章）。 |
| 二元对称信道 | Binary Symmetric Channel (BSC) | 每位以翻转概率 $p$ 被翻转，容量 $C=1-H(p)$（第 7 章）。 |
| 翻转概率 | Crossover probability | BSC 中比特被翻转的概率（第 7 章）。 |
| 二元擦除信道 | Binary Erasure Channel (BEC) | 每位以概率 ε 被擦成 $e$、永不翻转，容量 $C=1-\varepsilon$（第 7 章）。 |
| 擦除符号 | Erasure symbol | BEC 输出的"丢失"标记 $e$，已知错误位置（第 7 章）。 |
| Z 信道 | Z-channel | 单向出错的非对称信道，最优输入非均匀（第 7 章）。 |
| 信道容量 | Channel capacity | $C=\max_{p(x)}I(X;Y)$，可靠通信速率的上确界（第 7 章）。 |
| 信息容量 vs 操作容量 | Information vs Operational capacity | 优化定义的容量 vs 错误可趋零的最大码率；香农证二者相等（第 7 章）。 |
| Blahut–Arimoto 算法 | Blahut–Arimoto algorithm | 交替优化数值求容量/率失真的坐标上升/下降法（第 7、9 章）。 |
| 码本 / 码书 | Codebook | 全部码字的集合（第 7 章）。 |
| 码率 | Rate | $R=\log_2 M/n$ 比特/每次信道使用（第 7 章）。 |
| 可达 | Achievable | 存在码序列使错误率随 $n$ 趋零的速率（第 7、9 章）。 |
| 有噪信道编码定理 | Noisy-channel coding theorem | 香农第二定理：$R<C$ 可达、$R>C$ 不可达（第 7 章）。 |
| 随机编码 | Random coding | 随机生成码本证明"平均好故存在好码"的存在性论证（第 7 章）。 |
| 概率方法 | Probabilistic method | 证明对象存在却不给出构造的数学手法（第 7 章）。 |
| 逆定理 / 弱逆 / 强逆 | Converse / Weak / Strong converse | $R>C$ 时错误率不能趋零（弱）/必趋 1（强）的证明（第 7 章）。 |
| 球填充 | Sphere-packing | 在输出空间塞互不重叠噪声球，球数 ≈ $2^{nC}$ 的几何直觉（第 7 章）。 |
| 反馈不增容量 | Feedback does not increase capacity | DMC 的容量不因反馈增大，但反馈简化工程（第 7 章）。 |
| 信源-信道分离定理 | Source–channel separation theorem | 先压到 $H$、再纠到 $C$ 两段式渐近最优，充要条件 $H<C$（第 1、7 章）。 |
| 联合信源信道编码 | Joint source–channel coding (JSCC) | 网络/有限时延下可能优于分离方案的联合设计（第 1、7 章）。 |
| 重复码 | Repetition code | 每位重复多次+多数表决，以速率趋零换可靠性（第 7 章）。 |
| 汉明码 | Hamming code | (7,4) 线性码，用校正子直接定位单比特错（第 7 章）。 |
| 校验矩阵 | Parity-check matrix | 定义线性码 $Hc^\top=0$ 的矩阵（第 7 章）。 |
| 校正子 | Syndrome | $s=Hr^\top$，直接读出错误位置（第 7 章）。 |
| 最小汉明距离 | Minimum Hamming distance | 纠 $t$ 错需 $d_{\min}\ge2t+1$（第 7 章）。 |
| LDPC 码 | Low-Density Parity-Check (LDPC) code | 稀疏校验码 + 置信传播译码，逼近容量（第 7 章）。 |
| 置信传播 / 和积算法 | Belief propagation / Sum-product | Tanner 图上迭代近似后验推断的译码=图模型推断（第 7 章）。 |
| Turbo 码 | Turbo code | 1993 年首个实践逼近容量的迭代软信息码（第 7 章）。 |
| Polar 码 | Polar code | 2009 年首个被证明达容量的显式构造码，靠信道极化（第 7 章）。 |
| 信道极化 | Channel polarization | 递归变换把信道分化成近完美与近无用两类（第 7 章）。 |
| 有限码长信息论 | Finite blocklength information theory | 刻画有限 $n$ 下离容量差距（Polyanskiy–Poor–Verdú）（第 7 章）。 |

---

## 8. 微分熵与高斯信道

| 中文名 | English | 解释 |
|---|---|---|
| 微分熵 | Differential entropy | $h(X)=-\int f\log f$；可为负、依赖坐标量纲，非"信息位数"（第 8 章）。 |
| 量化 | Quantization | 把连续量切成宽 Δ 格子离散化；$H(X^\Delta)\approx h(X)-\log\Delta$（第 8 章）。 |
| 缩放变换律 | Scaling law | $h(aX+b)=h(X)+\log\lvert a\rvert$；多维为 $+\log\lvert\det A\rvert$（第 8 章）。 |
| 雅可比行列式项 | Log-Jacobian term | 坐标变换下微分熵的修正，即归一化流里的 $\log\lvert\det J\rvert$（第 8 章）。 |
| 归一化流 | Normalizing flow | 用可逆变换+对数雅可比构造灵活密度的生成模型（第 8 章）。 |
| 最大熵分布 | Maximum-entropy distribution | 给定矩约束下熵最大的分布，必为指数族（第 8、10 章）。 |
| 最大熵界 | Maximum-entropy bound | 如方差固定时 $h(X)\le\frac12\log(2\pi e\sigma^2)$，高斯取等（第 8 章）。 |
| 加性高斯白噪声信道 | Additive White Gaussian Noise (AWGN) channel | $Y=X+N$，$N$ 高斯，连续信道的核心模型（第 8 章）。 |
| 功率约束 | Power constraint | $\mathbb{E}[X^2]\le P$，连续信道相对离散最本质的新元素（第 8 章）。 |
| 信噪比 | Signal-to-noise ratio (SNR) | $P/\mathcal{N}$，容量公式的核心无量纲量（第 8 章）。 |
| 高斯信道容量 | Gaussian channel capacity | $C=\frac12\log_2(1+\mathrm{SNR})$，最优输入为高斯（第 8 章）。 |
| 香农–哈特利公式 | Shannon–Hartley theorem | 带限信道 $C=W\log_2(1+\mathrm{SNR})$ 比特/秒（第 8 章）。 |
| 奈奎斯特采样 | Nyquist sampling | 带宽 $W$ 信道每秒 $2W$ 次独立使用（第 8 章）。 |
| 频谱效率 | Spectral efficiency | $C/W=\log_2(1+\mathrm{SNR})$ 比特/秒/赫兹（第 8 章）。 |
| 香农极限 | Shannon limit | 每比特能量下界 $E_b/N_0\ge\ln2\approx-1.59$ dB（第 8 章）。 |
| 并联高斯信道 | Parallel Gaussian channels | 多个独立噪声不同的子信道（OFDM/MIMO 本征模式）（第 8 章）。 |
| 注水 | Water-filling | 在总功率约束下最优功率分配 $P_i=(\mu-\mathcal{N}_i)^+$（第 8 章）。 |
| 水位 | Water level | 注水统一水面 $\mu$，由总功率约束确定（第 8 章）。 |
| 熵幂不等式 | Entropy Power Inequality (EPI) | $e^{2h(X+N)}\ge e^{2h(X)}+e^{2h(N)}$，证"高斯噪声最坏"（第 8 章）。 |

---

## 9. 率失真理论（有损压缩）

| 中文名 | English | 解释 |
|---|---|---|
| 有损压缩 | Lossy compression | 允许可控失真换取更高压缩比（JPEG/MP3/隐空间）（第 9 章）。 |
| 重建字母表 | Reconstruction alphabet | 重建符号 $\hat X$ 的取值集，不必等于源字母表（第 9 章）。 |
| 失真度量 | Distortion measure | $d(x,\hat x)\ge0$ 量化重建代价，不必是数学度量（第 9 章）。 |
| 汉明失真 | Hamming distortion | $\mathbb{1}[x\ne\hat x]$，平均即逐符号错误率（第 9 章）。 |
| 平方误差失真 | Squared-error distortion | $(x-\hat x)^2$，平均即均方误差 MSE（第 9 章）。 |
| 重构损失 | Reconstruction loss | ML 里失真度量的别名（重构得多差）（第 9 章）。 |
| 率失真函数 | Rate-distortion function | $R(D)=\min_{p(\hat x\mid x):\mathbb{E}[d]\le D}I(X;\hat X)$，失真 $D$ 下最小比特率（第 9 章）。 |
| 虚拟信道 | Virtual / Test channel | 率失真中自己设计的 $p(\hat x\mid x)$，与物理信道相对（第 9 章）。 |
| 临界失真 | $D_{\max}$ | $\min_{\hat x}\mathbb{E}[d(X,\hat x)]$，超过它 $R=0$（用常数猜测即可）（第 9 章）。 |
| 香农率失真定理 | Shannon's rate-distortion theorem | $R\ge R(D)$ 才可达，否则失真必超 $D$（第 9 章）。 |
| 球覆盖 | Sphere covering | 用尽量少的失真球盖住源空间，与信道编码球填充对偶（第 9 章）。 |
| 时间共享 | Time sharing | 凸组合两个方案证 $R(D)$ 凸性的论证（第 9 章）。 |
| 伯努利源率失真 | Bernoulli source $R(D)$ | $R(D)=H(p)-H(D)$ = 源不确定性 − 赦免的不确定性（第 9 章）。 |
| 高斯源率失真 | Gaussian source $R(D)$ | $R(D)=\frac12\log(\sigma^2/D)$，与高斯信道容量镜像对称（第 9 章）。 |
| 容量–率失真对偶 | Capacity / rate-distortion duality | max 填充 vs min 覆盖，同一变分问题的两个方向（第 9 章）。 |
| β（拉格朗日乘子） | β / Lagrange multiplier | $\beta=-dR/dD$，率失真曲线切线斜率，= β-VAE 的 β（第 9 章）。 |
| 信息瓶颈 | Information Bottleneck (IB) | $\min I(X;T)-\beta I(T;Y)$，失真换成"对 Y 没用的程度"的率失真（第 9、10 章）。 |
| 变分自编码器 | Variational Autoencoder (VAE) | 神经网络化的变分推断；−ELBO = 失真(重构) + 率(KL)（第 9、10 章）。 |
| 证据下界 | Evidence Lower BOund (ELBO) | $\log p(x)$ 的下界 = 重构项 − KL 正则项（第 4、9、10 章）。 |
| β-VAE | β-VAE | 显式参数化率-失真权衡的 VAE，β 即拉格朗日乘子（第 9 章）。 |

---

## 10. 最大熵、指数族与机器学习

| 中文名 | English | 解释 |
|---|---|---|
| 最大熵原理 | Principle of maximum entropy (MaxEnt) | 约束下选熵最大者 = 注入最少额外假设（Jaynes 1957）（第 10 章）。 |
| 最小相对熵原理 | Minimum relative entropy / discrimination | 在先验 $q$ 下选离 $q$ 最近（最小 KL）的分布，MaxEnt 的推广（第 10 章）。 |
| Wallis 推导 | Wallis derivation | 用多项式计数论证熵最大分布"压倒性最可能"（第 10 章）。 |
| Shore–Johnson 公理 | Shore–Johnson axioms | 证明最大熵是唯一满足自洽性的推断规则（第 10 章）。 |
| 特征函数 | Feature function | 约束所用的统计量 $f_k(x)$（广义矩）（第 10 章）。 |
| 指数族 | Exponential family | 矩约束下的最大熵分布 $p\propto\exp(-\sum\lambda_k f_k)$（第 10 章）。 |
| 吉布斯分布 | Gibbs distribution | 指数族/玻尔兹曼形式 $\propto e^{-\sum\lambda_k f_k}$（第 10 章）。 |
| 玻尔兹曼分布 | Boltzmann distribution | $p\propto e^{-\beta E}$，"给定平均能量、其余最大熵"的结果（第 10 章）。 |
| 配分函数 | Partition function ($Z$) | 归一化常数 $Z=\sum_x e^{-\sum\lambda_k f_k}$，常 #P-难（第 10 章）。 |
| 对数配分函数 | Log-partition / Cumulant generating function | $A(\lambda)=\log Z$，凸；一阶导=特征期望、二阶导=协方差（第 10 章）。 |
| 正相 − 负相 | Positive phase − negative phase | "数据期望 − 模型期望"的梯度形式（RBM/能量模型）（第 10 章）。 |
| 伪似然 | Pseudo-likelihood | 用条件分布连乘代替联合似然，绕开配分函数（第 10、11 章）。 |
| 对比散度 | Contrastive divergence | 用短程 MCMC 近似负相梯度训练能量模型（第 10 章）。 |
| Potts 模型 | Potts model | 成对约束下的最大熵分布 $p\propto\exp(\sum h_i+\sum J_{ij})$，DCA 内核（第 10、11 章）。 |
| 伊辛模型 | Ising model | Potts 模型的二态特例（第 10 章）。 |
| 场 | Field ($h_i$) | Potts 模型单点参数，编码保守性（第 10、11 章）。 |
| 耦合 | Coupling ($J_{ij}$) | Potts 模型成对参数，编码直接耦合/共进化（第 10、11 章）。 |
| 表观 / 间接耦合 | Apparent / Indirect coupling | 含传递相关的总相关，互信息看到的信号（第 4、10、11 章）。 |
| 直接耦合 | Direct coupling | 扣除间接路径后的耦合，由 $J_{ij}$（≈精度矩阵）度量（第 10、11 章）。 |
| 传递相关 | Transitive correlation | $i$–$k$–$j$ 链造成 $i,j$ 间的假相关（第 10、11 章）。 |
| 偏相关 / 条件相关 | Partial / Conditional correlation | 给定其余变量后的相关，对应图上"直接边"（第 10、11 章）。 |
| Fisher 信息 | Fisher information | 得分协方差=负对数似然 Hessian 期望，度量参数敏感度（第 10 章）。 |
| 得分函数 | Score function | $s(x;\theta)=\nabla_\theta\log p(x;\theta)$，均值为零（第 10 章）。 |
| Cramér–Rao 下界 | Cramér–Rao lower bound (CRLB) | 无偏估计方差下界 $\mathrm{Var}(\hat\theta)\ge1/\mathcal{I}(\theta)$（第 10 章）。 |
| 信息几何 | Information geometry | 把分布族看作以 Fisher 信息为度量的黎曼流形（第 10 章）。 |
| 自然梯度 | Natural gradient | 用 $\mathcal{I}^{-1}$ 校正梯度方向，在分布空间均匀前进（第 10 章）。 |
| 贝叶斯信息准则 | Bayesian Information Criterion (BIC) | $-2\ln\hat L+k\ln N$，由两部分编码导出的模型选择准则（第 10 章）。 |
| 赤池信息准则 | Akaike Information Criterion (AIC) | $-2\ln\hat L+2k$，关注预测风险的模型选择（第 10 章）。 |
| 贝叶斯证据 | Bayesian evidence / Marginal likelihood | $p(\mathcal{D})$，经 Laplace 近似与 MDL/BIC 一致（第 10 章）。 |
| 压缩即泛化 | Compression = Generalization | 短描述 → 强泛化保证（PAC-Bayes/MDL）（第 10 章）。 |
| 充分性 / 最小性 | Sufficiency / Minimality | 好表示应保留任务信息 $I(T;Y)$、丢弃无关信息 $I(X;T)$（第 10 章）。 |
| 变分信息瓶颈 | Variational Information Bottleneck (VIB) | IB 的可 SGD 训练变分上界形式（第 10 章）。 |
| 信息平面 | Information plane | 训练中 $(I(X;T),I(T;Y))$ 轨迹；"压缩相变"说法有争议（第 10 章）。 |
| 变分推断 | Variational inference | 把"算后验"变成"优化"近似分布 $q_\phi$（第 10 章）。 |
| 变分分布 | Variational distribution | 用来近似真后验的简单分布 $q_\phi(z\mid x)$（第 10 章）。 |
| 重参数化技巧 | Reparameterization trick | $z=\mu+\sigma\odot\epsilon$ 让梯度穿过采样（第 10 章）。 |
| 期望最大化 | Expectation-Maximization (EM) | 后验可解析时在 ELBO 上的坐标上升（第 10 章）。 |
| 后验坍缩 | Posterior collapse | 反向 KL 导致 $q$ 钻进单峰、低估方差的通病（第 10 章）。 |
| 互信息神经估计 | Mutual Information Neural Estimation (MINE) | 用 Donsker–Varadhan 下界训练判别器估 MI（第 10 章）。 |
| Donsker–Varadhan 表示 | Donsker–Varadhan representation | KL 的变分下界表示，MINE 的基础（第 10 章）。 |
| InfoNCE | InfoNCE | $K$ 选 1 softmax 对比损失，MI 下界封顶 $\log K$（第 10 章）。 |
| 对比学习 | Contrastive learning | 拉近正样本、推远负样本，= 最大化视图间 MI 下界（SimCLR/CLIP/MoCo）（第 10 章）。 |
| MI 估计样本下界 | Sample lower bound for MI estimation | 下界 $I$ 比特互信息约需 $e^I$ 样本（McAllester–Stratos）（第 10 章）。 |
| softmax | softmax | $e^{z_c}/\sum e^{z_{c'}}$，= 条件吉布斯分布（logits=负能量）（第 10 章）。 |
| 交叉熵损失 | Cross-entropy loss | = 负对数似然 = 最大似然 = 最小前向 KL = 最大熵分类器（第 4、10 章）。 |
| 最大熵分类器 / 逻辑回归 | Maximum-entropy classifier / Logistic regression | 满足经验特征约束下条件熵最大的分类器，即 softmax 回归（第 10 章）。 |
| 温度 | Temperature | softmax 的 $\tau$（=逆温度倒数），调节确定性 vs 均匀（第 10 章）。 |

---

## 11. 生物信息学与蛋白质折叠应用

| 中文名 | English | 解释 |
|---|---|---|
| 多序列比对 | Multiple Sequence Alignment (MSA) | 对齐的同源序列矩阵，蕴含保守性与共进化信号（第 1–11 章）。 |
| 同源序列 | Homolog | 共享进化起源的序列（"进化亲戚"）（第 11 章）。 |
| 列分布 / 列熵 | Column distribution / Column entropy | MSA 某列氨基酸经验分布及其熵 = 序列保守性度量（第 3、11 章）。 |
| 序列保守性 | Sequence conservation | 某位点跨同源序列的恒定程度，低列熵=高保守（第 3、11 章）。 |
| sequence logo | Sequence logo | 逐列字母堆叠图，总高度=信息含量（第 3、11 章）。 |
| 信息含量 | Information content | 列高 $R=\log\lvert\mathcal{A}\rvert-H=D_{KL}(\hat p\Vert \text{uniform})$（第 3、11 章）。 |
| 相对熵信息含量 | Relative-entropy information content | 扣背景频率版 $D_{KL}(\hat p_i\Vert q)$（第 11 章）。 |
| 熵估计偏差 | Entropy estimation bias | 浅 MSA 插值熵系统性低估熵、高估信息含量（第 5、11 章）。 |
| Miller–Madow 校正 | Miller–Madow correction | 熵估计偏差的一阶修正项（第 11 章）。 |
| 小样本校正 | Small-sample correction | Schneider 的 logo 信息含量修正项 $e(M)$（第 11 章）。 |
| 共进化 | Co-evolution | 接触残基对补偿性协同突变的现象，接触预测信号源（第 1、4、11 章）。 |
| 互信息（共进化） | Mutual information (co-evolution) | $I(\text{col}_i;\text{col}_j)$，共进化原始信号，但含假阳性（第 4、11 章）。 |
| 接触图 | Contact map | $L\times L$ 的残基对是否空间接触的 0/1 图（第 11 章）。 |
| 接触预测 | Contact prediction | 从 MSA 预测哪些残基对在三维空间接触（第 4、11 章）。 |
| 系统发育噪声 / 偏差 | Phylogenetic noise / bias | 序列因共同祖先而非独立，制造与接触无关的背景相关（第 2、4、11 章）。 |
| 序列重加权 | Sequence reweighting | 给近亲扎堆的序列降权，估计有效样本数 $M_{eff}$（第 11 章）。 |
| 有效序列数 | Effective number of sequences ($M_{eff}$) | 校正系统发育冗余后的独立信息量（第 11 章）。 |
| 平均乘积校正 | Average Product Correction (APC) | 从耦合得分里减背景项 $\mathrm{MI}_{i\cdot}\mathrm{MI}_{\cdot j}/\mathrm{MI}_{\cdot\cdot}$（第 11 章）。 |
| 直接耦合分析 | Direct Coupling Analysis (DCA) | 拟合全局 Potts 模型、用 $J_{ij}$ 分离直接耦合做接触预测（第 1、4、10、11 章）。 |
| mfDCA | mean-field DCA | 平均场近似 $J\approx-(\hat C)^{-1}$ 的最早 DCA 实现（第 11 章）。 |
| PSICOV | PSICOV | 稀疏逆协方差估计（graphical lasso）做接触预测（第 11 章）。 |
| plmDCA | pseudo-likelihood maximization DCA | 伪似然 = 逐位置 softmax，绕开配分函数，强基线（第 11 章）。 |
| GREMLIN / CCMpred | GREMLIN / CCMpred | 正则化最大似然/伪似然的工程化 GPU 实现（第 11 章）。 |
| 精度矩阵 | Precision matrix | 协方差的逆，其非零元对应直接边（DCA 的高斯类比）（第 11 章）。 |
| Frobenius 范数 | Frobenius norm | 把耦合矩阵 $J_{ij}$ 压成标量接触得分的常用范数（第 11 章）。 |
| 对数似然比 | Log-likelihood ratio (LLR) | 比对打分本质 $\log\frac{p(a,b)}{q(a)q(b)}$（第 11 章）。 |
| 替换打分矩阵 | Substitution matrix (BLOSUM/PAM) | 氨基酸替换得分=对数似然比，判别力=其相对熵（第 11 章）。 |
| profile / 位置权重矩阵 | Profile / Position Weight Matrix (PWM) | 逐列氨基酸分布的概率刻画（第 11 章）。 |
| profile 隐马尔可夫模型 | profile Hidden Markov Model (profile HMM) | 列发射分布+插入/删除转移的家族概率模型（HMMER/Pfam）（第 11 章）。 |
| 发射分布 | Emission distribution | profile HMM 匹配态的氨基酸分布 $e_i(a)$（第 11 章）。 |
| motif 发现 | Motif discovery | 无监督找反复出现的非随机短模式（MEME）（第 11 章）。 |
| 结合位点信息含量 | Binding site information content | 结合位点总信息含量 ≈ 基因组中唯一定位所需比特（Schneider）（第 11 章）。 |
| 蛋白质语言模型 | Protein language model (pLM) | 海量序列自监督训练的模型（ESM 系列）（第 1、6、10、11 章）。 |
| 掩码语言建模 | Masked language modeling (MLM) | 遮住残基、用上下文预测，训练目标=交叉熵（第 11 章）。 |
| 困惑度 | Perplexity | $2^{\text{每残基交叉熵}}$ = 每残基比特数 = 有效分支数（第 1、3、4、6、10、11 章）。 |
| 每残基比特数 | Bits per residue | 模型平均编码一个残基所需比特，= 交叉熵（第 6、11 章）。 |
| 伪困惑度 | Pseudo-perplexity | MLM 友好的困惑度替代，用条件对数似然累加（第 11 章）。 |
| 突变效应预测 | Variant effect prediction (zero-shot) | 用突变前后对数似然之差预测有害性（惊奇度=反常度）（第 11 章）。 |
| 遗传密码 | Genetic code | 64 密码子→20 氨基酸的冗余映射，有"被动纠错"味道（启发性类比）（第 7、11 章）。 |
| 突变鲁棒性 | Mutational robustness | 同义/保守替换使点突变多落在等价类内的被动容错（第 11 章）。 |
| 信号转导信道容量 | Signal-transduction channel capacity | 把通路建模为信道，实测单通路容量常仅 1–2 比特（严格应用）（第 7、11 章）。 |
| 孤儿蛋白 | Orphan protein | 同源序列极少、浅 MSA 的蛋白，DCA/频率估计易崩（第 11 章）。 |
| 高阶共进化 | Higher-order co-evolution | 三体及以上协同约束，超出成对 Potts 的开放问题（第 11 章）。 |

---

## 12. 人物、里程碑与文献

| 中文名 | English | 解释 |
|---|---|---|
| 香农 | Claude E. Shannon | 信息论创始人，1948《通信的数学理论》（第 1 章）。 |
| 通信的数学理论 | A Mathematical Theory of Communication | 香农 1948 奠基论文，一举建立信息论（第 1 章）。 |
| 奈奎斯特 / 哈特利 | Nyquist / Hartley | 1924/1928 信息度量早期思想（用对数衡量消息数）（第 1 章）。 |
| Jaynes | E. T. Jaynes | 1957 提出最大熵原理，连接统计力学与推断（第 8、10 章）。 |
| Khinchin | A. I. Khinchin | 1953 给出熵的严格公理化（Shannon–Khinchin 定理）（第 3 章）。 |
| Kullback / Leibler | Kullback / Leibler | KL 散度与"信息与充分性"的提出者（第 4 章）。 |
| Huffman | David Huffman | 1952 最优前缀码的构造者（第 6 章）。 |
| Lempel / Ziv | Lempel / Ziv | 1977/78 通用压缩 LZ 系列的提出者（第 6 章）。 |
| Hamming | R. W. Hamming | 1950 汉明码，校正子定位错误（第 7 章）。 |
| McMillan / Breiman | McMillan / Breiman | 把 AEP 推广到平稳遍历源（SMB 定理）（第 5 章）。 |
| Arıkan | Erdal Arıkan | 2009 polar 码，首个证明达容量的显式码（第 7 章）。 |
| Berrou | Claude Berrou | 1993 turbo 码，首次实践逼近容量（第 7 章）。 |
| MacKay | David J. C. MacKay | LDPC 码复兴者，《Information Theory, Inference, and Learning Algorithms》作者（第 7 章）。 |
| Tishby | Naftali Tishby | 信息瓶颈方法的提出者之一（第 9、10 章）。 |
| 信息瓶颈方法 | The Information Bottleneck Method | Tishby–Pereira–Bialek 1999，连接率失真与表示学习（第 9、10 章）。 |
| Cover & Thomas | Cover & Thomas | 《Elements of Information Theory》标准教材（全课程）。 |
| Weigt / Morcos | Weigt / Morcos | DCA 奠基（PNAS 2009/2011）（第 10、11 章）。 |
| ESM | Evolutionary Scale Modeling (ESM) | Meta 的蛋白语言模型系列，交叉熵训练、困惑度评估（第 11 章）。 |
| AlphaFold2 / Evoformer | AlphaFold2 / Evoformer | 端到端"嚼 MSA"学习保守性+共进化的结构预测系统（第 11 章）。 |
| ProteinGym | ProteinGym | 突变效应预测（DMS 实验）的基准（第 11 章）。 |

---

> 备注：本表覆盖第 1–11 章正文首次出现并给出英文的核心术语。阅读英文文献时，请结合上下文确认缩写在具体语境中的指代（如 $N$ 既可指噪声功率也可指符号数，$\beta$ 既是逆温度也是拉格朗日乘子）。
