# 概率与统计课程 · 中英文术语对照表（GLOSSARY）

> 速查约定：每条格式为 **中文名 | English | 一句话精准解释（注所在章）**。
> 分组大致按课程主题；同一术语跨章出现时归入其最核心的主题组，必要时在解释里指出其它出场章。
> 缩写在领域内通用时随术语标注（如 MLE、MAP、CLT、LLN、MCMC、HMC、FDR、FWER、ESS、GLM、UQ、DCA、MSA）。

---

## 目录

1. [概率空间与基础](#1-概率空间与基础)
2. [期望·矩·母函数·不等式](#2-期望矩母函数不等式)
3. [常见分布与方向统计](#3-常见分布与方向统计)
4. [多元分布与高斯图模型](#4-多元分布与高斯图模型)
5. [收敛与极限定理](#5-收敛与极限定理)
6. [点估计与似然](#6-点估计与似然)
7. [贝叶斯推断](#7-贝叶斯推断)
8. [假设检验与多重检验](#8-假设检验与多重检验)
9. [蒙特卡洛与 MCMC](#9-蒙特卡洛与-mcmc)
10. [蛋白质折叠 / 计算生物学应用](#10-蛋白质折叠--计算生物学应用)
11. [人物·文献·里程碑](#11-人物文献里程碑)

---

## 1. 概率空间与基础

| 中文名 | English | 解释 |
|---|---|---|
| 概率（正问题） | Probability / forward problem | 已知模型/分布，推演数据会长什么样（"上帝视角"）；本课前半 2–6 章在搭这套引擎（第 1 章）。 |
| 统计（反问题） | Statistics / inverse problem | 拿到数据，反推模型/参数/不确定性（"凡人视角"）；反问题常是病态的（第 1 章）。 |
| 频率派 | Frequentist | 把参数看作未知但固定的常数，不确定性在估计量的抽样分布里，概率=长程频率（第 1、8 章）。 |
| 贝叶斯派 | Bayesian | 把参数看作随机变量、有先验，不确定性直接在后验上，概率=信念程度（第 1、8 章）。 |
| 似然函数 | Likelihood function | 把"数据概率"表达式中数据固定、参数当自变量得到的函数；连接正反问题的桥梁，不是 θ 的概率分布（第 1、7 章）。 |
| 玻尔兹曼分布 / 吉布斯分布 | Boltzmann / Gibbs distribution | $p(x)\propto e^{-E(x)/k_BT}$，贯穿全课的暗线，一身四任（物理核心、标准分布、采样场景、能量最小↔概率最大）（第 1、3、10 章）。 |
| 配分函数 | Partition function | 玻尔兹曼分布的归一化常数 $Z=\sum_x e^{-E(x)/k_BT}$；高维下算不动，是计算难题的共同根源（第 1、3、10 章）。 |
| 归一化常数 | Normalizing constant | 让分布全体积分/求和为 1 的常数；贝叶斯证据 $p(\text{data})$ 与配分函数 $Z$ 是同一类对象（第 1、2 章）。 |
| 样本空间 | Sample space $\Omega$ | 一次随机试验所有可能结果的集合，其元素称样本点/结果（第 2 章）。 |
| 样本点 / 结果 | Sample point / outcome | 样本空间中的单个元素 $\omega\in\Omega$（第 2 章）。 |
| 事件 | Event | 样本空间的（可测）子集，可被赋予概率（第 2 章）。 |
| σ-代数 | σ-algebra $\mathcal F$ | 对补与可数并封闭的事件族，规定"哪些子集算事件"（第 2 章）。 |
| Borel σ-代数 | Borel σ-algebra | 实数轴上由开集生成的标准 σ-代数，是连续随机变量的事件族（第 2 章）。 |
| 不可测集 | Non-measurable set | 无法自洽地赋"长度/概率"的集合，是需要 σ-代数的根本原因（第 2 章）。 |
| 概率测度 | Probability measure $\mathbb P$ | 把事件映到 [0,1]、满足 Kolmogorov 三公理的集函数（第 2 章）。 |
| Kolmogorov 公理 | Kolmogorov axioms | 非负、归一、可数可加三条公理，是现代概率论的地基（第 2 章）。 |
| 可数可加性 | Countable additivity | 互斥事件可数并的概率等于各概率之和，比有限可加更强（第 2 章）。 |
| 概率测度的连续性 | Continuity of probability | 由可数可加导出：单调事件列的概率收敛到极限事件概率（第 2 章）。 |
| 容斥原理 | Inclusion–exclusion | 用交集修正并集概率的计数公式（第 2 章）。 |
| 条件概率 | Conditional probability | $\mathbb P(A\mid B)=\mathbb P(A\cap B)/\mathbb P(B)$，在已知 B 发生下重标度概率（第 2 章）。 |
| 乘法公式 / 链式法则 | Chain / multiplication rule | 把联合概率拆成条件概率连乘（第 2、5 章）。 |
| 独立 | Independent | $\mathbb P(A\cap B)=\mathbb P(A)\mathbb P(B)$；两两独立不蕴含相互独立（第 2 章）。 |
| 相互独立 | Mutually independent | 任意子集联合概率都等于各概率之积，强于两两独立（第 2 章）。 |
| 全概率公式 | Law of total probability | 按一个划分把事件概率分门别类加起来，是贝叶斯分母的来源（第 2 章）。 |
| 划分 | Partition | 把样本空间分成互斥且穷尽的若干事件（第 2 章）。 |
| 贝叶斯定理 | Bayes' theorem | 后验 ∝ 似然 × 先验；把条件概率反过来用、用全概率处理分母（第 1、2、8 章）。 |
| 边缘化 | Marginalization | 把不关心的变量"积掉/加掉"得到边缘分布（第 2、5 章）。 |
| 基础率谬误 | Base rate fallacy | 忽略低先验（基础率）而高估后验的常见错误，医学检验/变体识别里能救命（第 2 章）。 |
| 随机变量 | Random variable | 从样本空间到实数的可测函数 $X:\Omega\to\mathbb R$；须与其取值 $x$ 严格区分（第 2 章）。 |
| 可测性 | Measurability | 随机变量的定义性质：原像把 Borel 集拉回到事件（第 2 章）。 |
| 分布 / 律 | Distribution / law | 随机变量在取值空间上诱导的概率 $\mathbb P(X\in B)$（第 2 章）。 |
| 累积分布函数 | Cumulative distribution function (CDF) | $F(x)=\mathbb P(X\le x)$，单调不减、右连续、两端趋 0/1，统一离散/连续/混合（第 2 章）。 |
| 概率质量函数 | Probability mass function (pmf) | 离散分布 $p(x)=\mathbb P(X=x)$，非负且求和为 1（第 2 章）。 |
| 概率密度函数 | Probability density function (pdf) | 连续分布的密度 $f(x)$，非负、积分为 1、可大于 1，单点概率为 0（第 2 章）。 |
| 混合型分布 | Mixed distribution | 既含点质量又含密度的分布（第 2 章）。 |
| 分位数函数 / 逆 CDF | Quantile function / inverse CDF | CDF 的反函数，逆变换采样的基础（第 2、10 章）。 |
| 基测度 / 主测度 | Base / dominating measure | 统一离散（计数测度）与连续（Lebesgue）的参照测度，密度是 Radon–Nikodym 导数（第 2、4 章）。 |
| 雅可比 | Jacobian | 一维变量替换中密度需乘的 $|g'|$ 因子（第 2 章）。 |
| 雅可比矩阵 / 行列式 | Jacobian matrix / determinant | 多维变量替换中体积缩放因子 $|\det J|$（第 2、5 章）。 |
| 联合 CDF | Joint CDF | 随机向量的累积分布函数，统一刻画多维分布（第 2、5 章）。 |
| 测度论 | Measure theory | 为概率提供严格地基的数学分支；本课"严格而够用"，不陷入其重机械（第 2 章）。 |

---

## 2. 期望·矩·母函数·不等式

| 中文名 | English | 解释 |
|---|---|---|
| 期望 | Expectation / expected value | 分布的加权平均 $\mathbb E[g(X)]$；是关于分布的线性泛函（第 3 章）。 |
| LOTUS / 无意识统计学家定律 | Law of the Unconscious Statistician | 不必先求 $g(X)$ 的分布，直接 $\mathbb E[g(X)]=\sum g(x)p(x)$ 算期望（第 3 章）。 |
| 线性性 | Linearity (of expectation) | $\mathbb E[aX+bY]=a\mathbb E X+b\mathbb E Y$，无需独立的"超级武器"（第 3 章）。 |
| 指示变量 | Indicator variable | 取 0/1 的随机变量，配线性性可巧算计数期望（第 3 章）。 |
| 柯西主值 | Principal value | 奇函数积分的对称化取值；柯西分布期望不存在的反面教材（第 3 章）。 |
| 方差 | Variance | $\mathrm{Var}(X)=\mathbb E[(X-\mu)^2]$，刻画分布的离散程度（第 3 章）。 |
| 标准差 | Standard deviation | 方差的平方根，与原变量同量纲（第 3 章）。 |
| 协方差 | Covariance | $\mathrm{Cov}(X,Y)$ 刻画两变量的线性协同；双线性、方差是其特例（第 3、5 章）。 |
| 相关系数 | (Pearson) correlation coefficient | 标准化协方差，$|\rho|\le1$；不相关 ≠ 独立（第 3 章）。 |
| 矩 | Moment | $\mathbb E[X^k]$（原点矩）或中心矩，刻画分布形状（第 3 章）。 |
| 原点矩 / 中心矩 | Raw / central moment | 关于 0 或关于均值的 $k$ 阶矩（第 3 章）。 |
| 标准化矩 | Standardized moment | 用标准差归一的中心矩，给出无量纲形状量（第 3 章）。 |
| 偏度 | Skewness | 三阶标准化矩，刻画分布不对称性（第 3 章）。 |
| 峰度 / 超额峰度 | Kurtosis / excess kurtosis | 四阶标准化矩；正态峰度为 3，超额峰度减 3（第 3 章）。 |
| 矩母函数 | Moment generating function (MGF) | $M(t)=\mathbb E[e^{tX}]$，求导取矩、相乘求独立和；可能不存在（第 3 章）。 |
| 特征函数 | Characteristic function | $\varphi(t)=\mathbb E[e^{itX}]$，永远存在、是分布的傅里叶变换，CLT 的工具（第 3、6 章）。 |
| 概率母函数 | Probability generating function (PGF) | $\mathbb E[t^X]$，离散计数与分支过程的利器（第 3 章）。 |
| 累积量母函数 | Cumulant generating function (CGF) | $\log M(t)$，生成累积量；配分函数 $\log Z$ 是其物理化身（第 3 章）。 |
| 再生性 | Reproductive property | 同族独立和仍属同族（如正态+正态、泊松+泊松）（第 3、4 章）。 |
| 条件期望 | Conditional expectation | $\mathbb E[Y\mid X]$ 是一个随机变量，也是均方最优预测（第 3 章）。 |
| 塔性质 / 全期望公式 | Tower property / law of total expectation | $\mathbb E[\mathbb E[Y\mid X]]=\mathbb E[Y]$（第 3 章）。 |
| 全方差公式 | Law of total variance | 把方差拆成"组内期望方差 + 组间均值方差"（第 3 章）。 |
| $L^2$ 投影 / 正交投影 | $L^2$ / orthogonal projection | 条件期望是 Y 在某子空间上的最小二乘投影（第 3 章）。 |
| Markov 不等式 | Markov inequality | 非负变量尾概率 $\le \mathbb E X/t$，最弱最通用（第 3 章）。 |
| Chebyshev 不等式 | Chebyshev inequality | 用方差界尾概率，是弱大数定律的引擎（第 3、6 章）。 |
| Jensen 不等式 | Jensen inequality | 凸函数 $\mathbb E[g(X)]\ge g(\mathbb E X)$；EM 与 KL 非负的根源（第 3、7 章）。 |
| Cauchy–Schwarz 不等式 | Cauchy–Schwarz inequality | 内积/期望版的对偶界，相关系数有界由它来（第 3 章）。 |
| Hölder 不等式 | Hölder inequality | Cauchy–Schwarz 的推广，配对偶指数（第 3 章）。 |
| 布尔不等式 / Union bound | Union bound | $\mathbb P(\cup A_i)\le\sum\mathbb P(A_i)$，多重检验 Bonferroni 的源头（第 3、9 章）。 |
| Chernoff 界 | Chernoff bound | 用 MGF 优化出指数尾界（第 3 章）。 |
| Hoeffding 不等式 | Hoeffding inequality | 有界独立和的指数集中不等式（第 3、6 章）。 |
| 集中不等式 | Concentration inequality | "偏离均值多远的概率有多小"的一类界，高维统计的引擎（第 3、6 章）。 |
| 速率函数 / 大偏差 | Rate function / large deviations | 尾概率指数衰减速率，大偏差理论的核心量（第 3、6 章）。 |
| 涨落–耗散关系 | Fluctuation–dissipation relation | 热容 = 能量方差 / $k_BT^2$，把统计与物理量焊在一起（第 3 章）。 |
| 重尾 | Heavy tail | 尾部衰减慢、矩可能不存在的分布（如柯西），会让 CLT 失效（第 3、4 章）。 |

---

## 3. 常见分布与方向统计

| 中文名 | English | 解释 |
|---|---|---|
| 伯努利分布 | Bernoulli distribution | 一次抛硬币、取 0/1 的分布；位点是否保守的指示变量（第 4 章）。 |
| 二项分布 | Binomial distribution | $n$ 次独立伯努利的成功数（第 4 章）。 |
| 类别分布 | Categorical distribution | 伯努利的 $K$ 元推广，一次掷 K 面骰子（第 4 章）。 |
| 多项分布 | Multinomial distribution | 类别分布的 $n$ 次重复计数；MSA 某列就是一个多项样本（第 4 章）。 |
| 几何分布 | Geometric distribution | 等到第一次成功的试验次数，具无记忆性（第 4 章）。 |
| 负二项分布 | Negative binomial distribution | 等到第 r 次成功的次数；现代用作过度离散计数模型（第 4 章）。 |
| 过度离散 | Overdispersion | 方差超过泊松/二项所预测，RNA-seq 计数常见（第 4、8 章）。 |
| 泊松分布 | Poisson distribution | 稀有事件计数，是二项的稀有极限（第 4 章）。 |
| 稀有事件定律 | Law of rare events / Poisson limit | 大 $n$ 小 $p$ 下二项趋于泊松（第 4 章）。 |
| 均匀分布 | Uniform distribution | "最大无知"的基准分布，逆变换采样的起点（第 4 章）。 |
| 正态分布 / 高斯分布 | Normal / Gaussian distribution | 钟形分布，是 CLT 的共同归宿与万物的近似（第 4 章）。 |
| 指数分布 | Exponential distribution | 连续世界的无记忆等待时间（第 4 章）。 |
| 无记忆性 | Memorylessness | 已等待时长不影响未来等待分布（几何/指数特有）（第 4 章）。 |
| Gamma 分布 | Gamma distribution | 正支撑等待时间族，含指数与卡方为特例，有形状–速率/形状–尺度两套参数化（第 4 章）。 |
| Beta 分布 | Beta distribution | $[0,1]$ 上的分布，"概率的概率"，二项的共轭先验（第 4、8 章）。 |
| Dirichlet 分布 | Dirichlet distribution | 单纯形上的分布，"概率向量的分布"，多项的共轭先验（第 4、8 章）。 |
| 单纯形 | Simplex | 概率向量所在的 $K-1$ 维约束空间（第 4 章）。 |
| 伪计数 | Pseudocount | 加到频数上避免"零概率"的小量，本质是 Dirichlet 先验（第 4、7、8 章）。 |
| 卡方分布 | Chi-square distribution | 正态平方和，拟合优度/Wilks 检验的零分布（第 4、9 章）。 |
| Student-t 分布 | Student's t-distribution | 正态比卡方根，小样本均值推断用；重尾（第 4 章）。 |
| F 分布 | F-distribution | 两个卡方之比，方差比较与 ANOVA 用（第 4 章）。 |
| Laplace 分布 | Laplace distribution | 双指数、尖峰重尾，对应 L1 正则/lasso 的先验（第 4 章）。 |
| 方向 / 圆周统计 | Directional / circular statistics | 处理角度等周期量的统计分支，本课"独门特色"（第 4 章）。 |
| 圆周均值 | Circular mean | 用合向量定义的角度平均，绕过 0°/360° 拼接问题（第 4 章）。 |
| 平均合向量长度 | Mean resultant length $R$ | 角度集中度的度量，$R\to1$ 越集中（第 4 章）。 |
| von Mises 分布 | von Mises distribution | 圆周上的"高斯"，参数为均值方向 $\mu$ 与集中度 $\kappa$；二面角 $\phi,\psi$ 的天然分布（第 4 章）。 |
| 集中度参数 | Concentration parameter $\kappa$ | von Mises 的"精度"（方差倒数类比），$\kappa$ 越大越尖（第 4 章）。 |
| 修正贝塞尔函数 | Modified Bessel function | von Mises 归一化常数 $I_0(\kappa)$ 用到的特殊函数（第 4 章）。 |
| 环面上的二元 von Mises | Bivariate von Mises (on torus) | 联合建模 $(\phi,\psi)$ 的环面分布（sine/cosine 变体）（第 4 章）。 |
| von Mises 混合 | Mixture of von Mises | 用多个 von Mises 描述拉氏图多个区域（第 4 章）。 |
| 环面 | Torus | 两个角的联合取值空间（第 4 章）。 |
| 指数族 | Exponential family | $p(x)\propto e^{\eta\cdot T(x)}$ 形式的分布大家庭，玻尔兹曼/softmax 都在其中（第 4、7 章）。 |
| 自然参数 / 标准参数 | Natural / canonical parameter | 指数族指数里与充分统计量配对的参数 $\eta$（第 4 章）。 |
| 充分统计量 | Sufficient statistic | 数据中对参数有用信息的无损压缩，指数族里现成写在指数里（第 4、7 章）。 |
| 对数配分函数 / 累积函数 | Log-partition / cumulant function | 指数族归一化的对数，凸函数，"求导=矩"（第 4 章）。 |
| 底测度 / 载体 | Base measure / carrier | 指数族里不依赖参数的参考测度部分（第 4 章）。 |
| 共轭先验 | Conjugate prior | 使后验与先验同族、更新只改参数的先验（如 Beta–二项）（第 4、8 章）。 |
| 分布关系网 | Map of distributions | 把各分布按"极限/共轭/构造"关系连成一张图（第 4 章）。 |

---

## 4. 多元分布与高斯图模型

| 中文名 | English | 解释 |
|---|---|---|
| 随机向量 | Random vector | 由多个随机变量组成的向量，有联合分布（第 5 章）。 |
| 联合分布 | Joint distribution | 一组变量同时取值的分布（第 5 章）。 |
| 边缘分布 | Marginal distribution | 把不关心的维度积掉得到的子集分布（第 5 章）。 |
| 条件分布 | Conditional distribution | 固定一部分变量、问另一部分的分布（第 5 章）。 |
| 条件独立 | Conditional independence | 给定第三变量后两变量独立；高斯图模型的核心（第 5 章）。 |
| 协方差矩阵 | Covariance matrix $\Sigma$ | 刻画整组变量两两耦合的对称半正定矩阵（第 5 章）。 |
| 半正定 | Positive semi-definite (PSD) | 协方差矩阵的必备性质，保证方差非负（第 5 章）。 |
| 相关矩阵 | Correlation matrix | 把协方差按标准差标准化后的矩阵（第 5 章）。 |
| 多元高斯 | Multivariate Gaussian $\mathcal N(\mu,\Sigma)$ | 高维世界的"工作母机"，仿射/求和/边缘/条件皆封闭（第 5 章）。 |
| 马氏距离 | Mahalanobis distance | $\sqrt{(x-\mu)^\top\Sigma^{-1}(x-\mu)}$，高斯等高线的"尺子"（第 5 章）。 |
| 仿射封闭性 | Affine closure | 高斯线性变换后仍是高斯（第 5 章）。 |
| Schur 补 | Schur complement | 高斯条件协方差的表达式，分块求逆的核心（第 5 章）。 |
| 精度矩阵 | Precision matrix $\Lambda=\Sigma^{-1}$ | 协方差的逆，其零元素 ⟺ 条件独立；对应"直接耦合"（第 5、11 章）。 |
| 信息形式 / 规范形式 | Information / canonical form | 高斯用 $\Lambda$ 与 $\Lambda\mu$ 表示的参数化，便于条件运算（第 5 章）。 |
| 高斯图模型 | Gaussian graphical model | 用精度矩阵稀疏结构编码条件独立图的模型（第 5 章）。 |
| 偏相关 | Partial correlation | 控制其它变量后两变量的相关，由精度矩阵给出（第 5 章）。 |
| 直接 vs 间接耦合 | Direct vs indirect coupling | 精度矩阵剥离传递相关污染，区分真接触与链式假象（第 5、11 章）。 |
| 白化 | Whitening | 线性变换把椭球协方差揉成各向同性球（第 5 章）。 |
| 主成分分析 | Principal component analysis (PCA) | 协方差矩阵的特征分解，找最大方差方向（第 5 章）。 |
| 弹性网络模型 | Elastic network model | 把蛋白涨落建成多元高斯的粗粒化模型（第 5 章）。 |
| 高斯混合模型 | Gaussian mixture model (GMM) | 多个高斯加权混合，描述多峰分布；EM 拟合（第 5、7 章）。 |
| 多峰性 | Multimodality | 分布有多个峰，单高斯无法刻画（第 5、10 章）。 |
| 椭圆分布 / copula | Elliptical distribution / copula | 高斯失效时建模相依结构的"逃生口"（第 5 章）。 |
| 收缩估计（协方差） | Shrinkage estimation (Ledoit–Wolf) | 高维下把样本协方差向结构目标收缩以稳健（第 5 章）。 |

---

## 5. 收敛与极限定理

| 中文名 | English | 解释 |
|---|---|---|
| 收敛模式 | Modes of convergence | 几乎必然/依概率/$L^p$/依分布四种"趋近"的精确含义（第 6 章）。 |
| 几乎必然收敛 | Almost sure convergence | 路径以概率 1 收敛，最强的收敛模式（第 6 章）。 |
| 依概率收敛 | Convergence in probability | 偏差超阈值的概率趋 0，弱大数定律的收敛（第 6 章）。 |
| 均方收敛 | Mean-square convergence | $L^2$ 收敛，$\mathbb E[|X_n-X|^2]\to0$（第 6 章）。 |
| 依分布收敛 | Convergence in distribution | 只比较 CDF，最弱、唯一不需共同概率空间的模式（第 6 章）。 |
| 大数定律 | Law of large numbers (LLN) | 样本均值收敛到真值，"多次平均可信"的定理（第 6 章）。 |
| 弱大数定律 | Weak LLN (WLLN) | 依概率收敛版，两行 Chebyshev 即证（第 6 章）。 |
| 强大数定律 | Strong LLN (SLLN) | 几乎必然收敛版（第 6 章）。 |
| 经验分布 / 经验 CDF | Empirical distribution / empirical CDF | 把数据当离散分布；Glivenko–Cantelli 保证它一致逼近真 CDF（第 6 章）。 |
| 中心极限定理 | Central limit theorem (CLT) | 标准化样本均值的涨落收敛到高斯，解释误差棒/置信区间（第 6 章）。 |
| 普适性 | Universality | 无论原分布是什么，求和+缩放后都趋向高斯（第 6 章）。 |
| Lindeberg–Feller 条件 | Lindeberg–Feller condition | 非同分布 CLT 成立的"无个体主导"条件（第 6 章）。 |
| Lyapunov 条件 | Lyapunov condition | 比 Lindeberg 更易验证的非同分布 CLT 充分条件（第 6 章）。 |
| 多元 CLT | Multivariate CLT | 向量均值的多维高斯极限（第 6 章）。 |
| Cramér–Wold 器件 | Cramér–Wold device | 把多元收敛化为所有一维投影收敛（第 6 章）。 |
| Berry–Esseen 定理 | Berry–Esseen theorem | 给出 CLT 的有限样本收敛速率（$1/\sqrt n$）（第 6 章）。 |
| Delta 方法 | Delta method | 把 CLT 通过一阶泰勒传给非线性函数，工程称"误差传播"（第 6 章）。 |
| 二阶 Delta 方法 | Second-order delta method | 一阶导为零时用二阶项，极限是卡方而非高斯（第 6 章）。 |
| 方差稳定变换 | Variance-stabilizing transformation | 用变换让方差与均值脱钩（如 logit、arcsin）（第 6 章）。 |
| Slutsky 定理 | Slutsky's theorem | 收敛序列的和/积的极限规则，CLT 证明常用（第 6 章）。 |
| sub-Gaussian 变量 | sub-Gaussian variable | 尾部不重于高斯的变量，高维集中的标准假设（第 6 章）。 |
| Bernstein 不等式 | Bernstein inequality | 用方差的集中不等式，对低概率事件更紧（第 6 章）。 |
| McDiarmid 不等式 | McDiarmid / bounded-difference inequality | 有界差分函数的集中，超越"均值"形式（第 6 章）。 |
| 标准误 | Standard error | 估计量抽样分布的标准差，误差棒的来源（第 6 章）。 |
| 自助法 / 重采样 | Bootstrap / resampling | 有放回重抽样近似抽样分布，不靠解析公式（第 6、9 章）。 |
| 渐近 vs 非渐近 | Asymptotic vs non-asymptotic | 大样本极限刻画 vs 有限样本的明确界（第 6 章）。 |

---

## 6. 点估计与似然

| 中文名 | English | 解释 |
|---|---|---|
| 估计量 | Estimator | 由数据构造、用来估参数的统计量；本身是随机变量（第 7 章）。 |
| 抽样分布 | Sampling distribution | 估计量随数据随机而产生的分布，频率派不确定性的所在（第 7 章）。 |
| 偏差 | Bias | 估计量期望与真值之差，零则称无偏（第 7 章）。 |
| 无偏 | Unbiased | $\mathbb E[\hat\theta]=\theta$（第 7 章）。 |
| 均方误差 | Mean squared error (MSE) | 偏差²+方差，综合评价估计量（第 7 章）。 |
| 偏差–方差权衡 | Bias-variance tradeoff | 模型复杂度在欠拟合与过拟合间的折中（第 7、11 章）。 |
| 一致性 | Consistency | 样本越多估计越逼近真值（依概率收敛到 θ）（第 7 章）。 |
| 矩估计法 | Method of moments (MoM) | 令样本矩等于理论矩反解参数，最朴素的估计法（第 7 章）。 |
| 极大似然估计 | Maximum likelihood estimation (MLE) | 取让数据最"像样"的参数，几乎一切拟合的底层（第 7 章）。 |
| 对数似然 | Log-likelihood | 似然取对数，把乘积变和、便于求导与防下溢（第 7 章）。 |
| 交叉熵 | Cross-entropy | 经验分布对模型的交叉熵 = 负对数似然，深度学习损失的本体（第 7 章）。 |
| 不变性 | Invariance (of MLE) | 参数函数的 MLE 等于 MLE 的函数（第 7 章）。 |
| 渐近正态 | Asymptotic normality | MLE 大样本下近似高斯，置信区间的依据（第 7 章）。 |
| 渐近有效 | Asymptotic efficiency | MLE 渐近达到 Cramér–Rao 下界（第 7 章）。 |
| 正则性条件 | Regularity conditions | 保证 MLE 渐近三件套成立的光滑/可识别等前提（第 7 章）。 |
| 模型设定错误 | Misspecification | 假设模型与真分布不符，估计性质会变（第 7 章）。 |
| 因子分解定理 | Fisher–Neyman factorization theorem | 似然能按充分统计量分解的判据（第 7 章）。 |
| Rao–Blackwell 定理 | Rao–Blackwell theorem | 对充分统计量取条件期望可改良估计量（第 7 章）。 |
| 最小方差无偏估计 | Minimum-variance unbiased estimator (MVUE) | 无偏估计中方差最小者（第 7 章）。 |
| 得分函数 | Score function | 对数似然对参数的梯度，令其为零求 MLE（第 7 章）。 |
| Fisher 信息 | Fisher information | 得分方差/对数似然曲率，度量数据含的参数信息，也是局部 KL 曲率（第 7 章）。 |
| Cramér–Rao 下界 | Cramér–Rao lower bound (CRLB) | 无偏估计方差的物理下限 $1/\mathcal I(\theta)$（第 7 章）。 |
| 效率 | Efficiency | 估计量方差相对 CRLB 的接近程度（第 7 章）。 |
| 矩匹配 | Moment matching | 指数族 MLE 的一阶条件：模型期望统计量 = 经验统计量（第 7 章）。 |
| 隐变量 | Latent / hidden variable | 未观测但参与生成数据的变量，使似然难算（第 7 章）。 |
| EM 算法 | Expectation–maximization (EM) | 交替 E 步（求责任）与 M 步（重估参数）的隐变量 MLE 引擎，单调上升（第 7 章）。 |
| 证据下界 | Evidence lower bound (ELBO) | 用 Jensen 给 log∑ 造的可优化下界，EM/变分推断的核心目标（第 7、11 章）。 |
| 变分推断 | Variational inference | 用可优化下界近似难算的后验/似然（第 7、11 章）。 |
| 信息几何 | Information geometry | 把参数空间视为以 Fisher 信息为度规的黎曼流形（第 7 章）。 |
| 自然梯度 | Natural gradient | 按 Fisher 度规修正的梯度方向（第 7 章）。 |
| 惩罚似然 / 正则化 | Penalized likelihood / regularization | 给似然加罚项控制方差，等价于 MAP（第 7 章）。 |
| 岭回归 / Tikhonov | Ridge / Tikhonov ($L_2$) | L2 罚 = 高斯先验下的 MAP（第 7、8、11 章）。 |
| lasso | Lasso ($L_1$) | L1 罚 = 拉普拉斯先验，诱导稀疏（第 7、11 章）。 |
| 伪似然 | Pseudo-likelihood | 用条件似然之积绕开难算的配分函数；plmDCA 的核心（第 7、10、11 章）。 |
| 对比散度 | Contrastive divergence | 训练能量模型时近似似然梯度的方法（第 7 章）。 |
| 复合似然 | Composite likelihood | 拼接局部似然得到的一致但非全似然估计（第 11 章）。 |

---

## 7. 贝叶斯推断

| 中文名 | English | 解释 |
|---|---|---|
| 贝叶斯推断 | Bayesian inference | 用贝叶斯定理把先验更新成后验的整套世界观（第 8 章）。 |
| 先验分布 | Prior distribution | 看数据前对参数的信念 $p(\theta)$（第 8 章）。 |
| 后验分布 | Posterior distribution | 看数据后的信念 $p(\theta\mid\text{data})\propto$ 似然×先验（第 8 章）。 |
| 后验预测分布 | Posterior predictive distribution | 对后验积分得到的新数据预测分布（第 8 章）。 |
| 边际似然 / 证据 | Marginal likelihood / evidence | $p(\text{data})$，模型对数据的整体预测力，模型比较的核心（第 8 章）。 |
| 共轭先验 | Conjugate prior | 使后验与先验同族的先验，更新只改超参数（第 4、8 章）。 |
| 伪观测 / 伪计数 | Pseudo-observations / pseudo-counts | 共轭先验把先验解读为"虚拟数据"加进充分统计量（第 8 章）。 |
| Beta–二项 | Beta–Binomial | 抛硬币的共轭对，后验仍是 Beta（第 8 章）。 |
| Dirichlet–多项 | Dirichlet–Multinomial | 多类计数的共轭对，MSA 频率+伪计数的来源（第 8 章）。 |
| Normal–Normal | Normal–Normal | 高斯均值的自共轭更新（第 8 章）。 |
| Gamma–泊松 | Gamma–Poisson | 计数速率的共轭对，后验预测为负二项（第 8 章）。 |
| Normal–Inverse–Gamma | Normal–Inverse–Gamma (NIG) | 同时未知均值与方差时的共轭先验（第 8 章）。 |
| 无信息先验 / 均匀先验 | Uninformative / flat prior | 想"什么都不假设"的先验，可能不可归一（第 8 章）。 |
| 非正常先验 | Improper prior | 积分发散、非合法分布的先验，需谨慎使用（第 8 章）。 |
| Jeffreys 先验 | Jeffreys prior | 参数化不变的"客观"先验，$\propto\sqrt{\det\mathcal I(\theta)}$（第 8 章）。 |
| 参考先验 | Reference prior | 最大化先验到后验信息增益的客观先验族（第 8 章）。 |
| 弱信息先验 | Weakly informative prior | 现代实践主流，温和约束而不强行注入信念（第 8 章）。 |
| 最大后验估计 | Maximum a posteriori (MAP) | 后验众数 = 带先验的 MLE；正则化的贝叶斯身份（第 7、8 章）。 |
| 后验均值 / 中位数 | Posterior mean / median | 后验的两种点估计，分别最小化平方/绝对损失（第 8 章）。 |
| 可信区间 | Credible interval | 后验概率为 95% 的区间，可正确说"参数有 95% 概率在其中"（第 8、9 章）。 |
| 等尾区间 / 最高后验密度区间 | Equal-tailed / highest posterior density (HPD) | 两种构造可信区间的方式（第 8 章）。 |
| 层次贝叶斯模型 | Hierarchical / multilevel Bayesian model | 用上层先验让多个单位互相借力、共享统计强度（第 8 章）。 |
| 汇合 | Pooling (no/complete/partial) | 各管各/完全合并/部分合并三种处理多组数据的方式（第 8 章）。 |
| 收缩 | Shrinkage | 让信息少的单位向整体均值靠拢，常更准（James–Stein）（第 8 章）。 |
| 经验贝叶斯 | Empirical Bayes (EB) | 用数据估先验超参数的捷径（limma 等），会低估不确定性（第 8 章）。 |
| 批次效应 | Batch effect | 层次模型可自动校正的多批次系统偏差（第 8 章）。 |
| 贝叶斯因子 | Bayes factor | 两模型边际似然之比，贝叶斯式模型比较（第 8 章）。 |
| Occam 剃刀 | Occam's razor | 边际似然自动惩罚过复杂模型的内禀机制（第 8 章）。 |
| 信息准则 | Information criteria (AIC / BIC / WAIC) | 近似样本外预测误差以做模型选择的准则（第 8 章）。 |
| 留一交叉验证 | Leave-one-out cross-validation (LOO-CV) | 与 WAIC/AIC 渐近等价的预测误差估计（第 8 章）。 |
| 认知不确定性 | Epistemic uncertainty | 因知识/数据不足、可被更多数据消减的不确定性（第 8、11 章）。 |
| 偶然不确定性 | Aleatoric uncertainty | 数据内禀噪声、不可被消减的不确定性（第 11 章）。 |
| 似然原理 | Likelihood principle | 推断只应依赖观测到的似然，与停止规则无关（第 8 章）。 |
| 停止规则 / 可选停止 | Stopping rule / optional stopping | 数据收集何时停的规则；影响频率派 p 值而非似然（第 8、9 章）。 |
| 贝叶斯决策论 | Bayesian decision theory | 在损失函数下用后验做最优决策的框架（第 8 章）。 |
| 后验预测检查 | Posterior predictive check | 用后验生成的数据自检模型是否合身（第 8 章）。 |

---

## 8. 假设检验与多重检验

| 中文名 | English | 解释 |
|---|---|---|
| 假设检验 | Hypothesis testing | 回答"有没有信号"的频率派框架（第 9 章）。 |
| 原假设 | Null hypothesis $H_0$ | "无效应/无关联"的默认假设（第 9 章）。 |
| 备择假设 | Alternative hypothesis $H_1$ | 与原假设对立的"有效应"假设（第 9 章）。 |
| 检验统计量 | Test statistic | 反映"数据离 $H_0$ 多远"的数（第 9 章）。 |
| 零分布 | Null distribution | $H_0$ 为真时检验统计量的分布，量化"这么极端"的标尺（第 9 章）。 |
| p 值 | p-value | $H_0$ 下出现当前或更极端数据的概率，$P(\text{data}\mid H_0)$，不是 $P(H_0\mid\text{data})$（第 9 章）。 |
| 显著性水平 | Significance level $\alpha$ | 容忍的假阳性率阈值（第 9 章）。 |
| 拒绝域 | Rejection region | 落入即拒绝 $H_0$ 的统计量取值集合（第 9 章）。 |
| I 型错误 / 假阳性 | Type I error / false positive | $H_0$ 真却拒绝它（第 9 章）。 |
| II 型错误 / 假阴性 | Type II error / false negative | $H_1$ 真却没拒绝 $H_0$（第 9 章）。 |
| 功效 | Power | 真有效应时正确拒绝的概率 = 真阳性率（第 9 章）。 |
| 功效分析 | Power analysis | 估计为达目标功效所需样本量/效应（第 9 章）。 |
| 效应量 | Effect size | 效应的大小本身，统计显著不等于实际重要（第 9 章）。 |
| 等效性检验 | Equivalence testing | 主张"两者实际等效"需专门设计的检验（第 9 章）。 |
| Neyman–Pearson 引理 | Neyman–Pearson lemma | 简单假设下似然比检验是最强检验（第 9 章）。 |
| 似然比 | Likelihood ratio | 两假设似然之比，最优检验统计量（第 9 章）。 |
| 一致最强检验 | Uniformly most powerful test (UMP) | 对整个备择都最强的检验，常不存在（第 9 章）。 |
| 广义似然比检验 | Generalized likelihood ratio test (GLRT/LRT) | 复合假设的通用检验（第 9 章）。 |
| Wilks 定理 | Wilks' theorem | $-2\log\lambda$ 渐近服从卡方分布（第 9 章）。 |
| z 检验 / t 检验 | z-test / t-test | 比较均值；t 检验小样本用、对正态偏离较稳健（第 9 章）。 |
| Welch t 检验 | Welch's t-test | 不等方差版 t 检验，默认推荐（第 9 章）。 |
| 卡方检验 | Chi-squared test | 拟合优度与独立性检验（第 9 章）。 |
| 比例检验 | Proportion test | 比较成功比例（第 9 章）。 |
| 方差分析 | Analysis of variance (ANOVA) | 比较多组均值，F 检验是嵌套线性模型 LRT 的特例（第 9 章）。 |
| 置信区间 | Confidence interval (CI) | 频率派区间估计；95% 指"程序"的长程覆盖率，不是这个区间的概率（第 9 章）。 |
| 覆盖率 | Coverage | 构造区间程序长程盖住真值的比例（第 9、11 章）。 |
| 枢轴量 | Pivotal quantity | 分布不依赖未知参数的量，用来构造 CI（第 9 章）。 |
| Wald 区间 | Wald interval | 用 MLE 渐近正态 $\hat\theta\pm z^*\widehat{\mathrm{SE}}$ 造的 CI（第 9 章）。 |
| 置换检验 | Permutation test | 打乱标签生成零分布，几乎不依赖分布假设（第 9 章）。 |
| 可交换性 | Exchangeability | 置换检验的核心前提："标签与数值无关"（第 9 章）。 |
| Bootstrap 置信区间 | Bootstrap CI (percentile / basic / BCa) | 用重采样造 CI，BCa 最稳健（第 9 章）。 |
| 多重比较问题 | Multiple comparisons problem | 同时做大量检验导致假阳性失控（第 9 章）。 |
| 族错误率 | Family-wise error rate (FWER) | 犯哪怕一个假阳性的概率（第 9、11 章）。 |
| Bonferroni 校正 | Bonferroni correction | 把阈值除以检验数控制 FWER，极保守（第 9 章）。 |
| 错误发现率 | False discovery rate (FDR) | 宣称的发现里假阳性占的期望比例，高通量筛选的核心纪律（第 9、11 章）。 |
| Benjamini–Hochberg 程序 | Benjamini–Hochberg (BH) | 控制 FDR 的经典升序阈值算法（独立/正相关下有效）（第 9 章）。 |
| Benjamini–Yekutieli | Benjamini–Yekutieli (BY) | 任意相依结构下仍控 FDR 的更保守版本（第 11 章）。 |
| PRDS | Positive regression dependency on subsets | BH 有效所需的正相依条件（第 9、11 章）。 |
| q 值 / local FDR | q-value / local FDR | 单个发现对应的最小 FDR 阈值/局部假阳率（第 9 章）。 |
| p-hacking / 数据窥探 | p-hacking / data dredging | 反复试到"显著"为止，破坏 p 值含义（第 9 章）。 |
| 研究者自由度 | Researcher degrees of freedom | 分析中可调的选择，无意中制造假显著（第 9 章）。 |
| HARKing | Hypothesizing After Results are Known | 看到结果后再编假设，事后合理化（第 9 章）。 |
| 发表偏倚 | Publication bias | 只有显著结果被发表，扭曲文献（第 9 章）。 |
| 预注册 | Pre-registration | 事先登记假设与分析，遏制 p-hacking（第 9 章）。 |
| 可重复性危机 | Reproducibility crisis | 大量已发表结果无法被独立复现的现象（第 9 章）。 |
| 检察官谬误 | Prosecutor's fallacy | 把 $P(\text{data}\mid H_0)$ 误当 $P(H_0\mid\text{data})$（第 9 章）。 |

---

## 9. 蒙特卡洛与 MCMC

| 中文名 | English | 解释 |
|---|---|---|
| 维度诅咒 | Curse of dimensionality | 网格积分随维数指数爆炸而不可行（第 10 章）。 |
| 蒙特卡洛积分 | Monte Carlo (MC) integration | 用样本均值估期望/积分，误差 $\sigma/\sqrt N$ 与维度无关（第 10 章）。 |
| 逆变换采样 | Inverse transform sampling | 用 CDF 反函数把均匀样本变成目标样本（第 10 章）。 |
| 拒绝采样 | Rejection sampling | 在包络分布下打靶、按比例接受（第 10 章）。 |
| 重要性采样 | Importance sampling (IS) | 从提议分布抽样再加重要性权重估期望（第 10 章）。 |
| 重要性权重 | Importance weight | 目标密度与提议密度之比 $p/q$（第 10 章）。 |
| 自归一化重要性采样 | Self-normalized IS (SNIS) | 只知未归一化目标时用权重归一估期望（第 10 章）。 |
| 马尔可夫链 | Markov chain | 下一步只依赖当前态的随机过程（第 10 章）。 |
| 转移矩阵 / 转移核 | Transition matrix / kernel | 描述一步状态转移概率（离散/连续）（第 10 章）。 |
| 平稳分布 | Stationary distribution | 链演化的"不动点"分布，MCMC 的目标（第 10 章）。 |
| 细致平衡 | Detailed balance | $\pi(x)P(x\to x')=\pi(x')P(x'\to x)$，构造平稳分布的充分条件（第 10 章）。 |
| 可逆性 | Reversibility | 满足细致平衡的链的等价表述（第 10 章）。 |
| 不可约 | Irreducible | 任意态间可达，遍历性所需（第 10 章）。 |
| 非周期 | Aperiodic | 不被困在固定循环里，遍历性所需（第 10 章）。 |
| 正常返 | Positive recurrent | 期望返回时间有限，遍历性所需（第 10 章）。 |
| 遍历 | Ergodic | 不可约+非周期+正常返，保证收敛到唯一平稳分布（第 10 章）。 |
| 遍历定理 / 遍历假设 | Ergodic theorem / hypothesis | 时间平均 = 空间（系综）平均，把模拟变可算（第 10 章）。 |
| 马尔可夫链蒙特卡洛 | Markov chain Monte Carlo (MCMC) | 构造平稳分布为目标的链来采样的通用引擎（第 10 章）。 |
| Metropolis–Hastings | Metropolis–Hastings (MH) | 提议+接受/拒绝的通用 MCMC，接受率只依赖目标比值、绕开 $Z$（第 10 章）。 |
| 接受概率 | Acceptance probability | MH 中决定是否接受提议的概率，含提议比与目标比（第 10 章）。 |
| 提议分布 | Proposal distribution | 从当前态产生候选态的分布（第 10 章）。 |
| 随机游走 Metropolis | Random-walk Metropolis (RWM) | 对称高斯提议的 Metropolis 特例（第 10 章）。 |
| 混合 | Mixing | 链遍历空间、去相关的快慢（第 10 章）。 |
| Gibbs 采样 | Gibbs sampling | 逐坐标按全条件分布更新，接受率恒为 1（第 10 章）。 |
| 分块 Gibbs | Block Gibbs | 一次更新一组相关坐标以改善混合（第 10 章）。 |
| 扫描 | Sweep | 一轮遍历所有坐标的 Gibbs 更新（第 10 章）。 |
| 模拟退火 | Simulated annealing (SA) | 把 $1/T$ 当 $\beta$ 缓慢降温做全局优化（第 10 章）。 |
| 探索–利用权衡 | Exploration–exploitation tradeoff | 高温探索/低温利用，退火与采样的核心旋钮（第 1、10 章）。 |
| 副本交换 / 并行回火 | Replica exchange / parallel tempering | 多温度副本互换以翻越能垒、解多峰（第 10 章）。 |
| 自由能微扰 | Free energy perturbation (FEP) | 用样本估两配分函数之比算自由能差（Zwanzig）（第 10 章）。 |
| 热力学积分 | Thermodynamic integration | 沿耦合参数积分算自由能差（第 10 章）。 |
| Langevin 动力学 | Langevin dynamics | 梯度漂移+噪声的采样，只需得分函数（第 10 章）。 |
| MALA | Metropolis-adjusted Langevin algorithm | 带 MH 校正的 Langevin 采样（第 10 章）。 |
| 随机梯度 Langevin | Stochastic gradient Langevin dynamics (SGLD) | 用小批量梯度+噪声从后验采样（第 10 章）。 |
| 哈密顿蒙特卡洛 | Hamiltonian Monte Carlo (HMC) | 借辅助动量沿哈密顿轨迹高效穿越高维（第 10 章）。 |
| 蛙跳 / 辛积分器 | Leapfrog / symplectic integrator | 近似守恒能量的积分器，HMC 高接受率的关键（第 10 章）。 |
| NUTS | No-U-Turn Sampler | 自动选轨迹长度的 HMC，Stan/PyMC 的默认采样器（第 10 章）。 |
| 得分函数（采样） | Score function $\nabla\log p$ | 从 $\pi$ 采样所需的全部信息，扩散/Langevin 的核心（第 10 章）。 |
| 去噪得分匹配 | Denoising score matching | 学习加噪数据得分以训练扩散模型（第 10 章）。 |
| 老化期 | Burn-in | 丢掉链尚未收敛的开头样本（第 10 章）。 |
| 自相关时间 | Autocorrelation time | 链中样本去相关所需步数（第 10 章）。 |
| 有效样本数 | Effective sample size (ESS) | 相关链中"相当于多少独立样本"，混合差则 ESS≪N（第 10、11 章）。 |
| 轨迹图 | Trace plot | 画样本随步数变化以目检收敛（第 10 章）。 |
| Gelman–Rubin 诊断 | Gelman–Rubin $\hat R$ | 多链方差比诊断收敛（潜在尺度缩减因子）（第 10 章）。 |
| 伪收敛 | Pseudo-convergence | 链看似稳定却漏掉某些众数的危险（第 10 章）。 |
| 准蒙特卡洛 | Quasi-Monte Carlo | 用低差异序列降低积分误差（第 10 章）。 |

---

## 10. 蛋白质折叠 / 计算生物学应用

| 中文名 | English | 解释 |
|---|---|---|
| 多序列比对 | Multiple sequence alignment (MSA) | 同源序列对齐成的矩阵，是带系统偏差、非 i.i.d. 的样本（第 1、6、11 章）。 |
| 系统发育偏差 | Phylogenetic bias | 序列沿进化树相关导致的非独立，使有效样本远小于行数（第 1、6 章）。 |
| 序列重加权 | Sequence reweighting | 给近亲序列降权以修正系统发育冗余（第 6、7、11 章）。 |
| 有效样本数（MSA） | Effective sample size $M_{\text{eff}}$ | 重加权后 MSA 的"有效序列数"，常远小于名义条数（第 1、11 章）。 |
| 共进化 | Coevolution / co-evolution | 空间接触的残基协同变异，是接触预测的信号源（第 1、5、11 章）。 |
| 直接耦合分析 | Direct coupling analysis (DCA) | 用最大熵 Potts 模型从 MSA 剥离直接耦合、预测接触，调用全课工具（第 11 章）。 |
| Potts 模型 | Potts model | MSA 的成对最大熵/玻尔兹曼模型，参数为场 $h_i$ 与耦合 $J_{ij}$（第 11 章）。 |
| 最大熵原理 | Maximum entropy principle | 在给定约束下选最不偏分布，逼出 Potts 模型（第 11 章，呼应信息论课程）。 |
| 单点场 / 耦合 | Field $h_i$ / coupling $J_{ij}$ | Potts 模型的单点偏好与成对相互作用参数（第 11 章）。 |
| 平均场 DCA | Mean-field DCA (mfDCA) | 对耦合做平均场近似后解析求逆的早期 DCA（Morcos 2011）（第 11 章）。 |
| 伪似然最大化 DCA | plmDCA | 用伪似然绕开配分函数估 Potts 参数的主流 DCA（第 7、11 章）。 |
| 高斯 DCA | Gaussian DCA (GaussDCA) | 把 MSA 当多元高斯、用精度矩阵剥直接耦合（第 11 章）。 |
| 平均乘积校正 | Average product correction (APC) | 给耦合分数去除采样/保守偏置的校正（第 1、11 章）。 |
| 接触图 | Contact map | 残基对是否空间接触的矩阵，喂给结构预测（第 3、11 章）。 |
| 保守度 | Conservation | MSA 某列氨基酸的保守程度（第 4 章）。 |
| 氨基酸频率谱 | Amino-acid profile | MSA 某列的氨基酸频率分布（PSSM 的基础）（第 4、6 章）。 |
| 拉氏图 | Ramachandran plot | 主链二面角 $(\phi,\psi)$ 的分布图，方向统计的应用现场（第 1、4 章）。 |
| 二面角 | Dihedral / torsion angle $\phi,\psi$ | 主链扭转角，是落在环面上的二维随机变量（第 1、4 章）。 |
| 构象系综 | Conformational ensemble | 蛋白在温度下采样的整组构象，即分布 $p(x)$（第 1 章）。 |
| B 因子 | B-factor | 正比于原子均方位移的晶体学量，系综宽度的"带噪代理"（第 1 章）。 |
| 不确定性量化 | Uncertainty quantification (UQ) | 给预测配上可信度/区间，从"会预测"升级到"会给可信预测"（第 11 章）。 |
| pLDDT | predicted Local Distance Difference Test | AF2 对每残基局部精度的自评分（0–100），需校准检验（第 1、11 章）。 |
| PAE | Predicted Aligned Error | AF2 的 $n\times n$ 误差矩阵，判断结构域相对取向，非对称（第 1、11 章）。 |
| 校准 | Calibration | 声称的置信度兑现为实际频率（说 90% 就真约 90% 达标）（第 6、11 章）。 |
| 过度自信 | Overconfidence | 模型置信度系统性高于实际正确率（第 1、11 章）。 |
| 可靠性图 | Reliability diagram | 名义置信度 vs 实际达标率的校准曲线（第 1、11 章）。 |
| 深度集成 | Deep ensemble | 多个随机初始化网络近似贝叶斯后验、量化认知不确定性（第 11 章）。 |
| 共形预测 | Conformal prediction | 给出分布无关的有限样本覆盖保证的 UQ（第 11 章）。 |
| 生成模型 | Generative model | "学一个采样器"，蛋白设计 = 受约束地从学到分布采样（第 1、11 章）。 |
| 扩散模型 | Diffusion model | 学得分函数、反向去噪采样的生成模型（RFdiffusion）（第 10、11 章）。 |
| 变分自编码器 | Variational autoencoder (VAE) | 优化 ELBO 的生成模型，EM 的神经版（第 11 章）。 |
| 归一化流 | Normalizing flow | 可逆网络、能精确算似然的生成模型（第 2、10 章）。 |
| Boltzmann generator | Boltzmann generator | 用流直接采玻尔兹曼分布、绕过 MCMC 能垒（Noé 2019）（第 10、11 章）。 |
| 模式坍缩 | Mode collapse | 生成模型漏掉分布某些众数（第 10、11 章）。 |
| 回归 | Regression | "序列 → 标量性质"映射，变体效应预测的基础工具（第 11 章）。 |
| 广义线性模型 | Generalized linear model (GLM) | 指数族响应+连接函数的回归框架（第 11 章）。 |
| 连接函数 | Link function | 把线性预测子映到响应均值的函数（如 logit）（第 11 章）。 |
| 深度突变扫描 | Deep mutational scanning (DMS) | 大规模测变体效应的实验，是变体回归的数据源（第 11 章）。 |
| 变体效应预测 | Variant effect prediction | 预测突变对稳定性/功能的影响（如 $\Delta\Delta G$）（第 1、11 章）。 |
| 混杂因子 | Confounder | 同时影响自变量与结果的未观测因素，"相关不是因果"的根源（第 1、11 章）。 |
| 全基因组关联研究 | Genome-wide association study (GWAS) | 在百万位点找与表型相关者，多重检验的现场（第 1、11 章）。 |
| 富集分析 | Enrichment analysis | 找显著富集的通路/基序，受多重检验约束（第 1、11 章）。 |
| 差异表达分析 | Differential expression | 找表达显著变化的基因，q 值控制 FDR（第 11 章）。 |
| 群体结构 / p 值膨胀 | Population structure / genomic inflation | GWAS 中未校正相关导致 p 值系统偏小（第 11 章）。 |
| Anfinsen 热力学假说 | Anfinsen's thermodynamic hypothesis | 天然态 = 生理条件下吉布斯自由能最低的态（构象分布主峰）（第 1 章）。 |
| 折叠漏斗 | Folding funnel | 能量地形上引导折叠到低能态的漏斗形貌（第 1 章）。 |

---

## 11. 人物·文献·里程碑

| 中文名 | English | 解释 |
|---|---|---|
| Kolmogorov | Andrey Kolmogorov | 概率论公理化奠基人，三公理以他命名（第 2 章）。 |
| Bayes / 贝叶斯定理 | Thomas Bayes | 贝叶斯定理与贝叶斯推断的名祖（第 2、8 章）。 |
| Fisher | Ronald A. Fisher | 极大似然、充分统计量、Fisher 信息、方差分析的奠基者（第 7 章）。 |
| Cramér–Rao 下界 | Harald Cramér & C. R. Rao | 无偏估计方差下界的提出者（第 7 章）。 |
| Rao–Blackwell | C. R. Rao & David Blackwell | 用充分统计量改良估计量的定理（第 7 章）。 |
| Neyman–Pearson | Jerzy Neyman & Egon Pearson | 假设检验"最强检验"引理的提出者（第 9 章）。 |
| Pearson | Karl Pearson | 相关系数与矩估计法（1894）的先驱（第 3、7 章）。 |
| Wilks 定理 | Samuel S. Wilks | 似然比统计量渐近卡方的定理（第 9 章）。 |
| Wald | Abraham Wald | Wald 等式与基于 MLE 渐近正态的 Wald 区间（第 3、9 章）。 |
| Jensen | Johan Jensen | 凸函数期望不等式的名祖（第 3 章）。 |
| Markov | Andrey Markov | Markov 不等式与马尔可夫链的名祖（第 3、10 章）。 |
| Chebyshev | Pafnuty Chebyshev | 用方差界尾概率的不等式（第 3 章）。 |
| Hoeffding / Chernoff | Wassily Hoeffding / Herman Chernoff | 有界和与 MGF 型指数集中不等式的提出者（第 3 章）。 |
| Bernstein | Sergei Bernstein | 用方差更紧的指数集中不等式（第 6 章）。 |
| Lindeberg–Feller / Lyapunov | Lindeberg–Feller / Lyapunov | 非同分布 CLT 成立条件的提出者（第 6 章）。 |
| Berry–Esseen | Andrew Berry & Carl-Gustav Esseen | CLT 有限样本收敛速率定理（第 6 章）。 |
| Slutsky | Eugen Slutsky | 收敛序列和/积极限规则的名祖（第 6 章）。 |
| Glivenko–Cantelli | Glivenko & Cantelli | 经验 CDF 一致收敛"统计学基本定理"（第 6 章）。 |
| von Mises | Richard von Mises | 圆周"高斯"分布的名祖，二面角建模核心（第 4 章）。 |
| Gauss | Carl Friedrich Gauss | 正态/高斯分布的名祖（第 4 章）。 |
| Poisson | Siméon Denis Poisson | 稀有事件计数分布的名祖（第 4 章）。 |
| Dirichlet | Peter Gustav Lejeune Dirichlet | 单纯形上分布、多项共轭先验的名祖（第 4、8 章）。 |
| Student | William Gosset ("Student") | t 分布的提出者（第 4 章）。 |
| Schur 补 | Issai Schur | 高斯条件协方差/分块求逆的名祖（第 5 章）。 |
| EM 算法（DLR 1977） | Dempster, Laird & Rubin (1977) | 系统提出 EM 算法的奠基论文（第 7 章）。 |
| Jeffreys | Harold Jeffreys | 参数化不变"客观"先验的提出者（第 8 章）。 |
| James–Stein | Willard James & Charles Stein (1961) | 高维下收缩估计优于逐点估计的惊人结果（第 8 章）。 |
| Bernstein–von Mises | Bernstein–von Mises theorem | 大样本下贝叶斯后验近似为以 MLE 为中心的高斯、两派握手（第 8 章）。 |
| Occam | William of Occam | "如无必要勿增实体"，被边际似然自动实现（第 8 章）。 |
| Benjamini–Hochberg (1995) | Benjamini & Hochberg | 控制 FDR 的 BH 程序原论文（第 9 章）。 |
| Benjamini–Yekutieli | Benjamini & Yekutieli | 任意相依下控 FDR 的 BY 修正（第 11 章）。 |
| Storey / q 值 | John Storey | q 值与正 FDR 框架的提出者（第 9 章）。 |
| Metropolis 等（1953） | Metropolis et al. (1953) | 为从玻尔兹曼分布采样而发明 Metropolis 算法，MCMC 的"创世纪"（第 10 章）。 |
| Hastings (1970) | W. K. Hastings | 把 Metropolis 推广为非对称提议的 Metropolis–Hastings（第 10 章）。 |
| Gibbs | Josiah Willard Gibbs | 吉布斯分布与 Gibbs 采样的名祖（第 10 章）。 |
| Hamilton / HMC | Hamiltonian mechanics → HMC | 借哈密顿力学的高效 MCMC，与分子动力学同源（第 10 章）。 |
| Langevin | Paul Langevin | 梯度漂移+噪声采样动力学的名祖（第 10 章）。 |
| Gelman–Rubin | Andrew Gelman & Donald Rubin | 多链 $\hat R$ 收敛诊断的提出者（第 10 章）。 |
| Boltzmann generator（Noé 2019） | Noé et al. (2019) | 用流直接采玻尔兹曼分布的代表工作（第 10、11 章）。 |
| Morcos 等（2011，DCA） | Morcos et al., PNAS 2011 | 直接耦合分析捕获天然接触的里程碑论文（第 11 章）。 |
| AlphaFold2（Jumper 2021） | Jumper et al., Nature 2021 | 端到端结构预测 + pLDDT/PAE 不确定性量化的里程碑（第 1、11 章）。 |
| ESM（蛋白语言模型） | Rives et al., PNAS 2021 | 用交叉熵（负对数似然）自监督训练的蛋白语言模型（第 1、7 章）。 |
| Wasserman《All of Statistics》 | L. Wasserman | 本课首要主参考，专为数学/CS 背景者写的统计速成（第 1、11 章）。 |
| Casella–Berger《Statistical Inference》 | Casella & Berger | 严谨的研究生统计推断经典，第 7、9 章深挖依靠（第 11 章）。 |
| Gelman 等《Bayesian Data Analysis》 | Gelman et al. | 贝叶斯建模"圣经"，呼应第 8、10 章（第 11 章）。 |
| Robert–Casella《Monte Carlo Statistical Methods》 | Robert & Casella | MCMC 权威专著，呼应第 10 章（第 11 章）。 |
| ASA p 值声明（2016） | Wasserstein & Lazar (2016) | 美国统计学会逐条澄清"p 值不是什么"的权威文献（第 1、9 章）。 |

---

> **一句话收束**：这张表是全课术语的索引而非替代——遇到不熟的词先在此定位中英名与所在章，再回正文精读。把"中文直觉 + 英文术语 + 所在章"三者绑定，你读折叠/计算生物学论文方法部分的速度与深度都会上一个台阶。
