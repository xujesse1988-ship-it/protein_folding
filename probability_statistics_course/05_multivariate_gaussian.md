# 第 5 章 多元分布、协方差与多元高斯

> **本章定位**：前四章的随机量大多是「一个数」——一个角度、一次成功、一个能量。但折叠研究里几乎没有真正的一维对象：一条链有成百上千个原子坐标、MSA 有几百列联动的氨基酸、一帧 MD 快照是几万维的向量。这些维度**互相牵制**——一个残基动了，邻居跟着动；一列突变了，配对列也得跟着补偿。本章就来处理「向量随机变量」，并把第 2–4 章的工具升维：联合/边缘/条件分布的关系（§5.1）、把方差升级成**协方差矩阵 $\Sigma$**（§5.2，对称半正定，它的正定性有干净的几何意义），然后是全统计学最重要的「工作母机」——**多元高斯 $\mathcal N(\boldsymbol\mu,\Sigma)$**（§5.3）：它的等高线是椭球、主轴是 $\Sigma$ 的特征向量、仿射变换/边缘/**条件全都还是高斯**，而条件均值恰好就是「线性回归」、条件协方差恰好是「Schur 补」。本章的智力高潮在 §5.4：协方差的逆——**精度矩阵（precision matrix）$\Lambda=\Sigma^{-1}$**——它的零元素**精确地**编码「条件独立」，于是给出**高斯图模型（Gaussian graphical model）**。这一条「零元素 = 条件独立」直接命中折叠里的**接触预测**：协方差（表观耦合）会被传递相关污染，精度矩阵（直接耦合）才指向物理接触——这正是 DCA/Potts 思想（信息论课程 ch10/ch11、折叠课程第 5 章）在高斯情形下的**线性版本**，是理解「直接 vs 间接」的最干净入口。最后两节落到工程与诚实：白化与 PCA 作为 $\Sigma$ 的特征分解（§5.5，连接 MD 主成分/essential dynamics 与弹性网络 GNM/ANM），以及高斯假设的边界（§5.6，重尾、多峰、流形——真实构象分布远不止一个高斯，为第 10 章采样与生成模型埋雷）。本章有交互页 `05_multivariate_gaussian.html`（拖协方差看椭圆、看条件切片、看精度矩阵），强烈建议边读边玩。读完本章，你将拥有处理「高维强相关变量」的全套线性武器，并第一次看清「接触图为什么要用精度矩阵而非相关矩阵」。

---

## 5.1 随机向量与联合 / 边缘 / 条件分布

### 5.1.1 从「一个数」到「一个向量」

一个 **随机向量（random vector）** $\mathbf X=(X_1,\dots,X_d)^\top$ 就是把 $d$ 个随机变量打包成一列。它仍然住在某个概率空间 $(\Omega,\mathcal F,\mathbb P)$ 上（第 2 章），只是取值落在 $\mathbb R^d$ 而非 $\mathbb R$。它的分布由**联合（joint）** 描述：

- 离散：联合 pmf $p(\mathbf x)=\mathbb P(X_1=x_1,\dots,X_d=x_d)$。
- 连续：联合 pdf $f(\mathbf x)$，满足 $\mathbb P(\mathbf X\in A)=\int_A f(\mathbf x)\,\mathrm d\mathbf x$、$\int_{\mathbb R^d}f=1$、$f\ge0$。
- 统一：联合 CDF $F(\mathbf x)=\mathbb P(X_1\le x_1,\dots,X_d\le x_d)$。

直觉上，联合分布回答的是「这 $d$ 个量**同时**取某组值的可能性」。它比「各自的边缘」携带**更多信息**——这正是本章的全部要害：变量之间的**耦合**只活在联合里，边缘看不到。

### 5.1.2 边缘分布：把不关心的维度积掉

**边缘分布（marginal distribution）** 是「只看其中几个分量、对其余分量取平均」的结果。把 $\mathbf X$ 拆成两块 $\mathbf X=(\mathbf X_A,\mathbf X_B)$（$A,B$ 是下标的两组），

$$f_{\mathbf X_A}(\mathbf x_A)=\int f(\mathbf x_A,\mathbf x_B)\,\mathrm d\mathbf x_B\qquad(\text{连续；离散把积分换求和}).$$

「边缘」这个名字来自二维列联表：把每行（每列）求和写在表格的**边上**（margin），就得到一个变量的边缘分布。求边缘是一个**信息损失**的操作：你扔掉了「另一组变量如何与它联动」的全部信息。

> **CS 读者陷阱（边缘相同 ≠ 联合相同）**：两个截然不同的联合分布可以有**完全一样的所有边缘**。最干净的例子：$(X,Y)$ 取 $(0,0),(1,1)$ 各 $1/2$（完全正相关）；与 $(X,Y)$ 取 $(0,1),(1,0)$ 各 $1/2$（完全负相关）——两者的边缘都是「$X,Y$ 各自 $\mathrm{Bernoulli}(1/2)$」，但联合天差地别。**所以你永远不能从边缘重建联合**（除非额外假设独立）。这件事在折叠里是核心痛点：MSA 给你每列的边缘频率（保守性）很容易，但「两列如何协同」的联合信息才是共进化信号，必须专门去估（§5.4、信息论 ch11）。把联合从边缘 +「耦合参数」重建出来，正是 §5.3–5.4 的多元高斯/Potts 在做的事。

### 5.1.3 条件分布：固定一部分、问另一部分

**条件分布（conditional distribution）** 是「已知 $\mathbf X_B=\mathbf x_B$，$\mathbf X_A$ 的分布」：

$$f_{\mathbf X_A\mid\mathbf X_B}(\mathbf x_A\mid\mathbf x_B)=\frac{f(\mathbf x_A,\mathbf x_B)}{f_{\mathbf X_B}(\mathbf x_B)}\qquad(f_{\mathbf X_B}(\mathbf x_B)>0).$$

几何上，这是把联合密度沿「$\mathbf X_B=\mathbf x_B$」这张**切片**取出来，再重新归一化成一个概率分布。条件分布是「在已知部分信息后更新对其余的认识」——整个统计推断（第 7、8 章）都是它的展开。本章会反复用到一个奇迹：**多元高斯的条件分布仍是高斯**，且条件均值是线性的、条件协方差是常数（§5.3.4），这让「高维条件推理」变成解线性方程组。

### 5.1.4 独立、条件独立与链式法则

- **独立（independence）**：$\mathbf X_A\perp\mathbf X_B$ 当且仅当 $f(\mathbf x_A,\mathbf x_B)=f_{\mathbf X_A}(\mathbf x_A)f_{\mathbf X_B}(\mathbf x_B)$，即「联合 = 边缘乘积」。
- **条件独立（conditional independence）**：$X_i\perp X_j\mid \mathbf X_R$（在给定第三组 $\mathbf X_R$ 后 $i,j$ 独立）当且仅当 $f(x_i,x_j\mid\mathbf x_R)=f(x_i\mid\mathbf x_R)f(x_j\mid\mathbf x_R)$。**这是本章 §5.4 的主角**——它区分「$i,j$ 直接相关」与「$i,j$ 只是通过 $R$ 间接相关」。
- **链式法则（chain rule）**：任何联合都能逐维分解 $f(\mathbf x)=\prod_{k=1}^d f(x_k\mid x_1,\dots,x_{k-1})$。这是把高维分布拆成一串条件的通用工具（自回归模型、语言模型、扩散模型采样的骨架）。

> **要点**：随机向量的分布是**联合**；从联合积掉一些维度得**边缘**（信息损失，不可逆）、固定一些维度得**条件**（信息更新）。变量之间的耦合只活在联合里，边缘看不到——这是为什么 MSA 的「列频率」不够、必须估「成对/联合」才能拿到共进化信号。条件独立（§5.4）是「直接 vs 间接相关」的精确语言。（§5.1）

---

## 5.2 协方差矩阵 $\Sigma$：刻画整组变量的耦合

### 5.2.1 定义与基本性质

把第 3 章的方差与协方差升维。随机向量 $\mathbf X$ 的**均值向量（mean vector）** 是逐分量期望

$$\boldsymbol\mu=\mathbb E[\mathbf X]=(\mathbb E[X_1],\dots,\mathbb E[X_d])^\top,$$

它的**协方差矩阵（covariance matrix）** 是

$$\boxed{\ \Sigma=\mathrm{Cov}(\mathbf X)=\mathbb E\big[(\mathbf X-\boldsymbol\mu)(\mathbf X-\boldsymbol\mu)^\top\big],\qquad \Sigma_{ij}=\mathrm{Cov}(X_i,X_j).\ }$$

对角元 $\Sigma_{ii}=\mathrm{Var}(X_i)$ 是各分量的方差，非对角元 $\Sigma_{ij}=\mathbb E[(X_i-\mu_i)(X_j-\mu_j)]$ 是分量两两之间的协方差。一个等价、更好记的写法是 $\Sigma=\mathbb E[\mathbf X\mathbf X^\top]-\boldsymbol\mu\boldsymbol\mu^\top$。

它有三条立刻能用的性质：

1. **对称（symmetric）**：$\Sigma=\Sigma^\top$，因为 $\mathrm{Cov}(X_i,X_j)=\mathrm{Cov}(X_j,X_i)$。
2. **半正定（positive semi-definite, PSD）**：对任意常向量 $\mathbf a\in\mathbb R^d$，$\mathbf a^\top\Sigma\mathbf a\ge0$。
3. **线性变换法则**：对任意矩阵 $A$、向量 $\mathbf b$，若 $\mathbf Y=A\mathbf X+\mathbf b$，则
$$\mathbb E[\mathbf Y]=A\boldsymbol\mu+\mathbf b,\qquad \mathrm{Cov}(\mathbf Y)=A\,\Sigma\,A^\top.$$

第 3 条是高维统计的**主力公式**，本章后面（仿射变换仍是高斯、白化、PCA）全靠它，务必背熟。第 2 条的证明只有一行，但揭示了协方差矩阵的本质，单独拎出来讲。

### 5.2.2 半正定性：一行证明与它的几何意义

> **定理（协方差矩阵半正定）。** 任何随机向量（各分量方差有限）的协方差矩阵 $\Sigma$ 都对称半正定。

**一行证明**：对任意 $\mathbf a$，令标量 $S=\mathbf a^\top(\mathbf X-\boldsymbol\mu)$。它是 $\mathbf X$ 各分量的一个线性组合，故
$$\mathbf a^\top\Sigma\mathbf a=\mathbf a^\top\mathbb E[(\mathbf X-\boldsymbol\mu)(\mathbf X-\boldsymbol\mu)^\top]\mathbf a=\mathbb E[(\mathbf a^\top(\mathbf X-\boldsymbol\mu))^2]=\mathrm{Var}(S)\ge0.$$
方差非负，证毕。换言之：**「在方向 $\mathbf a$ 上投影后的方差」就是二次型 $\mathbf a^\top\Sigma\mathbf a$**——协方差矩阵是「所有方向上的散布」打包成的一个对象。

这给了 $\Sigma$ 一个干净的几何图像。因为 $\Sigma$ 对称，谱定理保证它可正交对角化：

$$\Sigma=Q\Lambda_{\rm eig}Q^\top,\qquad Q=[\mathbf q_1,\dots,\mathbf q_d]\ (\text{正交}),\quad \Lambda_{\rm eig}=\mathrm{diag}(\lambda_1,\dots,\lambda_d),\ \lambda_i\ge0.$$

特征值 $\lambda_i$ 是「沿特征方向 $\mathbf q_i$ 的方差」（因为 $\mathbf q_i^\top\Sigma\mathbf q_i=\lambda_i$）。半正定 $\Leftrightarrow$ 所有 $\lambda_i\ge0$ $\Leftrightarrow$ 没有任何方向上的方差是负的（方差当然不能为负，所以这个约束是天然的）。这套特征分解就是 §5.5 PCA 的全部数学。

> **诚实的边界（半正定 vs 正定，何时退化）**：$\Sigma$ 总是半正定，但**不一定正定**。若存在方向 $\mathbf a\ne0$ 使 $\mathbf a^\top\Sigma\mathbf a=0$，意味着 $\mathrm{Var}(\mathbf a^\top\mathbf X)=0$，即 $\mathbf a^\top\mathbf X$ 几乎处处是常数——数据落在一个**低维仿射子空间**里（有一条线性约束）。此时 $\Sigma$ **奇异、不可逆**，精度矩阵 $\Sigma^{-1}$（§5.4）不存在，多元高斯密度公式（§5.3）也失效。这在折叠里很常见：MD 轨迹有 $3N$ 个坐标，但整体平动（3 个）和转动（3 个）是冗余自由度，对齐（superpose）后这 6 个方向方差为 0，协方差矩阵秩亏 6；做 PCA/GNM 前必须先去掉这些零模（zero modes），否则求逆会炸。**「样本数 $<$ 维数」时样本协方差几乎必然奇异**，这是高维统计（含 DCA）反复要面对的病，靠正则化（加 $\epsilon I$、收缩估计）来救（§5.4.4）。

### 5.2.3 相关矩阵：把协方差标准化

协方差的大小依赖于各变量的单位（把长度从埃换成纳米，协方差就缩 100 倍），不便横向比较。把它**标准化**得到**相关矩阵（correlation matrix）**：

$$\rho_{ij}=\frac{\Sigma_{ij}}{\sqrt{\Sigma_{ii}\Sigma_{jj}}}=\frac{\mathrm{Cov}(X_i,X_j)}{\sigma_i\sigma_j}\in[-1,1].$$

矩阵形式 $R=D^{-1/2}\Sigma D^{-1/2}$，其中 $D=\mathrm{diag}(\Sigma_{11},\dots,\Sigma_{dd})$。相关矩阵对角线全是 1，非对角是 **Pearson 相关系数（Pearson correlation coefficient）**（前提 $\sigma_i,\sigma_j>0$，否则 $\rho$ 无定义）。$|\rho_{ij}|=1$ 当且仅当 $X_i,X_j$ 完全线性相关，更精确地：**以概率 1 有 $X_i=aX_j+b$（仿射关系，$a\ne0$，含常数项 $b$）**。

注意：$R$ 也是半正定的（它是 $D^{-1/2}\mathbf X$ 的协方差），所以「随便填一张对称、对角为 1、元素在 $[-1,1]$ 的矩阵」**未必**是合法相关矩阵——它还得半正定。这是初学者构造「假设的相关结构」时最常踩的坑。

### 5.2.4 协方差 vs 相关 vs 独立：诚实的三层关系

这是全章最容易混淆、也最高频被误用的地方，必须钉死。三个概念由强到弱：

$$\text{独立}\ \Longrightarrow\ \text{不相关}\ (\rho=0),\qquad \text{但反过来一般不成立}.$$

- **独立**：$f(x_i,x_j)=f_i(x_i)f_j(x_j)$。这是最强的——它说**任何**函数都不耦合。
- **不相关（uncorrelated）**：$\mathrm{Cov}(X_i,X_j)=0$。这只说**线性**部分不耦合，非线性的耦合它看不见。

> **诚实的边界（不相关 ≠ 独立）**：协方差/相关只度量**线性**关联。一个反例足以钉死：设 $X\sim\mathcal N(0,1)$，$Y=X^2$。则 $\mathrm{Cov}(X,Y)=\mathbb E[X^3]-\mathbb E[X]\mathbb E[X^2]=0-0=0$（奇矩为零），**完全不相关**；但 $Y$ 由 $X$ 完全决定，**强烈依赖**、远非独立。所以「相关系数为 0」绝不能读成「无关系」——它只意味着「无**线性**关系」。这在折叠里是真实陷阱：二面角与能量、距离与接触常是**非线性**关系，用 Pearson 相关筛选会漏掉强非线性耦合（这正是信息论课程为什么用**互信息**而非相关来量化共进化——互信息抓任意依赖，相关只抓线性，见信息论 ch4、ch11）。

那么「不相关 ⇒ 独立」什么时候成立？只有一个干净的特例，而且极其重要：

> **定理（高斯下不相关即独立）。** 若 $(\mathbf X_A,\mathbf X_B)$ **联合**服从多元高斯，则 $\mathbf X_A\perp\mathbf X_B$ $\iff$ $\mathrm{Cov}(\mathbf X_A,\mathbf X_B)=0$（即跨块协方差为零矩阵）。

证明思路在 §5.3.3 给出（高斯的联合密度在 $\Sigma$ 为块对角时恰好因子化）。这条定理是高斯之所以「好用」的核心特权之一：**对高斯，二阶矩（协方差）就完全决定了独立性**。

> **CS 读者陷阱（「联合高斯」是硬条件）**：上面定理要求**联合**高斯，而不只是「各自边缘高斯」。两个边缘都是高斯、但联合不是高斯的情形，不相关并不蕴含独立——存在 $X,Y$ 各自标准正态、不相关、却不独立的反例（例如 $Y=X$ 当 $|X|>c$、$Y=-X$ 当 $|X|\le c$，调 $c$ 使 $\mathrm{Cov}=0$）。**为什么 $Y$ 的边缘仍是 $N(0,1)$**：标准正态关于原点对称（$X$ 与 $-X$ 同分布），所以无论怎样按 $|X|$ 分段地在 $X$ 与 $-X$ 之间切换，$Y$ 的边缘都不变——于是得到两个边缘均高斯、不相关、却不独立、且联合非高斯的合法反例。所以引用这条定理时务必确认是「联合高斯」。这也提醒：判断两个变量是否**联合**高斯，比判断单变量是否高斯难得多。

> **要点**：协方差矩阵 $\Sigma$ 对称、半正定（任意方向的投影方差 $\mathbf a^\top\Sigma\mathbf a\ge0$），特征向量给主轴、特征值给主轴方差。标准化得相关矩阵（对角为 1、元素 $\in[-1,1]$、仍半正定）。三层关系：独立 ⇒ 不相关，反之**一般不成立**（相关只抓线性，$X,X^2$ 是经典反例）；唯有**联合高斯**时不相关 ⇔ 独立。$\Sigma$ 奇异 = 数据落在低维子空间（MD 的平动/转动零模、样本数小于维数），求逆前须正则化。（§5.2）

---

## 5.3 多元高斯 $\mathcal N(\boldsymbol\mu,\Sigma)$：高维世界的工作母机

多元高斯（multivariate Gaussian / multivariate normal）是整个统计学使用频率最高的多维分布。它之所以是「工作母机」，是因为它对几乎所有操作都**封闭**：仿射变换、边缘化、条件化、求和，结果**永远还是高斯**，而且只需追踪 $(\boldsymbol\mu,\Sigma)$ 这两个对象。本节把它的全套性质讲透。

### 5.3.1 密度公式与「马氏距离」

$\mathbf X\sim\mathcal N(\boldsymbol\mu,\Sigma)$（$\Sigma$ 正定）的 pdf 是

$$\boxed{\ f(\mathbf x)=\frac{1}{(2\pi)^{d/2}\,|\Sigma|^{1/2}}\exp\!\Big(-\tfrac12(\mathbf x-\boldsymbol\mu)^\top\Sigma^{-1}(\mathbf x-\boldsymbol\mu)\Big).\ }$$

把它和一维 $\frac1{\sqrt{2\pi}\sigma}e^{-(x-\mu)^2/2\sigma^2}$ 对照，结构完全平行：$\sigma^2\to\Sigma$、$(x-\mu)^2/\sigma^2\to(\mathbf x-\boldsymbol\mu)^\top\Sigma^{-1}(\mathbf x-\boldsymbol\mu)$、归一化常数里的 $\sigma\to|\Sigma|^{1/2}$。

指数里那个二次型有专门名字：

$$\Delta^2(\mathbf x)=(\mathbf x-\boldsymbol\mu)^\top\Sigma^{-1}(\mathbf x-\boldsymbol\mu)$$

叫**马氏距离（Mahalanobis distance）的平方**。它是「考虑了协方差结构的距离」——在方差大的方向上，同样的欧氏距离对应更小的马氏距离（因为「这点偏差在那个方向上很正常」）。等密度面 $\{\mathbf x:\Delta^2(\mathbf x)=c\}$ 是一族**椭球**，下一小节专讲。

> **给数学/CS 读者的视角**：把高斯的对数密度写出来，$\log f=-\tfrac12\mathbf x^\top\Lambda\mathbf x+\boldsymbol\eta^\top\mathbf x+\text{const}$，其中 $\Lambda=\Sigma^{-1}$、$\boldsymbol\eta=\Sigma^{-1}\boldsymbol\mu$。这是一个**关于 $\mathbf x$ 的二次型**——高斯是「对数密度是凹二次函数」的分布。这恰好把它放进**指数族**（本课第 4 章 §4.5）：充分统计量是 $T(\mathbf x)=(\mathbf x,\,\mathbf x\mathbf x^\top)$（一阶 + 二阶矩），自然参数是 $(\boldsymbol\eta,\Lambda)$。两个供严谨读者对账的细节：与二阶统计量 $\mathbf x\mathbf x^\top$ 配对的自然参数其实是 $-\tfrac12\Lambda$（那个 $-\tfrac12$ 来自二次型系数）；又因 $\mathbf x\mathbf x^\top$ 对称，二阶部分的独立分量只有 $d(d+1)/2$ 个（写成内积 $\langle\boldsymbol\eta,\mathbf x\rangle-\tfrac12\langle\Lambda,\mathbf x\mathbf x^\top\rangle$ 时，非对角项的对称冗余须按惯例处理，或改用 $\mathrm{vech}(\mathbf x\mathbf x^\top)$ 去冗余）——这都不影响「自然参数 = 精度矩阵」的结论，但在第 7 章（Fisher 信息/充分统计量）、第 8 章（高斯共轭 Normal–Wishart）会要对上号。这个「自然参数 = 精度矩阵」的视角在 §5.4 会大放异彩，也连回信息论课程 ch8（高斯是固定均值方差下的最大熵分布）。$(\boldsymbol\eta,\Lambda)$ 这套表示叫**信息形式 / 规范形式（information / canonical form）**，在高斯图模型和卡尔曼滤波里比 $(\boldsymbol\mu,\Sigma)$ 形式更方便。

### 5.3.2 几何：等高线是椭球，主轴 = $\Sigma$ 的特征向量

把 §5.2.2 的谱分解 $\Sigma=Q\Lambda_{\rm eig}Q^\top$ 代进马氏距离。等密度椭球 $\Delta^2(\mathbf x)=c$ 的形状由 $\Sigma$ 完全决定：

- **中心** 在 $\boldsymbol\mu$。
- **主轴方向** 是 $\Sigma$ 的特征向量 $\mathbf q_1,\dots,\mathbf q_d$（它们正交）。
- **主轴半长** 正比于 $\sqrt{\lambda_i}$（即沿该方向的标准差）：第 $i$ 根轴的半长是 $\sqrt{c\,\lambda_i}$。

```
   x2
    │        ┌────────────┐        等密度椭圆 Δ²=c
    │      ╱   q1 (长轴)    ╲      长轴方向 = 最大特征值 λ1 的特征向量
    │    ╱  ●μ ───────────→  ╲    长轴半长 ∝ √λ1
    │    ╲   ↑                ╱   短轴方向 = q2，半长 ∝ √λ2
    │      ╲  q2 (短轴)      ╱
    │        └────────────┘
    └────────────────────────── x1
```

两个直觉锚点：
- $\Sigma=\sigma^2 I$（各向同性）：椭球退化成**球**，各方向方差相同，变量两两不相关、各自方差 $\sigma^2$。
- $\Sigma$ 非对角元越大（相关越强）：椭球越**扁、越倾斜**——长轴指向「正相关方向」（$x_1$ 大时 $x_2$ 也大）。

这就是交互页 `05_multivariate_gaussian.html` 的核心演示：**拖动 $\Sigma$ 的非对角元 $\Sigma_{12}$，实时看椭圆如何从正圆变扁、变斜**。把 $\rho$ 从 0 推到接近 ±1，椭圆会塌缩成一条线（数据趋于完全线性相关，$\Sigma$ 趋于奇异）。强烈建议现在就打开它玩一遍——这比任何公式都快地让你「看见」协方差。

### 5.3.3 三大封闭性质：仿射、求和、边缘

> **定理（仿射变换仍是高斯）。** 若 $\mathbf X\sim\mathcal N(\boldsymbol\mu,\Sigma)$，$A$ 为矩阵、$\mathbf b$ 为向量，则
> $$A\mathbf X+\mathbf b\sim\mathcal N(A\boldsymbol\mu+\mathbf b,\ A\Sigma A^\top).$$

证明可用特征函数（高斯的特征函数 $\varphi_{\mathbf X}(\mathbf t)=\exp(i\mathbf t^\top\boldsymbol\mu-\tfrac12\mathbf t^\top\Sigma\mathbf t)$，本课第 3 章 §3.3.3 特征函数 $\varphi_X(t)=\mathbb E[e^{itX}]$ 的升维版；代入线性变换直接出结果）。均值/协方差的变化就是 §5.2.1 的线性变换法则，新鲜的是「**结果仍是高斯**」这件事。这条性质是白化（§5.5.1）、采样（用 Cholesky 把标准高斯线性变换成任意高斯）的根。

**两个推论**：

1. **任意线性组合是一维高斯**：取 $A=\mathbf a^\top$（一行），则 $\mathbf a^\top\mathbf X\sim\mathcal N(\mathbf a^\top\boldsymbol\mu,\ \mathbf a^\top\Sigma\mathbf a)$。事实上「$\mathbf X$ 的每个线性组合都是一维高斯」**等价于** $\mathbf X$ 是多元高斯——这常被当作多元高斯的**定义**（它比「联合密度是那个公式」更普适，能容纳 $\Sigma$ 奇异的退化高斯）。

2. **独立高斯之和仍是高斯**：$\mathbf X\sim\mathcal N(\boldsymbol\mu_1,\Sigma_1)$、$\mathbf Y\sim\mathcal N(\boldsymbol\mu_2,\Sigma_2)$ 独立，则 $\mathbf X+\mathbf Y\sim\mathcal N(\boldsymbol\mu_1+\boldsymbol\mu_2,\Sigma_1+\Sigma_2)$。

**边缘也是高斯**：把 $\mathbf X$ 分块 $\mathbf X=\begin{pmatrix}\mathbf X_A\\\mathbf X_B\end{pmatrix}$，相应分块

$$\boldsymbol\mu=\begin{pmatrix}\boldsymbol\mu_A\\\boldsymbol\mu_B\end{pmatrix},\qquad \Sigma=\begin{pmatrix}\Sigma_{AA}&\Sigma_{AB}\\\Sigma_{BA}&\Sigma_{BB}\end{pmatrix}.$$

则**边缘**就是「拣出对应块」：

$$\mathbf X_A\sim\mathcal N(\boldsymbol\mu_A,\ \Sigma_{AA}).$$

证明只需对「拣出 $A$ 分量」的投影矩阵用仿射定理。**边缘化对高斯简单到不可思议**——直接读子矩阵，不用做任何积分。

现在可以补上 §5.2.4 那条定理的证明骨架：若跨块协方差 $\Sigma_{AB}=0$，则 $\Sigma$ 块对角，$\Sigma^{-1}$ 也块对角，联合密度的指数项 $-\tfrac12(\mathbf x-\boldsymbol\mu)^\top\Sigma^{-1}(\mathbf x-\boldsymbol\mu)$ **拆成 $A$ 块与 $B$ 块之和**，$|\Sigma|=|\Sigma_{AA}||\Sigma_{BB}|$，于是 $f(\mathbf x_A,\mathbf x_B)=f_A(\mathbf x_A)f_B(\mathbf x_B)$（几乎处处成立）——**联合因子化即独立**。所以对高斯，**协方差为零 ⇔ 独立**，证毕。（此处设 $\Sigma_{AA},\Sigma_{BB}$ 正定使各块密度存在；若 $\Sigma$ 退化无密度，可改用推论 1 的「任意线性组合是一维高斯」定义绕过，结论不变。）

### 5.3.4 条件分布也是高斯：条件均值是线性回归，条件协方差是 Schur 补

这是多元高斯**最深、最有用**的性质，也是 §5.4 接触预测的引擎。

> **定理（高斯的条件分布）。** $\begin{pmatrix}\mathbf X_A\\\mathbf X_B\end{pmatrix}\sim\mathcal N\Big(\begin{pmatrix}\boldsymbol\mu_A\\\boldsymbol\mu_B\end{pmatrix},\begin{pmatrix}\Sigma_{AA}&\Sigma_{AB}\\\Sigma_{BA}&\Sigma_{BB}\end{pmatrix}\Big)$。则给定 $\mathbf X_B=\mathbf x_B$，
> $$\mathbf X_A\mid\mathbf X_B=\mathbf x_B\ \sim\ \mathcal N\big(\boldsymbol\mu_{A\mid B},\ \Sigma_{A\mid B}\big),$$
> $$\boldsymbol\mu_{A\mid B}=\boldsymbol\mu_A+\Sigma_{AB}\Sigma_{BB}^{-1}(\mathbf x_B-\boldsymbol\mu_B),\qquad \Sigma_{A\mid B}=\Sigma_{AA}-\Sigma_{AB}\Sigma_{BB}^{-1}\Sigma_{BA}.$$

这两个公式值得逐项品味，它们是统计学里被引用最多的公式之一：

**(1) 条件均值是「线性回归」。** $\boldsymbol\mu_{A\mid B}$ 是 $\mathbf x_B$ 的**仿射函数**，斜率矩阵 $\Sigma_{AB}\Sigma_{BB}^{-1}$ 正是把 $\mathbf X_A$ 对 $\mathbf X_B$ 做**最小二乘回归（least-squares regression）** 的系数。换句话说：**对高斯，最优预测器（条件期望 $\mathbb E[\mathbf X_A\mid\mathbf X_B]$，本课第 3 章证过它是均方最优）恰好是线性的**。这就是为什么线性回归在「联合高斯」假设下是「正确」而非「凑合」的模型——回归不是近似，是高斯条件均值的精确形式。

**(2) 条件协方差是「Schur 补」，且不依赖 $\mathbf x_B$。** $\Sigma_{A\mid B}=\Sigma_{AA}-\Sigma_{AB}\Sigma_{BB}^{-1}\Sigma_{BA}$ 是块矩阵 $\Sigma$ 关于 $\Sigma_{BB}$ 的 **Schur 补（Schur complement）**。两个深刻之处：
   - 它**比 $\Sigma_{AA}$ 小**（半正定意义下 $\Sigma_{A\mid B}\preceq\Sigma_{AA}$）——观测了 $\mathbf X_B$ 后，$\mathbf X_A$ 的不确定性只会减少或不变，**永不增加**（信息单调）。减少多少取决于 $\mathbf X_A,\mathbf X_B$ 耦合多强（$\Sigma_{AB}$ 多大）。（须强调：这条「永不增加」是高斯独有的**恒定**条件协方差的性质；对一般分布，只有**平均意义**的 $\mathbb E[\mathrm{Var}(X_A\mid X_B)]\le\mathrm{Var}(X_A)$ 必然成立——即第 3 章方差分解里的那一项，而某个**具体**观测值下的条件方差可以临时变大。）
   - 它**不依赖具体观测值 $\mathbf x_B$**——只要是高斯，无论你观测到 $\mathbf x_B$ 是多少，条件后的散布都一样。这是高斯独有的奢侈（一般分布的条件方差会随观测值变）。

> **与你的研究的连接（高斯过程、卡尔曼滤波、缺失坐标补全）**：这对公式是一大批折叠/结构工具的数学心脏。**高斯过程（Gaussian process, GP）** 用它做回归与不确定性量化（给定若干观测点，预测新点的均值 + 方差就是上面两式），在「能量面/打分函数的代理建模」「贝叶斯优化挑实验条件」里常用。**卡尔曼滤波（Kalman filter）** 是它在时间序列上的递推应用（粗粒化 MD、单分子追踪轨迹的状态估计）。在结构里：若把一组原子坐标建模成联合高斯（如弹性网络，§5.5.3），「已知一部分残基位置、推断其余残基最可能位置及其涨落」就是直接套这两式——条件均值给最可能构象、Schur 补给残余涨落。理解这对公式，等于拿到了「在高维高斯里做条件推理」的万能钥匙。

> **要点**：多元高斯对仿射变换、求和、边缘化、条件化**全封闭**，只需追踪 $(\boldsymbol\mu,\Sigma)$。密度由马氏距离 $(\mathbf x-\boldsymbol\mu)^\top\Sigma^{-1}(\mathbf x-\boldsymbol\mu)$ 决定，等高线是椭球（主轴 = $\Sigma$ 特征向量、半长 $\propto\sqrt{\lambda_i}$）。边缘 = 拣子矩阵（零积分）。**条件还是高斯**：条件均值 = 线性回归 $\boldsymbol\mu_A+\Sigma_{AB}\Sigma_{BB}^{-1}(\mathbf x_B-\boldsymbol\mu_B)$，条件协方差 = Schur 补 $\Sigma_{AA}-\Sigma_{AB}\Sigma_{BB}^{-1}\Sigma_{BA}$（比边缘小、不依赖观测值）。这是 GP、卡尔曼、接触预测的共同引擎。打开 `05_multivariate_gaussian.html` 看「条件切片」如何把 2D 椭圆压成 1D 高斯。（§5.3）

---

## 5.4 精度矩阵与高斯图模型：直接耦合 vs 间接耦合

现在登场的是本章的智力高峰——也是与折叠研究咬合最紧的一节。我们要论证：**协方差矩阵 $\Sigma$ 看到的是「总相关」（含一切间接路径），而它的逆——精度矩阵 $\Lambda=\Sigma^{-1}$——看到的是「直接相关」（条件独立结构）**。这条区别就是接触预测的全部要害。

### 5.4.1 精度矩阵的定义与「条件视角」

**精度矩阵（precision matrix / concentration matrix）** 是协方差的逆：

$$\Lambda=\Sigma^{-1}\quad(\Sigma\ \text{正定时存在}),\qquad \Lambda\ \text{也对称正定}.$$

「精度」这个名字来自一维：$\Lambda=1/\sigma^2$，方差越小、精度越高、对该变量「知道得越准」。在高斯对数密度 $\log f=-\tfrac12\mathbf x^\top\Lambda\mathbf x+\boldsymbol\eta^\top\mathbf x+\text{const}$ 里，$\Lambda$ 是**二次型的系数矩阵**——它直接出现在「自然参数」里（§5.3.1）。

关键洞察来自把条件分布公式（§5.3.4）用到**单个变量对其余所有变量**。设 $\mathbf X\sim\mathcal N(\mathbf 0,\Sigma)$（去均值不失一般），把 $X_i$ 单拎出来、其余记为 $\mathbf X_{-i}$。条件分布 $X_i\mid\mathbf X_{-i}$ 是高斯，其条件均值是 $\mathbf X_{-i}$ 的线性回归。一个漂亮的代数事实把回归系数和精度矩阵联系起来：

$$\mathbb E[X_i\mid\mathbf X_{-i}]=-\frac{1}{\Lambda_{ii}}\sum_{j\ne i}\Lambda_{ij}X_j,\qquad \mathrm{Var}(X_i\mid\mathbf X_{-i})=\frac{1}{\Lambda_{ii}}.$$

读这个式子：**把 $X_i$ 对所有其他变量做回归，$X_j$ 的回归系数正比于 $-\Lambda_{ij}$**。于是——

### 5.4.2 核心定理：零元素 = 条件独立

> **定理（高斯图模型的核心）。** 设 $\mathbf X\sim\mathcal N(\boldsymbol\mu,\Sigma)$，$\Lambda=\Sigma^{-1}$。则对 $i\ne j$，
> $$\boxed{\ \Lambda_{ij}=0\quad\Longleftrightarrow\quad X_i\perp X_j\,\big|\,\mathbf X_{-\{i,j\}}.\ }$$
> 即「精度矩阵第 $(i,j)$ 元为零」**精确等价于**「给定其余所有变量后，$X_i$ 与 $X_j$ 条件独立」。

**证明骨架**：由 §5.4.1，$X_i$ 对其余变量回归时，$X_j$ 的系数 $\propto-\Lambda_{ij}$。$\Lambda_{ij}=0$ 意味着「在已经知道除 $i,j$ 外所有变量的情况下，$X_j$ 对预测 $X_i$ **再无任何额外贡献**」——对高斯，条件分布由条件均值（**线性**）与**常数**条件协方差完全决定，故「回归系数为零」在此恰好升级为完整的条件独立（这是高斯特权；**一般分布则不然**——回归系数为零只说线性均值不依赖 $X_j$，绝不蕴含条件独立）。反方向同理。（严格证明可由块矩阵求逆 + Schur 补，把「条件协方差的非对角为零」翻译成「精度的非对角为零」。）

这条定理把**条件独立结构编码进了一个矩阵的稀疏模式**。据此定义**高斯图模型（Gaussian graphical model, GGM）**，也叫**高斯马尔可夫随机场（Gaussian Markov random field）**：

```
   节点 = 变量 X1,...,Xd
   边 (i,j) 存在  ⟺  Λ_ij ≠ 0  ⟺  X_i, X_j 给定其余变量后仍直接相关

   精度矩阵 Λ 的稀疏模式  =  图的邻接结构
   Λ 越稀疏  =  图越稀疏  =  条件独立关系越多
```

注意这与「协方差矩阵的稀疏模式」完全是两回事：$\Sigma_{ij}=0$ 表示 $X_i,X_j$ **边缘**不相关；$\Lambda_{ij}=0$ 表示它们**条件**独立。两者一般互不蕴含——这正是「直接 vs 间接」的数学分水岭，下一小节用一个具体例子把它打透。

### 5.4.3 链式例子：协方差被传递相关污染，精度矩阵不会

考虑三个变量的**链** $X_1 - X_2 - X_3$：$X_1$ 直接影响 $X_2$，$X_2$ 直接影响 $X_3$，但 $X_1$ 与 $X_3$ **没有直接联系**（给定 $X_2$ 后条件独立）。一个具体的精度矩阵（注意它在 $(1,3)$ 位置为零，编码 $X_1\perp X_3\mid X_2$）：

$$\Lambda=\begin{pmatrix}2&-1&0\\-1&2&-1\\0&-1&2\end{pmatrix}\ \xrightarrow{\ \text{求逆}\ }\ \Sigma=\Lambda^{-1}=\frac14\begin{pmatrix}3&2&1\\2&4&2\\1&2&3\end{pmatrix}.$$

看这里的对比，它是整节的灵魂：

| 量 | $(1,3)$ 元 | 含义 |
|---|---|---|
| 精度 $\Lambda_{13}$ | $\mathbf{0}$ | $X_1,X_3$ 给定 $X_2$ 后**条件独立**（无直接耦合） |
| 协方差 $\Sigma_{13}$ | $\mathbf{1/4\ne0}$ | $X_1,X_3$ **边缘相关**（相关系数 $\rho_{13}=\tfrac{1/4}{\sqrt{3/4\cdot3/4}}=\tfrac13$） |

**$X_1$ 和 $X_3$ 明明没有直接联系，协方差却非零**——因为相关沿着 $X_1\to X_2\to X_3$ 这条路径**传递（transitively）** 过去了。如果你只看协方差/相关，会误以为 $X_1,X_3$ 之间有联系；只有看精度矩阵，那个 $\Lambda_{13}=0$ 才诚实地告诉你「它们没有直接边」。

```
真实结构（精度矩阵的稀疏图）:      协方差「看到」的（含传递）:
   X1 ── X2 ── X3                  X1 ─── X2 ─── X3
   (1,3 无边, Λ13=0)               └──────────────┘
                                   (1,3 也相关 Σ13≠0 ← 间接/假信号)
```

这张图和信息论课程 ch10 §10.3.2、ch11 §11.3 里「表观耦合 vs 直接耦合」的图**一模一样**——绝非巧合。

> **与你的研究的连接（接触预测：为什么用精度矩阵而非相关矩阵）**：这就是 AlphaFold 之前共进化/接触预测范式的统计心脏，以及它的「高斯线性版本」。把 MSA 的每一列看成一个变量，列与列之间的协同变化（共进化）类比成上面的 $X_1,X_2,X_3$。两列在三维结构里**直接接触**才有物理上的直接耦合（一个突变，配对残基要补偿性突变以维持接触）。但「列相关」会被**传递相关污染**：若 $i$ 接触 $k$、$k$ 接触 $j$，则 $i,j$ 也会显得相关，哪怕 $i,j$ 在空间上根本不接触。用相关/互信息直接预测接触，会被这些**间接信号**淹没（大量假阳性）。**直接耦合分析（direct coupling analysis, DCA）** 的全部要义，就是去估「**精度矩阵的离散类比**」——给定其余所有列后，$i,j$ 是否仍直接耦合。高斯情形下，这件事就是字面意义的「估 $\Lambda=\Sigma^{-1}$、看非对角元」；离散（20 个氨基酸）情形下，它升级成 Potts 模型的耦合矩阵 $J_{ij}$（信息论 ch10 §10.3、ch11 §11.4，折叠课程第 5 章 §5.7.3）。**「协方差 → 精度矩阵」这一步求逆，就是 DCA「从表观耦合分离出直接耦合」的核心动作的线性版本。** 实务里甚至有方法（如 GaussDCA / PSICOV）直接对 MSA 的协方差矩阵求逆（加正则化）来做接触预测，把这条「线性版本」原样用上。理解了高斯的 $\Lambda_{ij}=0\Leftrightarrow$ 条件独立，你就理解了整条 DCA 主线为什么成立。

> **给数学/CS 读者的视角（DCA = Potts 是 GGM 的离散升级）**：高斯图模型与 Potts 模型是同一个思想的两个版本，对照表如下：
> | | 高斯图模型（GGM） | Potts / DCA |
> |---|---|---|
> | 变量类型 | 连续（实数） | 离散（20 个氨基酸 + gap） |
> | 模型 | $p\propto\exp(-\tfrac12\mathbf x^\top\Lambda\mathbf x)$ | $p\propto\exp(\sum h_i(x_i)+\sum J_{ij}(x_i,x_j))$ |
> | 「直接耦合」参数 | 精度矩阵元 $\Lambda_{ij}$（一个标量） | 耦合矩阵 $J_{ij}(a,b)$（$20\times20$ 块） |
> | 条件独立判据 | $\Lambda_{ij}=0$ | $J_{ij}\equiv0$ |
> | 求解难度 | 求逆 $O(d^3)$，**有闭式** | 配分函数 #P-难，须近似（伪似然/平均场） |
> 两者都是「成对约束下的最大熵分布」（本课第 4 章 §4.5、信息论 ch10）。高斯版有闭式（一次求逆搞定），是理解离散版（DCA 须用伪似然等近似）的最佳脚手架。先在高斯里把「$\Lambda_{ij}=0$ = 条件独立 = 没有直接边」想透，再去读 DCA，会顺畅得多。

### 5.4.4 偏相关与诚实的边界

把精度矩阵标准化，得到**偏相关系数（partial correlation）**：

$$\rho_{ij\cdot\text{rest}}=-\frac{\Lambda_{ij}}{\sqrt{\Lambda_{ii}\Lambda_{jj}}}.$$

它是「除掉其余所有变量的影响后，$X_i,X_j$ 之间残余的相关」。$\rho_{ij\cdot\text{rest}}=0\Leftrightarrow\Lambda_{ij}=0\Leftrightarrow$ 条件独立。接触预测里排序候选接触对，用的就是这种「直接」量而非原始相关——但要分清两条不同路线：**高斯/PSICOV 版**直接用偏相关 $\rho_{ij\cdot\text{rest}}$（或 $|\Lambda_{ij}|$）排序；**离散 DCA 版**则对 $20\times20$ 的耦合矩阵 $J_{ij}$ 取 Frobenius 范数得到标量分数后，再做 **APC 校正（average product correction，平均乘积校正）**去掉列保守性/采样深度带来的背景偏置（信息论 ch11 §11.4.4）。要点：APC 主要属于**后者**（作用在矩阵范数分数上），不要把它误绑到标量偏相关上。

> **诚实的边界（求逆的两个陷阱）**：(1) **必须能求逆**。当样本数 $n$ 小于维数 $d$（MSA 里序列数 < 列数 × 20 是常态），样本协方差 $\hat\Sigma$ **奇异**，$\hat\Sigma^{-1}$ 不存在或极不稳定（小扰动被放大）。解药是**正则化**：加岭项 $\hat\Sigma+\epsilon I$、用**图 Lasso（graphical lasso）** 对 $\Lambda$ 加 $\ell_1$ 罚以直接求稀疏精度矩阵、或收缩估计（Ledoit–Wolf）。这正是 GaussDCA/PSICOV 等方法的核心工程。(2) **「零元素 = 条件独立」严格只对高斯成立**。真实 MSA 数据是离散的、强非线性的，高斯只是一个**线性近似**；它能跑、且常常够好（接触预测的 top hits 多半对），但会漏掉非线性/高阶耦合——这也是为什么离散的 Potts/DCA（和后来端到端的深度网络）会赢过纯高斯版本。把高斯 GGM 当「理解直接 vs 间接的最干净模型」和「能跑的 baseline」，而不是「真相」。

> **要点**：精度矩阵 $\Lambda=\Sigma^{-1}$ 把**条件独立**写进它的零元素：$\Lambda_{ij}=0\Leftrightarrow X_i\perp X_j\mid$ 其余全部，这定义了高斯图模型（边 = 非零 $\Lambda$）。协方差看**总相关**（含传递路径，会把不接触却间接相关的对算进来），精度矩阵看**直接相关**——这正是接触预测里「表观耦合 vs 直接耦合」的分水岭，是 DCA/Potts 的高斯线性版本。偏相关 $=-\Lambda_{ij}/\sqrt{\Lambda_{ii}\Lambda_{jj}}$ 是排序候选接触的统计量。陷阱：$n<d$ 时 $\hat\Sigma$ 奇异、求逆须正则化（图 Lasso/岭/收缩）；且该等价严格只对高斯成立。（§5.4）

---

## 5.5 线性变换、白化与主成分分析

本节把 §5.2 的特征分解兑现成两个天天用的操作——**白化**（把相关变量解耦成各向同性）与 **PCA**（找数据散布最大的几个方向）——并把它们接到 MD 主成分与弹性网络模型。

### 5.5.1 白化：把椭球揉成球

**白化（whitening）** 是找一个线性变换，把有协方差结构 $\Sigma$ 的数据变成**协方差为单位阵**的数据（各分量不相关、方差都为 1）。给定 $\mathbf X$（均值 $\boldsymbol\mu$、协方差 $\Sigma$），令

$$\mathbf Z=W(\mathbf X-\boldsymbol\mu),\qquad \text{要求}\ \mathrm{Cov}(\mathbf Z)=W\Sigma W^\top=I.$$

由 §5.2.1 线性变换法则，任何满足 $W\Sigma W^\top=I$ 的 $W$ 都行。常见两种：

- **ZCA / Mahalanobis 白化**：$W=\Sigma^{-1/2}$（用谱分解 $\Sigma^{-1/2}=Q\Lambda_{\rm eig}^{-1/2}Q^\top$）。此时 $\mathbf Z$ 的每个分量的方差是 1，且对单个样本 $\|\mathbf Z\|^2=\mathbf Z^\top\mathbf Z=(\mathbf x-\boldsymbol\mu)^\top\Sigma^{-1}(\mathbf x-\boldsymbol\mu)$ 正是该样本的马氏距离平方——白化把马氏距离变成了普通欧氏（范数）距离。
- **PCA 白化**：$W=\Lambda_{\rm eig}^{-1/2}Q^\top$（先旋到主轴、再各轴除以标准差）。

> **为什么不止一种白化（平方根不唯一）**：满足 $W\Sigma W^\top=I$ 的 $W$ **不唯一**——若 $W$ 是一个解，任取正交阵 $U$，$UW$ 仍是解（因 $(UW)\Sigma(UW)^\top=U(W\Sigma W^\top)U^\top=UU^\top=I$）。所以一切白化变换只差一个旋转：ZCA 取对称平方根 $\Sigma^{-1/2}$（在所有解里**最贴近原坐标**，故图像处理偏爱它）、PCA 白化取旋到主轴的那个、下面采样里 Cholesky 取下三角的那个。这和「$LL^\top=\Sigma$ 的平方根 $L$ 不唯一、Cholesky 只是其中一个」是同一件事。

几何上，白化就是「把 §5.3.2 那个倾斜的椭球先转正（$Q^\top$）、再沿各轴缩放成单位球（$\Lambda_{\rm eig}^{-1/2}$）」。**采样是它的逆操作**：要从 $\mathcal N(\boldsymbol\mu,\Sigma)$ 抽样，先抽标准高斯 $\mathbf Z\sim\mathcal N(\mathbf 0,I)$，再 $\mathbf X=\boldsymbol\mu+L\mathbf Z$，其中 $LL^\top=\Sigma$（$L$ 取 **Cholesky 分解**最常用）——由仿射定理 $\mathrm{Cov}(\mathbf X)=L I L^\top=\Sigma$。第 10 章蒙特卡洛会反复用这招生成相关高斯样本。

### 5.5.2 主成分分析：$\Sigma$ 的特征分解就是 PCA

**主成分分析（principal component analysis, PCA）** 问：哪个方向 $\mathbf a$（单位向量）上数据投影的方差 $\mathbf a^\top\Sigma\mathbf a$ 最大？这是一个带约束的二次型优化，答案由谱定理直接给出：

> **定理（PCA = 协方差的特征分解）。** 设 $\Sigma=Q\Lambda_{\rm eig}Q^\top$，特征值 $\lambda_1\ge\lambda_2\ge\dots\ge\lambda_d\ge0$。则方差最大的方向是最大特征值对应的特征向量 $\mathbf q_1$（第一主成分），其上方差 $=\lambda_1$；在与 $\mathbf q_1$ 正交的方向中方差最大的是 $\mathbf q_2$，依此类推。

证明用拉格朗日乘子：$\max_{\|\mathbf a\|=1}\mathbf a^\top\Sigma\mathbf a$ 的一阶条件是 $\Sigma\mathbf a=\lambda\mathbf a$，即 $\mathbf a$ 是特征向量、目标值 $=\lambda$，取最大特征值即可。所以**主成分就是 §5.3.2 那个等密度椭球的主轴**，主成分的方差就是椭球的半轴长平方（按 $\sqrt{\lambda_i}$）。PCA 的实务是「保留前 $k$ 个主成分」做降维：前 $k$ 个特征值之和占总方差（$\mathrm{tr}\Sigma=\sum\lambda_i$）的比例，就是降维后保留的「解释方差比」。

> **与你的研究的连接（MD 的 essential dynamics 与构象空间降维）**：把一条分子动力学（molecular dynamics, MD）轨迹的每一帧（对齐后去掉平动转动）看成 $3N$ 维向量的一个样本，算它们的协方差矩阵 $\Sigma$ 并做特征分解——这正是结构生物学里的 **主成分分析 / 本质动力学（essential dynamics）**。前几个主成分（最大特征值方向）通常对应蛋白质的**大尺度集体运动**（domain 开合、loop 摆动、铰链弯曲），而高阶主成分是局部热噪声。于是「几万维的构象涨落」被压缩成「少数几个集体坐标」，可以画出 2D 的自由能投影（free energy landscape projection）、识别亚稳态、定义反应坐标。注意 §5.2.2 的零模陷阱：必须先去掉平动/转动 6 个零特征值方向，否则它们会以「最大方差」混进前几个主成分把你骗了。

### 5.5.3 弹性网络模型：蛋白质涨落的高斯模型

把上面的逻辑反过来用——**不从轨迹估 $\Sigma$，而从结构直接构造一个高斯模型**——就得到弹性网络模型，这是「多元高斯 = 蛋白质涨落」最优雅的落地。

**高斯网络模型（Gaussian network model, GNM）** 与**各向异性网络模型（anisotropic network model, ANM）** 把蛋白质建模成一堆用弹簧连起来的珠子（每个残基一个珠子，近邻残基间连弹簧）。在简谐（harmonic）近似下，珠子的涨落服从**多元高斯**，其精度矩阵就是简谐势的二阶导（刚度矩阵）。两者用的是同一思想的不同维度版本，但精度矩阵**形态不同**，须分开看：

**GNM（标量版，最干净）**：用 $N\times N$（$N$ 为残基数）的 **Kirchhoff 矩阵 / 图拉普拉斯（graph Laplacian）** $\Gamma$ 当精度矩阵——$\Gamma$ 就是接触图的图拉普拉斯（$\Gamma_{ij}=-1$ 若 $i,j$ 在截断距离内接触、对角为各残基的接触数）。它只建模**各向同性**的涨落幅度与残基间相关（标量，不含方向）：

$$p(\Delta\mathbf R)\propto\exp\!\Big(-\frac{\gamma}{2k_BT}\,\Delta\mathbf R^\top\,\Gamma\,\Delta\mathbf R\Big),\qquad \Sigma=\mathrm{Cov}(\Delta\mathbf R)\propto\Gamma^{-1}\ (\text{伪逆，因有零模}).$$

这里每一块都干净地呼应本章：精度矩阵 $\Gamma$ 由**接触图**（哪些残基相邻）直接给出——**它的稀疏结构就是高斯图模型的边**（§5.4）；协方差 $\Gamma^{-1}$ 给出残基涨落幅度（$\langle\Delta R_i^2\rangle\propto\Gamma^{-1}_{ii}$）与**残基间的相关运动**（非对角）；$\Gamma$ 的低频模（小特征值对应的特征向量）就是蛋白质最柔软的集体运动方向。

**ANM（方向版）**：用 $3N\times3N$ 的 **Hessian 矩阵** $H$（简谐势对 $3N$ 个笛卡尔坐标的二阶导）当精度矩阵——它是**各向异性**的，能给出运动的**方向**（每个残基对的超元素是个 $3\times3$ 小块，由连线方向余弦构成）。注意 $H$ **并不简单正比于图拉普拉斯 $\Gamma$**：$\Gamma$ 只记「谁和谁接触」，$H$ 还把「沿哪个方向」编码进了方向余弦。可以把 GNM 看成「把接触图编码进精度矩阵」最干净的标量版，ANM 看成它的方向（Hessian）升级版——前者算涨落幅度，后者还算简正模的运动方向。两者都极便宜（不跑动力学，解一次特征问题即可），是快速预测蛋白质柔性、铰链、变构通路的常用工具；**GNM/ANM 的低频简正模和 §5.5.2 从 MD 轨迹做 PCA 得到的前几个主成分高度一致**（两条路殊途同归：一个从数据估、一个从结构推）。

> **给数学/CS 读者的视角**：GNM 是一句话——「**把接触图的图拉普拉斯当精度矩阵的高斯**」（标量、各向同性）；ANM 是它的方向版——「把简谐势的 Hessian 当精度矩阵」（$3N$ 维、各向异性）。无论哪个，「蛋白质动力学」都被还原成「一个高斯图模型」，本章 §5.4 的全部语言（精度矩阵 = 图、协方差 = 涨落相关、低频模 = 柔性方向）直接照搬。这也再次显示精度矩阵的「物理直接性」：弹簧（接触）写在精度矩阵（GNM 的 $\Gamma$、ANM 的 $H$）里是稀疏的、局部的；而它诱导的涨落相关（协方差 = 精度的逆）是稠密的、长程的（远端残基也会相关运动）——**物理在精度矩阵里、表观在协方差里**，与接触预测同构。

> **要点**：白化 = 用 $\Sigma^{-1/2}$ 把相关椭球揉成各向同性球（采样是其逆：$\mathbf X=\boldsymbol\mu+L\mathbf Z$，$LL^\top=\Sigma$ 用 Cholesky）。PCA = $\Sigma$ 的特征分解，主成分 = 等密度椭球主轴、方差 = 特征值；用于 MD 的 essential dynamics（前几个主成分 = 大尺度集体运动，须先去平动转动零模）。GNM/ANM = 「把接触图编码进精度矩阵的高斯」（GNM 用 $N\times N$ 图拉普拉斯 $\Gamma$ 算各向同性涨落幅度，ANM 用 $3N\times3N$ Hessian 算各向异性运动方向，$H$ 不简单正比于 $\Gamma$），从结构直接推涨落，精度矩阵稀疏局部（弹簧）、协方差稠密长程（相关运动）——又一次「物理在精度、表观在协方差」。（§5.5）

---

## 5.6 诚实的边界：真实构象分布远不止一个高斯

高斯是工作母机，但绝不是真相。本节明确划出它的适用边界，并指向后续章节的解药。

### 5.6.1 高斯的三个系统性失效

1. **重尾（heavy tails）**。高斯密度按 $e^{-\|\mathbf x\|^2}$ 衰减，极快——它系统性**低估极端事件**。能量分布、力的分布、某些距离涨落有比高斯重的尾（罕见的大涨落比高斯预测的频繁得多）。用多元高斯做异常检测会漏掉真正的离群构象。解药方向：多元 $t$ 分布（重尾的椭圆分布）、混合模型。

2. **多峰（multimodality）**。高斯是**单峰**的（一个椭球状的山包）。但真实构象分布几乎总是**多峰**的：折叠态 vs 未折叠态、不同的亚稳构象、不同的结合姿态——每个都是自由能面上的一个盆地（basin）。用单个高斯去拟合多峰分布（极大似然/矩匹配），均值会落到各盆地的（加权）平均处——也就是**两个盆地之间的低密度区，一个没有任何构象真正驻留的位置**（最糟糕的位置）。严格说这个点未必是能量面拓扑意义上的「鞍点」（一阶导为零、Hessian 不定的特定临界点）；只在对称双峰的特例下它才恰好落在中间的势垒/鞍附近，一般它只是密度的低谷居中区。这是 §4.2.2、§4.3.3 在高维的重演。解药方向：**高斯混合模型（Gaussian mixture model, GMM）**——用若干个高斯加权叠加，每个盆地一个分量（第 7 章 EM 算法专门估它）。

3. **流形结构与非线性约束（manifold / nonlinear constraints）**。蛋白质构象受**硬约束**：键长键角几乎固定、二面角活在环面上（第 4 章 §4.3）、原子不能重叠（排斥体积）。真实可行构象住在一个**低维、弯曲的流形**上，而高斯假设的是「整个 $\mathbb R^d$ 上的椭球」。在弯曲流形上贴一个平直椭球，必然「漏到流形外」（给非物理构象分配概率）。这也是为什么二面角要用 von Mises 而非高斯。解药方向：流形上的分布、内坐标（internal coordinates）建模、以及深度生成模型（流模型、扩散模型）学复杂分布。

### 5.6.2 椭圆分布与 copula：一句话的逃生口

放松高斯有两条标准路线，各记一句话即可：

- **椭圆分布（elliptical distribution）**：保留「等密度面是椭球」这个几何（因而仍有协方差/主轴的语言），但允许尾部更重——密度形如 $g\big((\mathbf x-\boldsymbol\mu)^\top\Sigma^{-1}(\mathbf x-\boldsymbol\mu)\big)$，$g$ 不必是 $e^{-t/2}$。多元 $t$、多元 Laplace 都在此列。许多关于「协方差/线性回归」的直觉对整个椭圆族都成立，但「不相关 ⇒ 独立」这条**只对高斯成立**（其余椭圆分布即便不相关也可能依赖）。事实上，**在椭圆分布族里，高斯是唯一一个分量可以相互独立的成员**——这正是「不相关 ⇒ 独立」为高斯独有的精确刻画。一个有教学价值的反直觉例子：多元 $t$ 的两个分量即使协方差为零，仍通过一个**公共的尺度混合**（所有分量共享同一个 $\chi^2$ 分母）耦合，使得「大尾事件倾向同时发生」（尾相依，tail dependence）——这与下面 copula 段落里 2008 危机的翻车是**同源**现象。

- **Copula**：把「边缘」与「相关结构」**解耦**——先各自建模每个变量的边缘分布（可以是任意形状），再用一个 copula 函数把它们粘成联合、单独刻画依赖结构。**高斯 copula** 就是「借用高斯的相关结构、但配任意边缘」。这在金融风险和某些生信场景常用；它的著名翻车（2008 危机里高斯 copula 低估联合尾部风险）正是「相关结构不足以刻画联合极端事件」的教训。

> **诚实的边界（为什么折叠最终需要采样与生成模型）**：把上面三条加起来——真实构象分布是**高维、多峰、重尾、住在弯曲流形上**的玻尔兹曼分布 $p(\mathbf x)\propto e^{-E(\mathbf x)/k_BT}$（折叠课程第 4 章），它**根本不是一个高斯**。这就是为什么本课程后面要花整章讲**蒙特卡洛与 MCMC**（第 10 章）：当分布无法用封闭形式（如高斯）写下、只能逐点算未归一化密度时，唯一的出路是**采样**（Metropolis、HMC、模拟退火），用样本去逼近期望与边缘。而 AlphaFold 之后的**生成式折叠/设计**（折叠课程第 8、9 章的扩散模型、RFdiffusion）则是用神经网络去学这个复杂分布——高斯只在其中作为「局部线性近似」（如 IPA 里的局部坐标、扩散过程的高斯转移核）出现。把高斯当**脚手架和局部近似**，把多峰/流形/采样当**真相**——这是本章留给第 7（混合模型/EM）、第 10（MCMC）章的接力棒。

> **要点**：高斯系统性失效于三处——重尾（低估极端事件）、多峰（真实构象是多盆地，单高斯均值落在盆地之间的低密度区，没有构象真正驻留）、流形/非线性约束（构象住在弯曲低维流形，平直椭球会漏到流形外）。逃生口：椭圆分布（保椭球几何、放宽尾部，但「不相关⇒独立」只属高斯）、copula（边缘与依赖解耦）、高斯混合（多峰，第 7 章 EM）。真实构象分布是高维多峰重尾的玻尔兹曼分布，不是一个高斯——这正是后续 MCMC（第 10 章）与生成模型（折叠 ch8/ch9）的存在理由。（§5.6）

---

## 5.7 本章小结

- **随机向量的分布是联合**；积掉维度得边缘（信息损失、不可逆），固定维度得条件（信息更新）。耦合只活在联合里，边缘看不到——这是 MSA「列频率不够、要估成对」的根。条件独立是「直接 vs 间接」的精确语言。（§5.1）
- **协方差矩阵 $\Sigma$ 对称、半正定**：$\mathbf a^\top\Sigma\mathbf a=\mathrm{Var}(\mathbf a^\top\mathbf X)\ge0$；特征向量 = 主轴、特征值 = 主轴方差。奇异 = 数据落在低维子空间（MD 平动转动零模、$n<d$），求逆前须正则化。线性变换法则 $\mathrm{Cov}(A\mathbf X+\mathbf b)=A\Sigma A^\top$ 是全章主力公式。（§5.2）
- **不相关 ≠ 独立**：相关只抓线性（$X,X^2$ 不相关却依赖）；唯有**联合**高斯时不相关 ⇔ 独立（要联合，不是各自边缘）。（§5.2）
- **多元高斯对仿射/求和/边缘/条件全封闭**，只追踪 $(\boldsymbol\mu,\Sigma)$。密度由马氏距离定，等高线是椭球（主轴 = $\Sigma$ 特征向量、半长 $\propto\sqrt{\lambda_i}$）。任意线性组合是一维高斯（可作定义）。（§5.3）
- **高斯条件还是高斯**：条件均值 = 线性回归 $\boldsymbol\mu_A+\Sigma_{AB}\Sigma_{BB}^{-1}(\mathbf x_B-\boldsymbol\mu_B)$，条件协方差 = Schur 补 $\Sigma_{AA}-\Sigma_{AB}\Sigma_{BB}^{-1}\Sigma_{BA}$（比边缘小、不依赖观测值）。这是高斯过程、卡尔曼滤波、坐标补全的共同引擎。（§5.3）
- **精度矩阵 $\Lambda=\Sigma^{-1}$ 编码条件独立**：$\Lambda_{ij}=0\Leftrightarrow X_i\perp X_j\mid$ 其余全部，定义高斯图模型（边 = 非零 $\Lambda$）。协方差看总相关（被传递路径污染），精度矩阵看直接相关——三变量链例子里 $\Lambda_{13}=0$ 但 $\Sigma_{13}\ne0$。（§5.4）
- **这是 DCA/Potts 的高斯线性版本**：接触预测要分离「直接耦合」就得看精度矩阵/偏相关而非相关/互信息；GGM（连续、求逆有闭式）是离散 Potts/DCA（须伪似然近似）的最佳脚手架。$n<d$ 须正则化（图 Lasso/岭/收缩），且「零元素=条件独立」严格只对高斯。（§5.4）
- **白化 = 用 $\Sigma^{-1/2}$ 揉椭球为球**（采样是其逆，Cholesky）；**PCA = $\Sigma$ 的特征分解**，主成分 = 椭球主轴，用于 MD 的 essential dynamics（须先去平动转动零模）。（§5.5）
- **GNM/ANM = 把接触图编码进精度矩阵的高斯**（GNM：$N\times N$ 图拉普拉斯 $\Gamma$，各向同性涨落幅度；ANM：$3N\times3N$ Hessian，各向异性运动方向，二者不简单成正比）：从结构直接推涨落，精度稀疏局部（弹簧）、协方差稠密长程（相关运动）——又一次「物理在精度、表观在协方差」。（§5.5）
- **高斯不是真相**：真实构象分布高维、多峰、重尾、住在弯曲流形上，是玻尔兹曼分布而非一个高斯。逃生口为椭圆分布/copula/高斯混合；根本出路是采样（第 10 章 MCMC）与生成模型（折叠 ch8/ch9）。（§5.6）

---

## 5.8 动手 / 思考小练习

**1（动手·编程，看椭圆与它的主轴）。** 仅用 `numpy/matplotlib`：取 $\boldsymbol\mu=(0,0)$、$\Sigma=\begin{pmatrix}3&1.5\\1.5&1\end{pmatrix}$。(a) 用 `np.random.multivariate_normal` 抽 2000 个样本画散点。(b) 用 `np.linalg.eigh(Sigma)` 求特征值/特征向量，在散点上画出两条主轴（方向 = 特征向量，长度 $\propto\sqrt{\lambda_i}$）。(c) 改 $\Sigma_{12}$ 从 0 增大到接近 $\sqrt{3\cdot1}\approx1.73$，观察椭圆如何变扁、变斜、趋于一条线。
（*提示*：(b) `vals,vecs=np.linalg.eigh(Sigma)`，第 $i$ 根主轴端点为 $\pm2\sqrt{\text{vals}[i]}\cdot\text{vecs}[:,i]$。(c) $\Sigma_{12}\to1.73$ 时 $\det\Sigma\to0$、$\Sigma$ 趋于奇异、$\rho\to1$，对应 §5.2.2 退化与 §5.3.2 椭圆塌缩成线。这就是 `05_multivariate_gaussian.html` 在做的事。）

**2（思考·辨析，不相关不是独立）。** 设 $X\sim\mathcal N(0,1)$、$Y=X^2-1$。(a) 证明 $\mathrm{Cov}(X,Y)=0$。(b) $X$ 与 $Y$ 独立吗？(c) 这与「联合高斯下不相关 ⇔ 独立」矛盾吗？为什么不矛盾？
（*提示*：(a) $\mathrm{Cov}(X,Y)=\mathbb E[X^3]-\mathbb E[X]=0-0=0$（标准正态奇矩为零）。(b) 不独立——$Y$ 由 $X$ 完全决定。(c) 不矛盾：$(X,Y)$ **不是联合高斯**（$Y$ 是 $X$ 的二次函数，$Y$ 边缘是平移卡方而非高斯），那条定理的前提不满足，§5.2.4 的 CS 陷阱说的正是这件事。）

**3（动手·编程，精度矩阵 vs 协方差：传递相关，与研究相关）。** 仅用 `numpy`：取链式精度矩阵 $\Lambda=\begin{pmatrix}2&-1&0\\-1&2&-1\\0&-1&2\end{pmatrix}$。(a) 求 $\Sigma=\Lambda^{-1}$，验证 $\Lambda_{13}=0$ 但 $\Sigma_{13}\ne0$。(b) 算相关矩阵和偏相关矩阵（$-\Lambda_{ij}/\sqrt{\Lambda_{ii}\Lambda_{jj}}$），比较它们对「1 和 3 之间有无直接边」的判断。(c) 说明这对接触预测意味着什么。
（*提示*：(a) `np.linalg.inv(Lambda)` 得 §5.4.3 那个 $\Sigma$，$\Sigma_{13}=1/4\ne0$。(b) 相关 $\rho_{13}=1/3\ne0$（误报有联系），偏相关 $\rho_{13\cdot2}=0$（正确报告无直接边）。(c) 若把 1,2,3 当三列共进化位点，「相关」会因传递把不接触的 1,3 误判为接触，「偏相关/精度矩阵」才指向真接触——这是 §5.4 接触预测与 DCA 的全部要害。）

**4（动手·编程，高斯条件 = 线性回归，与研究相关）。** 仅用 `numpy`：取 2D 高斯 $\boldsymbol\mu=(0,0)$、$\Sigma=\begin{pmatrix}1&0.8\\0.8&1\end{pmatrix}$。(a) 用 §5.3.4 公式算 $\mathbb E[X_1\mid X_2=x_2]$ 与 $\mathrm{Var}(X_1\mid X_2=x_2)$，给出关于 $x_2$ 的显式表达。(b) 抽 5000 个样本，把 $X_2$ 落在 $[1.4,1.6]$ 的那些点的 $X_1$ 平均，与 (a) 在 $x_2=1.5$ 的预测对比。(c) 解释「条件方差不依赖 $x_2$」这件高斯独有的事，以及它在「已知部分残基坐标补全其余」时的含义。
（*提示*：(a) $\mathbb E[X_1\mid X_2=x_2]=0+0.8\cdot1^{-1}\cdot x_2=0.8x_2$；$\mathrm{Var}=1-0.8\cdot1^{-1}\cdot0.8=0.36$（与 $x_2$ 无关）。(b) 经验条件均值应 $\approx0.8\times1.5=1.2$。(c) 高斯里观测值只移动条件均值、不改变条件散布——所以补全坐标时「残余涨落」是恒定的 Schur 补，§5.3.4 的研究连接。）

**5（思考·证明，Schur 补使不确定性单调下降）。** 用 §5.3.4 证明 $\Sigma_{A\mid B}=\Sigma_{AA}-\Sigma_{AB}\Sigma_{BB}^{-1}\Sigma_{BA}\preceq\Sigma_{AA}$（半正定序）。这在信息上意味着什么？
（*提示*：因 $\Sigma_{BB}^{-1}$ 正定，$\Sigma_{AB}\Sigma_{BB}^{-1}\Sigma_{BA}=\Sigma_{AB}\Sigma_{BB}^{-1}\Sigma_{AB}^\top$ 半正定——它形如 $M^\top M$（取 $M=\Sigma_{BB}^{-1/2}\Sigma_{BA}$），$M^\top M$ 总半正定；从 $\Sigma_{AA}$ 里减去一个半正定阵，结果 $\preceq\Sigma_{AA}$，且**等号 $\iff$ $\Sigma_{AB}=0$**（$A,B$ 不相关时观测 $B$ 一点也不减少 $A$ 的不确定性）。信息含义：**观测 $\mathbf X_B$ 只会减少或不改变 $\mathbf X_A$ 的不确定性，绝不增加**——这里的「不增加」指的是高斯下**恒定的**条件协方差（Schur 补）$\preceq$ 边缘协方差；它对一般分布只在**平均意义**上成立（$\mathbb E[\mathrm{Var}(X_A\mid X_B)]\le\mathrm{Var}(X_A)$，正是下面方差分解里的那一项），对某个**具体**观测值 $\mathbf x_B$，非高斯的条件方差是可能临时变大的。这正呼应本课第 3 章条件期望的方差分解 $\mathrm{Var}(X)=\mathbb E[\mathrm{Var}(X\mid Y)]+\mathrm{Var}(\mathbb E[X\mid Y])$。）

**6（思考·辨析，单高斯能描述构象分布吗，与研究相关）。** 某同学把一个小蛋白 10 万帧 MD 轨迹（对齐后）建模成单个多元高斯，用它的密度做「这帧构象正常吗」的异常检测，结果折叠态和未折叠态的帧都被判为「正常」。(a) 用 §5.6 指出至少两个根本问题。(b) 该用什么模型替代？(c) 为什么这类问题最终把我们逼向第 10 章的采样方法？(d) （额外·椭圆分布陷阱）一个二维多元 $t$ 分布的两个分量协方差恰好为零，它们独立吗？为什么这与「高斯下不相关 ⇔ 独立」不冲突？
（*提示*：(a) 真实构象分布**多峰**（折叠/未折叠/中间态各一盆地），单高斯把均值放在盆地之间的**低密度区**（没有构象真正驻留的位置，未必是能量面的鞍点）、对所有盆地都「不太像又不太不像」；且构象住在**弯曲流形**上（键长键角约束、二面角在环面），平直椭球会给非物理构象分配概率。(b) **高斯混合模型**（每个亚稳态一个分量，第 7 章 EM 估），或在合适内坐标 / 流形上建模、或深度生成模型。(c) 真实分布是无封闭形式的玻尔兹曼分布 $\propto e^{-E/k_BT}$，只能逐点算未归一化密度，于是只能靠 MCMC 采样去逼近——这正是第 10 章的存在理由，§5.6 的诚实边界。(d) **不独立**：多元 $t$ 的所有分量共享同一个 $\chi^2$ 尺度混合（公共分母），即便协方差为零，这个公共尺度也让它们的大尾事件倾向同时发生（尾相依），故仍依赖。不冲突，因为「不相关 ⇒ 独立」**只对高斯成立**——多元 $t$ 是另一种椭圆分布，而 §5.6.2 指出高斯是椭圆族里**唯一**分量可独立的成员。）

---

## 5.9 深入阅读指引

**多元高斯与图模型（本章核心 §5.3–5.4）**
- **Bishop, *Pattern Recognition and Machine Learning*（2006），第 2 章 §2.3「The Gaussian Distribution」**：多元高斯的边缘/条件/Schur 补、信息形式、最大似然估计讲得极清楚，是本章 §5.3 的标准参考，公式与本章记号几乎一致。
- **Koller & Friedman, *Probabilistic Graphical Models*（2009），高斯网络相关章节**：高斯图模型、精度矩阵与条件独立、信息形式推断的权威处理——对应本章 §5.4，想把图模型学深的首选。
- **Murphy, *Machine Learning: A Probabilistic Perspective*（2012），多元高斯与 GGM 章节**：MVN 的全套封闭性质 + 图 Lasso/稀疏精度估计的实用视角——对应 §5.4.4 的正则化求逆。

**协方差估计与高维（深入 §5.2、§5.4.4）**
- **Friedman, Hastie & Tibshirani, *"Sparse inverse covariance estimation with the graphical lasso"*（Biostatistics, 2008）**：图 Lasso 原始论文，$n<d$ 时如何稳健地估稀疏精度矩阵——对应 §5.4.4，接触预测/网络重建的算法基石。
- **Ledoit & Wolf, *"A well-conditioned estimator for large-dimensional covariance matrices"*（2004）**：收缩估计，高维下让样本协方差可逆且稳定——对应 §5.2.2、§5.4.4 的求逆陷阱。

**与蛋白质研究的桥梁（§5.4–5.5）**
- **Morcos et al., *"Direct-coupling analysis of residue coevolution captures native contacts"*（PNAS, 2011）**：DCA 的奠基论文，把「精度矩阵 / 直接耦合分离间接相关」用到接触预测——本章 §5.4 研究连接的离散升级版，必读。
- **Jones et al., *"PSICOV: precise structural contact prediction using sparse inverse covariance estimation"*（Bioinformatics, 2012）**：直接对 MSA 协方差做稀疏逆（图 Lasso）预测接触——本章「高斯线性版 DCA」的字面实现，把 §5.4 + §5.4.4 原样用上。
- **Bahar, Lezon, Bakan & Dubey, *"Normal mode analysis of biomolecular structures: GNM/ANM"*（综述，Chem. Rev. 2010）**：高斯/各向异性网络模型的权威综述，「图拉普拉斯当精度矩阵的高斯」——对应 §5.5.3。
- **Amadei, Linssen & Berendsen, *"Essential dynamics of proteins"*（Proteins, 1993）**：MD 轨迹 PCA / 本质动力学的开创论文——对应 §5.5.2。
- **信息论课程 ch10 §10.3、ch11 §11.3–11.4 与折叠课程第 5 章 §5.7**：把本章的高斯线性版本升级成离散 Potts/DCA（伪似然、平均场、APC 校正），务必对照阅读，看「连续→离散」这一跳。

> **下一章预告**：本章我们把高维变量的「静态结构」（协方差、高斯、图模型）讲透了，但一直回避了一个问题：当我们有**很多个**样本（很多帧、很多序列）时，**样本均值、样本协方差会收敛到真值吗？收敛得多快？围绕真值如何波动？** 第 6 章《收敛、大数定律与中心极限定理》就来回答。**大数定律（law of large numbers）** 保证样本平均收敛到期望（你的 MD 时间平均能逼近系综平均的依据）；**中心极限定理（central limit theorem）** 进一步说「样本均值围绕真值的波动近似高斯」——这把本章的多元高斯从「一个建模假设」提升为「一条普遍的极限定律」，并解释为什么误差棒、置信区间能用正态来算。本章 §5.3 埋下的「高斯是工作母机」，将在第 6 章获得它最深的辩护：不是因为高斯方便，而是因为**求和/平均这件事本身把万物推向高斯**。
