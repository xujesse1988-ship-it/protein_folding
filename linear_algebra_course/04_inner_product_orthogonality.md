# 第 4 章 内积、范数、正交性与投影：给空间装上尺子与量角器

> **本章定位**：前三章我们把空间（第 2 章）和变换（第 3 章）讲透了，但一直在偷偷赖账——第 2 章说向量「有方向」、第 3 章说旋转「保长度」、投影「拍扁」、外积测「相似」，可我们从未正式定义过**长度、角度、垂直**这些词。一个裸的向量空间其实是「无尺度的」：它知道谁和谁能线性组合，却不知道哪个向量更长、哪两个向量成直角。本章就给向量空间装上一套**几何测量工具**——核心是一个叫**内积（inner product）** 的小小双线性函数，它一出现，长度（范数）、角度（夹角余弦）、垂直（正交）就全部被一举定义出来，向量空间瞬间从「线性结构」升级成「几何结构」（欧几里得几何）。这次升级的回报，正是折叠研究里最频繁的操作：**RMSD = 叠合后坐标差的 $\ell_2$ 范数（均方根，root-mean-square deviation）**，**嵌入相似度 = 归一化内积（余弦）**，**正交矩阵 = 旋转 = 保持分子内部一切距离的合法刚体变换**，**用 Frobenius 范数比较两张接触图 / 两个距离矩阵**，而把这一切推到极致的，是本课第一个真正的「最优」问题——**最小二乘（least squares）**：当 $A\mathbf x=\mathbf b$ 无解时，把 $\mathbf b$ **正交投影**到 $A$ 的列空间，求最近的近似解。本章按七步走：先立内积公理、用内积定义长度与角度（§4.1）；再讲各类范数与两个不等式（Cauchy–Schwarz、三角，§4.2）；接着是正交、标准正交基、正交补、勾股与 Parseval——在正交基下坐标 = 内积，算起来出奇地省（§4.3）；然后是本章的几何高潮**正交投影**与投影矩阵 $P=A(A^TA)^{-1}A^T$，并第一次请出配套交互页面 `04_projection_least_squares.html`（§4.4）；再用 Gram–Schmidt 把任意基「正交化」（预告 QR，第 5 章）（§4.5）；随后正式确立**正交矩阵 = 保内积 = 保距 = 旋转 / 反射**，直通第 10 章的 $\mathrm{SO}(3)$（§4.6）；最后把最小二乘的几何完整收口（§4.7）。读完本章，你看到任何「测量相似 / 求最近 / 找最优拟合 / 比较两个结构」的场景，都应当能立刻翻译成「内积 / 范数 / 投影」三件套，并看清它通向 QR、PCA、Kabsch、注意力打分的那几条路。

---

## 4.1 内积：给空间装上尺子与量角器

### 4.1.1 从「点积」说起：它早就在你身边

你在高中和前几章里其实已经反复用到一个运算——两个向量 $\mathbf u,\mathbf v\in\mathbb R^n$ 的**点积（dot product）**：

$$
\mathbf u\cdot\mathbf v = \mathbf u^T\mathbf v = \sum_{i=1}^n u_i v_i.
$$

第 3 章 §3.1.2 的「行视角」里，$A\mathbf x$ 的每个分量就是一个点积；注意力打分 $QK^T$ 的每个元素也是一个点积。点积的神奇之处在于：它把两个向量「揉」成一个标量，而这个标量恰好编码了**两件几何信息**——它们各自有多长、它们之间夹角多大。具体地，欧氏几何里有那条你或许见过的公式：

$$
\mathbf u\cdot\mathbf v = \|\mathbf u\|\,\|\mathbf v\|\cos\theta,
$$

其中 $\theta$ 是两向量的夹角。但请注意一件容易被忽略的事：上式里的 $\|\cdot\|$（长度）和 $\theta$（角度）**本身又是怎么定义的**？如果我们把它们当成「物理上拿尺子量出来的」，那线性代数就退化成了画图。真正的现代视角恰恰相反——**我们用点积来定义长度和角度**，而不是反过来。这就是本章的观念跃迁：内积是第一性的，长度和角度是它的派生品。

### 4.1.2 内积的公理

我们把「点积」抽象成一个满足三条公理的运算，叫**内积（inner product）**。一个实向量空间 $V$ 上的内积是一个函数 $\langle\cdot,\cdot\rangle: V\times V\to\mathbb R$，对任意 $\mathbf u,\mathbf v,\mathbf w\in V$ 和标量 $a,b\in\mathbb R$ 满足：

| 公理 | 名称 | 含义 |
|---|---|---|
| $\langle\mathbf u,\mathbf v\rangle=\langle\mathbf v,\mathbf u\rangle$ | **对称性（symmetry）** | 谁在前谁在后无所谓 |
| $\langle a\mathbf u+b\mathbf w,\mathbf v\rangle=a\langle\mathbf u,\mathbf v\rangle+b\langle\mathbf w,\mathbf v\rangle$ | **（第一变元）线性 / 双线性（bilinearity）** | 对加法和数乘「分配」；由对称性可知对第二变元也线性 |
| $\langle\mathbf v,\mathbf v\rangle\ge 0$，且 $=0\iff\mathbf v=\mathbf 0$ | **正定性（positive-definiteness）** | 自己跟自己的内积是「长度平方」，非负，且只有零向量长度为零 |

满足这三条的运算就有资格叫内积；装了内积的向量空间叫**内积空间（inner product space）**；有限维实内积空间也叫**欧几里得空间（Euclidean space）**。标准点积 $\mathbf u^T\mathbf v$ 是 $\mathbb R^n$ 上最常用的内积，但它**只是众多内积中的一个**——这点下一小节会重要起来。

> **给数学 / CS 读者的视角（内积 = 一个「带正定性约束的双线性 API」）**：把内积想成一个接口 `inner(u, v) -> scalar`。三条公理就是这个接口的契约（contract）：对称 = 参数顺序无关；双线性 = 对线性组合可以「拆开算」（这让一切都能化归到基向量两两的内积，见 §4.1.4 的 Gram 矩阵）；正定 = `inner(v, v)` 永远 $\ge 0$ 且仅零向量取零——这一条最关键，它保证「长度」是良定义的、不会出现负长度或非零向量长度为零的怪物。任何满足这份契约的实现都能复用全套欧氏几何工具（范数、正交、投影、Gram–Schmidt），这正是抽象的威力：**写一遍几何，处处可用**（点云、函数、随机变量、核方法都来认领这个接口）。

### 4.1.3 由内积定义长度与角度

有了内积，**范数（norm）/ 长度（length）** 直接定义为：

$$
\|\mathbf v\| := \sqrt{\langle\mathbf v,\mathbf v\rangle}.
$$

正定性保证根号下非负、且只有零向量长度为零，所以这是个合法的「长度」。对标准点积，$\|\mathbf v\|=\sqrt{\sum_i v_i^2}$，就是熟悉的欧氏长度（勾股定理的 $n$ 维版）。

**角度（angle）** 则通过 §4.1.1 那条公式**反解**出来——我们直接**定义**两个非零向量的夹角 $\theta$ 为：

$$
\cos\theta := \frac{\langle\mathbf u,\mathbf v\rangle}{\|\mathbf u\|\,\|\mathbf v\|}\in[-1,1].
$$

这个定义要合法，必须保证右边落在 $[-1,1]$ 里（否则 $\cos\theta$ 无意义）——这正是 §4.2 要证的 **Cauchy–Schwarz 不等式**在背后撑腰。一旦角度有了定义，**垂直 / 正交（orthogonal）** 就是最干脆的特例：

$$
\mathbf u\perp\mathbf v \iff \langle\mathbf u,\mathbf v\rangle=0 \quad(\theta=90°,\ \cos\theta=0).
$$

注意 $\theta$ 的符号约定：$\cos\theta=1$（同向）、$=-1$（反向）、$=0$（垂直）、$>0$（夹角锐角，「方向相似」）、$<0$（钝角，「方向相反」）。这把「相似度」量化了——这正是嵌入相似度的来源。

> **与你的研究的连接（嵌入相似度 = 归一化内积 = 余弦相似度）**：折叠与序列模型里，每个残基 / 每条蛋白 / 每个氨基酸都被编码成一个高维向量（embedding，第 2 章；氨基酸 one-hot 是 20 维向量）。判断「两个 embedding 有多像」，标准做法是**余弦相似度（cosine similarity）** $\cos\theta=\dfrac{\langle\mathbf u,\mathbf v\rangle}{\|\mathbf u\|\|\mathbf v\|}$——也就是把两向量归一化到单位长再取内积。它只看**方向**、不看模长，所以「同义但强度不同」的两个表示能被判为相似。注意力打分 $QK^T$ 是**未归一化**的内积（再除以 $\sqrt{d}$ 做缩放、过 softmax），本质也是「查询与键有多对齐」的内积度量。所以你看 AlphaFold / Transformer 的注意力，第一层几何直觉应是：**它在用内积衡量谁和谁方向对齐，对齐度高的就多分配权重**。检索式方法（如用结构 / 序列 embedding 做近邻搜索找同源蛋白）也完全建立在这个内积 / 余弦度量之上。

### 4.1.4 一般内积：点积只是一种内积

公理没规定内积必须是 $\sum u_iv_i$。换一个内积，「长度、角度、垂直」全都随之改变——几何被重新定义了。两个重要例子：

**（1）加权内积 / 一般二次型内积。** 给定一个对称正定矩阵 $M$（第 7 章会精确定义「正定」），定义

$$
\langle\mathbf u,\mathbf v\rangle_M := \mathbf u^T M\,\mathbf v.
$$

正定性保证 $\mathbf v^T M\mathbf v>0$（非零 $\mathbf v$），符合内积公理。当 $M=I$ 时退化为标准点积；当 $M$ 是对角阵 $\operatorname{diag}(w_1,\dots,w_n)$ 时，$\langle\mathbf u,\mathbf v\rangle_M=\sum_i w_i u_iv_i$，即各坐标被赋予不同权重（重要的坐标量得更「重」）。这种加权内积在统计里无处不在：**马氏距离（Mahalanobis distance）** $\sqrt{(\mathbf x-\boldsymbol\mu)^T\Sigma^{-1}(\mathbf x-\boldsymbol\mu)}$ 就是以精度矩阵 $\Lambda=\Sigma^{-1}$ 为度量的加权范数（《概率与统计》第 5 章多元高斯），它「沿方差大的方向放宽、沿方差小的方向收紧」，是「考虑了相关性的距离」。

**（2）函数内积。** 在连续函数空间上，定义

$$
\langle f,g\rangle := \int_a^b f(x)\,g(x)\,\mathrm dx.
$$

它同样满足三条公理（正定性需要一点分析）。于是「两个函数正交」「函数的长度（$L^2$ 范数）」「把函数投影到一组基函数上」全都有了意义——这正是**傅里叶级数（Fourier series）**的几何根基：$\{\sin nx,\cos nx\}$ 是函数空间里的一组正交基，傅里叶系数 = 函数在这组正交基上的内积（§4.3 的「坐标 = 内积」在无穷维的兑现）。

> **诚实的边界（换内积 = 换几何；别默认「内积」就是点积）**：很多人把「内积」和「点积 $\sum u_iv_i$」当同义词，这在 $M=I$ 时没错，但一般不然。**度量是一个选择，不是天赋。** 你选了哪个内积，就选了哪一套「长度 / 角度 / 正交」。两个在标准点积下不正交的向量，换个加权内积可能就正交了。这件事在折叠里很要紧：用「原子坐标的欧氏距离」还是「考虑了原子质量加权的距离」「考虑了局部柔性（B 因子）加权的距离」，是不同的度量、给出不同的「最优叠合」。本章为简洁默认用标准点积，但请始终记得：**所有结论里的「正交 / 长度 / 投影」都是相对于某个选定内积而言的**；换内积，公式形式不变，但 $I$ 要换成 $M$（练习 8）。真正需要无穷维 / 测度论严格性的函数内积细节（完备性、希尔伯特空间），点到为止，留待将来。

> **要点（§4.1）**：内积 = 满足对称、双线性、正定三公理的 $\langle\cdot,\cdot\rangle$。它一出现就**派生**出长度 $\|\mathbf v\|=\sqrt{\langle\mathbf v,\mathbf v\rangle}$、夹角 $\cos\theta=\langle\mathbf u,\mathbf v\rangle/(\|\mathbf u\|\|\mathbf v\|)$、正交 $\langle\mathbf u,\mathbf v\rangle=0$。点积 $\mathbf u^T\mathbf v$ 只是 $\mathbb R^n$ 上最常用的一种内积；加权内积 $\mathbf u^TM\mathbf v$（→马氏距离）、函数内积 $\int fg$（→傅里叶）同样合法。**换内积 = 换几何**，嵌入相似度 / 注意力打分都是内积度量。

---

## 4.2 范数与不等式：度量「大小」的多种方式

### 4.2.1 范数的公理与常见范数

范数（norm）是「长度」的一般化，记 $\|\cdot\|:V\to\mathbb R_{\ge0}$，要满足三条：

1. **正定**：$\|\mathbf v\|\ge0$，且 $=0\iff\mathbf v=\mathbf 0$；
2. **齐次**：$\|a\mathbf v\|=|a|\,\|\mathbf v\|$（放大 $a$ 倍长度放大 $|a|$ 倍）；
3. **三角不等式**：$\|\mathbf u+\mathbf v\|\le\|\mathbf u\|+\|\mathbf v\|$（绕路不会更短）。

由内积导出的 $\|\mathbf v\|=\sqrt{\langle\mathbf v,\mathbf v\rangle}$ 一定是范数（三角不等式由 Cauchy–Schwarz 给出，下面证），但**并非所有范数都来自内积**——这是个关键区分。常见向量范数（统称 $\ell_p$ 范数，$\|\mathbf v\|_p=(\sum_i|v_i|^p)^{1/p}$）：

| 范数 | 定义 | 几何 / 用途 |
|---|---|---|
| **$\ell_2$（欧氏）** | $\|\mathbf v\|_2=\sqrt{\sum_i v_i^2}$ | 直线距离；唯一**来自内积**的 $\ell_p$；单位球是圆 / 球 |
| **$\ell_1$（曼哈顿）** | $\|\mathbf v\|_1=\sum_i|v_i|$ | 「街区距离」；诱导**稀疏**（LASSO）；单位球是菱形 |
| **$\ell_\infty$（最大）** | $\|\mathbf v\|_\infty=\max_i|v_i|$ | 最坏分量；单位球是方块 |

矩阵也有范数，最常用两个：

| 矩阵范数 | 定义 | 含义 |
|---|---|---|
| **Frobenius 范数** | $\|A\|_F=\sqrt{\sum_{i,j}a_{ij}^2}=\sqrt{\operatorname{tr}(A^TA)}$ | 把矩阵「拉直」成长向量取 $\ell_2$；逐元素的「总能量」 |
| **谱范数（spectral norm）** | $\|A\|_2=\max_{\|\mathbf x\|=1}\|A\mathbf x\|=\sigma_{\max}(A)$ | 矩阵能把单位向量最多拉多长 = 最大奇异值（第 8 章） |

Frobenius 范数其实就是矩阵空间上的「点积」诱导的范数：把 $m\times n$ 矩阵看成 $mn$ 维向量，内积 $\langle A,B\rangle_F=\operatorname{tr}(A^TB)=\sum_{ij}a_{ij}b_{ij}$，则 $\|A\|_F=\sqrt{\langle A,A\rangle_F}$。所以前面整套内积几何（正交、投影）对矩阵照样适用。

> **与你的研究的连接（Frobenius 范数 = 比较两个矩阵的标准尺子）**：折叠研究里你会反复需要「两个矩阵有多像」。(1) **比两张接触图 / 两个距离矩阵**：$\|D_{\text{pred}}-D_{\text{true}}\|_F$ 直接量化预测结构与真实结构在「残基对距离」上的总偏差，是结构相似性的一种全局度量。(2) **比两个协方差矩阵**（如不同 MD 轨迹片段的涨落是否一致）。(3) **低秩近似的误差**：Eckart–Young 定理说「截断 SVD 给出 Frobenius 范数下最优的低秩近似」（第 8 章），DCA / PCA 降维丢掉多少信息就是用 $\|A-A_k\|_F$ 衡量。(4) **谱范数**则衡量「最坏方向上的放大」——它正是数值稳定性（条件数 $\kappa=\sigma_{\max}/\sigma_{\min}$，第 9 章）和深度网络 Lipschitz 控制（谱归一化）的核心。记住：**逐元素总偏差用 Frobenius，最坏方向放大用谱范数。**

> **CS 读者陷阱（$\ell_2$ 与 $\ell_1$ 几何天差地别，别混用「距离」）**：很多人脑中「距离」只有欧氏一种，于是默认所有「范数 / 正则」都同质。错。$\ell_1$ 的单位球是**带尖角的菱形**，尖角戳在坐标轴上——这就是为什么 $\ell_1$ 正则（LASSO）会把解推到「某些坐标恰好为零」（稀疏），而 $\ell_2$ 正则（岭回归）的单位球是光滑的圆、不诱导稀疏。在折叠里这极有用：**稀疏精度矩阵估计（graphical lasso）** 就用 $\ell_1$ 罚把精度矩阵 $\Lambda$ 的非对角元逼到零，从而直接读出「哪些残基对是直接耦合（接触）」（《概率与统计》第 11 章、DCA）。$\ell_2$ 做不到这件事。**选范数 = 选你想要的解的几何性质，这是建模决策，不是审美。**

### 4.2.2 范数等价：在有限维里「殊途同归」

不同范数给出不同的「大小」数值，但在**有限维**空间里它们都「同阶」——存在常数 $0<c\le C$ 使

$$
c\,\|\mathbf v\|_a\le\|\mathbf v\|_b\le C\,\|\mathbf v\|_a\quad(\forall\mathbf v).
$$

例如 $\|\mathbf v\|_\infty\le\|\mathbf v\|_2\le\|\mathbf v\|_1\le\sqrt n\,\|\mathbf v\|_2\le n\,\|\mathbf v\|_\infty$。这叫**范数等价（equivalence of norms）**。它的实践意义：在有限维里，「收敛 / 有界 / 连续」这些拓扑性质**不依赖你选哪个范数**——一个序列在 $\ell_2$ 下收敛当且仅当在 $\ell_1$、$\ell_\infty$ 下也收敛。所以做理论分析时可以挑最顺手的范数。

> **诚实的边界（无穷维里范数不等价）**：范数等价**仅在有限维成立**。在无穷维空间（如函数空间）里，不同范数可以给出本质不同的拓扑——一个序列在 $L^2$ 下收敛但在 $L^\infty$ 下不收敛是家常便饭。这是泛函分析的入口。对折叠数值计算（永远是有限维的离散化）来说，范数等价是个安心的保证；但当你将来读到连续模型（如把蛋白构象当连续场、把分布当函数）时，要警惕「换范数可能换掉收敛性」。本课不深入，点到为止。

### 4.2.3 两个不等式：Cauchy–Schwarz 与三角

整套内积几何的「合法性证书」是 **Cauchy–Schwarz 不等式**：

$$
|\langle\mathbf u,\mathbf v\rangle|\le\|\mathbf u\|\,\|\mathbf v\|,\quad\text{等号}\iff\mathbf u,\mathbf v\text{ 线性相关（共线）}.
$$

它正是 §4.1.3 里「$\cos\theta\in[-1,1]$」的保证——没有它，夹角就没法定义。**一行证明骨架**：若 $\mathbf v=\mathbf 0$ 显然成立；否则考虑 $\mathbf u$ 减去它在 $\mathbf v$ 上的投影 $\mathbf w=\mathbf u-\dfrac{\langle\mathbf u,\mathbf v\rangle}{\langle\mathbf v,\mathbf v\rangle}\mathbf v$（§4.4 的投影公式），由正定性 $0\le\langle\mathbf w,\mathbf w\rangle=\langle\mathbf u,\mathbf u\rangle-\dfrac{\langle\mathbf u,\mathbf v\rangle^2}{\langle\mathbf v,\mathbf v\rangle}$，移项即得。等号当且仅当 $\mathbf w=\mathbf 0$，即 $\mathbf u$ 是 $\mathbf v$ 的倍数。这个证明本身就揭示了「不等式 = 投影后残差非负」的几何，下一节会重逢。

有了 Cauchy–Schwarz，**三角不等式**立刻跟来：

$$
\|\mathbf u+\mathbf v\|^2=\|\mathbf u\|^2+2\langle\mathbf u,\mathbf v\rangle+\|\mathbf v\|^2\le\|\mathbf u\|^2+2\|\mathbf u\|\|\mathbf v\|+\|\mathbf v\|^2=(\|\mathbf u\|+\|\mathbf v\|)^2,
$$

开方即 $\|\mathbf u+\mathbf v\|\le\|\mathbf u\|+\|\mathbf v\|$（「绕路不更短」/「两边之和大于第三边」）。注意中间那步**恰好**用 Cauchy–Schwarz 把 $\langle\mathbf u,\mathbf v\rangle$ 压上界——这就是为什么由内积导出的 $\|\cdot\|=\sqrt{\langle\cdot,\cdot\rangle}$ 一定是范数。

### 4.2.4 范数与内积的关系：极化恒等式

由内积能得范数（开根号），反过来——**能从范数恢复内积吗**？对来自内积的范数，能，靠**极化恒等式（polarization identity）**：

$$
\langle\mathbf u,\mathbf v\rangle=\tfrac14\big(\|\mathbf u+\mathbf v\|^2-\|\mathbf u-\mathbf v\|^2\big).
$$

（展开右边即验证。）这说明「内积」和「它诱导的范数」携带完全相同的信息——内积不比范数多、也不少。但**关键的反面**：不是每个范数都来自某个内积。一个范数来自内积，当且仅当它满足**平行四边形恒等式（parallelogram law）** $\|\mathbf u+\mathbf v\|^2+\|\mathbf u-\mathbf v\|^2=2\|\mathbf u\|^2+2\|\mathbf v\|^2$。$\ell_1$ 和 $\ell_\infty$ 都**不**满足它（练习 3），所以它们没有对应的内积——也就没有「角度」「正交投影」这些概念。**这正是 $\ell_2$ 在线性代数里地位独尊的原因**：只有它带着完整的欧氏几何（角度、正交、投影、勾股），所以最小二乘、PCA、Kabsch 全用 $\ell_2$。

> **给数学 / CS 读者的视角（$\ell_2$ = 唯一「自带几何」的范数）**：把范数想成「只会量长度的瘸腿度量」，内积则是「既量长度又量角度的全功能度量」。极化恒等式说：在 $\ell_2$ 世界里，量长度的能力**等价于**量角度的能力（知道所有长度就能算出所有角度）。而 $\ell_1/\ell_\infty$ 这些范数是「真瘸腿」——它们能量长度（适合做正则、做鲁棒优化），但量不出角度、做不了正交投影。所以当你的任务需要「投影 / 最优逼近 / 主方向」时，必须站在 $\ell_2$（内积）的地基上；需要「稀疏 / 鲁棒」时才用 $\ell_1/\ell_\infty$。**先问任务要不要角度，再选范数。**

> **要点（§4.2）**：范数 = 正定 + 齐次 + 三角。常见向量范数 $\ell_2$（欧氏、唯一来自内积）、$\ell_1$（稀疏）、$\ell_\infty$（最坏分量）；矩阵范数 Frobenius（逐元素总能量、比较两矩阵 / 两接触图）、谱范数（最大奇异值、最坏放大 / 条件数）。**Cauchy–Schwarz** $|\langle\mathbf u,\mathbf v\rangle|\le\|\mathbf u\|\|\mathbf v\|$ 保证夹角有定义、并推出**三角不等式**。**极化恒等式**从范数恢复内积，但只有满足平行四边形律的范数（即 $\ell_2$）才来自内积——这是 $\ell_2$ 独享完整欧氏几何的原因。有限维**范数等价**（无穷维不然）。

---

## 4.3 正交、正交基、正交补：在好坐标下一切变简单

### 4.3.1 正交与正交基

两个向量 $\mathbf u\perp\mathbf v$ 当 $\langle\mathbf u,\mathbf v\rangle=0$。一组两两正交的非零向量自动**线性无关**（一行证明：若 $\sum_i c_i\mathbf q_i=\mathbf 0$，与 $\mathbf q_j$ 取内积，正交性把别的项杀光，剩 $c_j\langle\mathbf q_j,\mathbf q_j\rangle=0$，由正定 $c_j=0$）。所以正交向量组「天生就是好基的料」。

若一组基 $\{\mathbf q_1,\dots,\mathbf q_n\}$ 不仅两两正交、且每个都是**单位长度**（$\|\mathbf q_i\|=1$），就叫**标准正交基 / 规范正交基（orthonormal basis）**，简洁地写成

$$
\langle\mathbf q_i,\mathbf q_j\rangle=\delta_{ij}=\begin{cases}1,&i=j\\0,&i\ne j.\end{cases}
$$

标准基 $\{\mathbf e_1,\dots,\mathbf e_n\}$ 是最熟悉的标准正交基。把标准正交基的向量并排成矩阵 $Q=[\mathbf q_1\,\cdots\,\mathbf q_n]$，上式紧凑写作 $Q^TQ=I$——这正是**正交矩阵**的定义（§4.6 主角）。

### 4.3.2 坐标 = 内积：标准正交基的超能力

第 2 章说「选基 = 给抽象对象配坐标」，但一般基下求坐标要解线性方程组（求逆）。**标准正交基的杀手锏**：坐标直接由内积给出，不用解方程。设 $\{\mathbf q_i\}$ 是标准正交基，任意 $\mathbf v=\sum_j c_j\mathbf q_j$，与 $\mathbf q_i$ 取内积：

$$
\langle\mathbf v,\mathbf q_i\rangle=\sum_j c_j\langle\mathbf q_j,\mathbf q_i\rangle=c_i\quad\Rightarrow\quad \boxed{\ \mathbf v=\sum_i\langle\mathbf v,\mathbf q_i\rangle\,\mathbf q_i.\ }
$$

**第 $i$ 个坐标 = $\mathbf v$ 与第 $i$ 个基向量的内积**——一次内积就读出一个坐标，无需求逆、无需解方程。这是标准正交基「好用到不像话」的核心原因，也是傅里叶分析、PCA 投影、信号处理里「展开系数 = 内积」的统一来源。

由此立得两条恒等式（取 $\mathbf u=\sum a_i\mathbf q_i$、$\mathbf v=\sum b_i\mathbf q_i$）：

$$
\langle\mathbf u,\mathbf v\rangle=\sum_i a_ib_i\quad(\textbf{Parseval}),\qquad \|\mathbf v\|^2=\sum_i\langle\mathbf v,\mathbf q_i\rangle^2\quad(\textbf{能量守恒}).
$$

**Parseval 恒等式**说：在标准正交基下，抽象的内积退化成坐标的普通点积——换到正交坐标系不改变内积、长度、角度（几何被完整保留）。**能量守恒**说：向量的长度平方 = 各方向投影的平方和（勾股定理的 $n$ 维、坐标版）。

> **与你的研究的连接（正交基 = 主成分基 / 简正模式基，「换到本征坐标」一切解耦）**：折叠数据分析里你会一次次「换到一组特殊的标准正交基」，换完之后复杂的耦合变成独立的一维问题——这就是 Parseval 的威力。(1) **MD 轨迹 PCA / 本质动力学（essential dynamics）**：协方差矩阵的特征向量构成一组标准正交基（主成分），把高维构象涨落投影上去，每个主成分坐标 = 构象在该主方向上的内积，前几个就抓住了主要的集体运动（第 8 章）。(2) **简正模式分析（NMA）/ 弹性网络模型（ENM）**：Hessian（对称矩阵）的特征向量也是一组标准正交基（简正模式），低频模式 = 集体的呼吸 / 铰链运动（第 7 章）。(3) 这两处都用到「对称矩阵的特征向量两两正交」（谱定理，第 7 章），所以「坐标 = 内积」让你能廉价地把任意构象分解到模式 / 主成分上。**记住这个模式：找到对的正交基 → 投影（取内积）→ 在解耦坐标里分析。这是折叠数据分析的通用套路。**

### 4.3.3 正交补与勾股定理

给定子空间 $W\subseteq\mathbb R^n$，它的**正交补（orthogonal complement）** 是「与 $W$ 中一切向量都正交」的向量全体：

$$
W^\perp:=\{\mathbf v\in\mathbb R^n:\langle\mathbf v,\mathbf w\rangle=0\ \forall\mathbf w\in W\}.
$$

$W^\perp$ 也是子空间，且与 $W$ 互补到整个空间：$\dim W+\dim W^\perp=n$，$W\cap W^\perp=\{\mathbf 0\}$，$W\oplus W^\perp=\mathbb R^n$。意思是**任何向量都能唯一分解成「$W$ 里的一块」加「$W^\perp$ 里的一块」**，这正是下一节正交投影的舞台。第 2 章 §2.4 的四个基本子空间其实就是两对正交补：$\operatorname{null}(A)=\operatorname{row}(A)^\perp$、$\operatorname{null}(A^T)=\operatorname{col}(A)^\perp$（行空间 ⊥ 零空间、列空间 ⊥ 左零空间）——这是线性代数最优美的对偶之一。

当 $\mathbf u\perp\mathbf v$ 时，三角不等式取「直角等号」，即 **$n$ 维勾股定理（Pythagorean theorem）**：

$$
\mathbf u\perp\mathbf v\ \Rightarrow\ \|\mathbf u+\mathbf v\|^2=\|\mathbf u\|^2+\|\mathbf v\|^2
$$

（因为展开式里的交叉项 $2\langle\mathbf u,\mathbf v\rangle=0$）。这条「正交则长度平方可加」是 §4.4 「投影 + 残差」分解和 §4.7 最小二乘「为什么投影最近」的几何引擎，务必记牢。

> **要点（§4.3）**：正交向量组自动线性无关。**标准正交基**（$\langle\mathbf q_i,\mathbf q_j\rangle=\delta_{ij}$，$Q^TQ=I$）下**坐标 = 内积**（$\mathbf v=\sum_i\langle\mathbf v,\mathbf q_i\rangle\mathbf q_i$），无需求逆；Parseval（内积变坐标点积）与能量守恒（长度平方 = 投影平方和）随之而来——这是 PCA / 简正模式 / 傅里叶「换到正交本征坐标后解耦」的统一根基。**正交补** $W^\perp$ 把空间正交分解为 $W\oplus W^\perp$，对应四个基本子空间的两对正交关系。正交则**勾股定理** $\|\mathbf u+\mathbf v\|^2=\|\mathbf u\|^2+\|\mathbf v\|^2$ 成立——这是投影最优性的引擎。

---

## 4.4 正交投影：把向量「拍」到子空间上的最近点

这是全章的几何高潮，也是最小二乘、PCA、Kabsch 的共同内核。问题是：给一个向量 $\mathbf b$ 和一个子空间 $W$，**$W$ 里哪个向量离 $\mathbf b$ 最近**？答案是 $\mathbf b$ 在 $W$ 上的**正交投影（orthogonal projection）**——把 $\mathbf b$ 笔直「拍」到 $W$ 上的影子。

> **动手提示：打开 `04_projection_least_squares.html` 边读边玩。** 这个页面让你拖动一个点 $\mathbf b$ 和一条线 / 一个子空间，实时看到投影点（影子）、残差（垂线）、以及「为什么垂足是最近点」。本节每讲一个公式，建议你在页面里找到对应的几何对象：投影向量、残差向量、它俩的直角、残差随 $\mathbf b$ 移动如何变化。投影是「看一眼就懂、自己推一遍才真懂」的概念。

### 4.4.1 投影到一条直线

先看最简单情形：$W=\operatorname{span}\{\mathbf a\}$ 是一条过原点的直线。$\mathbf b$ 在它上的投影是 $\hat{\mathbf b}=c\,\mathbf a$（某个标量倍），要求**残差 $\mathbf b-\hat{\mathbf b}$ 垂直于直线**（这是「最近」的几何条件，下面 §4.4.3 证）：

$$
\langle\mathbf b-c\mathbf a,\ \mathbf a\rangle=0\ \Rightarrow\ c=\frac{\langle\mathbf a,\mathbf b\rangle}{\langle\mathbf a,\mathbf a\rangle}=\frac{\mathbf a^T\mathbf b}{\mathbf a^T\mathbf a}.
$$

于是

$$
\hat{\mathbf b}=\frac{\mathbf a^T\mathbf b}{\mathbf a^T\mathbf a}\,\mathbf a=\underbrace{\frac{\mathbf a\mathbf a^T}{\mathbf a^T\mathbf a}}_{=:P}\,\mathbf b.
$$

注意 $P=\dfrac{\mathbf a\mathbf a^T}{\mathbf a^T\mathbf a}$ 是个矩阵（**外积 $\mathbf a\mathbf a^T$ = 秩 1 矩阵**，第 3 章 §3.5.1 的种子在此结果），它把任意 $\mathbf b$ 映到它在 $\mathbf a$ 方向的影子——这就是**投影矩阵（projection matrix）**。若 $\mathbf a$ 已归一化（$\|\mathbf a\|=1$），$P=\mathbf a\mathbf a^T$，$\hat{\mathbf b}=(\mathbf a^T\mathbf b)\mathbf a$——「坐标（内积）乘基向量」，正是 §4.3.2 那条公式的单项。

### 4.4.2 投影到子空间：投影矩阵 $P=A(A^TA)^{-1}A^T$

现在 $W=\operatorname{col}(A)$ 是矩阵 $A$（$m\times n$，列线性无关）的列空间。$\mathbf b$ 在 $W$ 上的投影 $\hat{\mathbf b}=A\hat{\mathbf x}$ 是列的某个线性组合（系数 $\hat{\mathbf x}$ 待定）。**最优性条件**：残差 $\mathbf r=\mathbf b-A\hat{\mathbf x}$ 必须**垂直于整个列空间**，即垂直于 $A$ 的每一列：

$$
A^T(\mathbf b-A\hat{\mathbf x})=\mathbf 0\ \Longrightarrow\ \boxed{\,A^TA\,\hat{\mathbf x}=A^T\mathbf b\,}\quad(\textbf{正规方程, normal equations}).
$$

「残差 ⊥ 列空间」这一行就是**正规方程的几何来源**——它说残差落在列空间的正交补（即左零空间 $\operatorname{null}(A^T)$）里。当 $A$ 列无关时 $A^TA$ 可逆（练习 5），解出 $\hat{\mathbf x}=(A^TA)^{-1}A^T\mathbf b$，于是投影向量与投影矩阵：

$$
\hat{\mathbf b}=A\hat{\mathbf x}=\underbrace{A(A^TA)^{-1}A^T}_{=:P}\,\mathbf b,\qquad P=A(A^TA)^{-1}A^T.
$$

ASCII 示意（投影把 $\mathbf b$ 分解成「列空间里的影子 $\hat{\mathbf b}$」+「正交残差 $\mathbf r$」）：

```
            b
           /|
          / |  r = b - b̂   （残差，⊥ 列空间）
         /  |
        /   |
 ------●----●--------  col(A)  （列空间 W）
       0    b̂ = P b    （影子，= W 里离 b 最近的点）
```

### 4.4.3 投影矩阵的性质与「为什么是最近点」

$P=A(A^TA)^{-1}A^T$ 有两条标志性性质，记牢它们就能一眼认出投影矩阵：

- **幂等（idempotent）**：$P^2=P$。几何上「影子的影子还是影子」——已经在 $W$ 上的向量再投影不动。代数验证：$P^2=A(A^TA)^{-1}\underbrace{A^TA(A^TA)^{-1}}_{=I}A^T=A(A^TA)^{-1}A^T=P$。
- **对称（symmetric）**：$P^T=P$（直接转置验证，用 $(A^TA)^{-1}$ 对称）。

反过来，**任何满足 $P^2=P$ 且 $P^T=P$ 的矩阵都是某个子空间上的正交投影**（这是投影矩阵的代数刻画）。互补投影 $I-P$ 投影到正交补 $W^\perp$（残差 $\mathbf r=(I-P)\mathbf b$），它也幂等对称，且 $P(I-P)=0$。

**为什么投影点是最近点**？设 $\mathbf w\in W$ 是 $W$ 里任意一点，把 $\mathbf b-\mathbf w$ 拆成 $(\mathbf b-\hat{\mathbf b})+(\hat{\mathbf b}-\mathbf w)$。前者 = 残差 $\mathbf r\perp W$，后者 $\hat{\mathbf b}-\mathbf w\in W$（两个 $W$ 中点之差仍在 $W$），二者正交，由**勾股定理**（§4.3.3）：

$$
\|\mathbf b-\mathbf w\|^2=\underbrace{\|\mathbf b-\hat{\mathbf b}\|^2}_{\text{固定}}+\underbrace{\|\hat{\mathbf b}-\mathbf w\|^2}_{\ge0}\ \ge\ \|\mathbf b-\hat{\mathbf b}\|^2,
$$

等号当且仅当 $\mathbf w=\hat{\mathbf b}$。**所以投影点 $\hat{\mathbf b}$ 是 $W$ 中唯一离 $\mathbf b$ 最近的点**——这就是「正交 = 最优逼近」的完整证明，也是下一步最小二乘的全部内容。勾股定理在这里是真正的主角。

> **诚实的边界（$A$ 列相关时 $A^TA$ 不可逆，投影仍在但公式要换）**：上面的公式 $P=A(A^TA)^{-1}A^T$ 要求 $A$ **列线性无关**（否则 $A^TA$ 奇异、求逆爆炸）。但投影本身永远存在且唯一（任何向量到子空间的最近点是良定义的）——出问题的只是「用 $\hat{\mathbf x}$ 表示」这一步：列相关时表示 $\hat{\mathbf b}=A\hat{\mathbf x}$ 的系数 $\hat{\mathbf x}$ 不唯一。正确做法不是去求 $(A^TA)^{-1}$（数值上还会平方条件数、很危险，第 9 章），而是：(1) 用 $A$ 列空间的一组**标准正交基** $Q$，则 $P=QQ^T$（练习 6，公式更简洁也更稳）；(2) 用 QR 分解（第 5 章）；(3) 用 SVD / 伪逆 $\hat{\mathbf x}=A^+\mathbf b$（第 8 章，对任何 $A$ 都给出最小范数解）。**实践纪律：教科书用 $A(A^TA)^{-1}A^T$ 讲清几何，真算时用 QR / SVD，永远别显式构造 $A^TA$ 再求逆。**

> **与你的研究的连接（投影 = 降维 / 去噪 / 约束到合法子空间）**：投影在折叠里到处都是。(1) **PCA 降维 = 投影到主成分子空间**：把高维构象向量投影到前 $k$ 个主成分张成的子空间，$P=QQ^T$（$Q$ = 前 $k$ 个主成分），丢掉的残差 $\mathbf r=(I-P)\mathbf b$ 就是被舍弃的「次要涨落 / 噪声」（第 8 章）。(2) **简正模式分析里投影掉刚体自由度**：Hessian 有 6 个零模式（3 平移 + 3 旋转，第 3 章 §3.6.2），分析内部运动前要先用 $I-P$ 把构象投影到「去掉刚体运动」的子空间。(3) **约束满足**：把一个不满足某线性约束的构象 / 力，投影到满足约束的子空间（如 SHAKE / 约束动力学把速度投影到与键长约束正交的方向）。(4) **去除均值 = 投影到「和为零」子空间**：PCA / 结构叠合第一步「中心化」（减去质心）就是投影掉全 1 方向。**凡是「把数据限制到一个合法 / 主要的低维子空间」，背后都是正交投影。**

> **要点（§4.4）**：$\mathbf b$ 到子空间 $W$ 的最近点 = **正交投影** $\hat{\mathbf b}=P\mathbf b$，残差 $\mathbf b-\hat{\mathbf b}\perp W$（这就是**正规方程** $A^TA\hat{\mathbf x}=A^T\mathbf b$ 的几何来源）。投影到直线 $P=\mathbf a\mathbf a^T/(\mathbf a^T\mathbf a)$（秩 1 外积）；投影到列空间 $P=A(A^TA)^{-1}A^T$。投影矩阵**幂等 $P^2=P$ + 对称 $P^T=P$**（互补投影 $I-P$ 投到 $W^\perp$）。**勾股定理证明投影点最近**。$A$ 列相关 / 数值上要换 $P=QQ^T$ 或 QR / SVD。投影 = 折叠里的降维 / 去噪 / 约束到合法子空间。务必玩 `04_projection_least_squares.html`。

---

## 4.5 Gram–Schmidt 正交化：把任意基「掰」成正交基

§4.3 反复说「标准正交基好用到不像话」（坐标 = 内积、投影 = $QQ^T$、几何被保留）。但现实给你的基往往不正交（比如随便几个数据向量、几个约束方向）。**Gram–Schmidt 正交化（Gram–Schmidt process）** 就是一台「掰直机」：输入任意一组线性无关向量 $\{\mathbf a_1,\dots,\mathbf a_n\}$，输出一组张成同一空间的标准正交基 $\{\mathbf q_1,\dots,\mathbf q_n\}$。

### 4.5.1 算法：逐个减掉已有方向上的投影

核心思想一句话：**每来一个新向量，就减掉它在「已经正交化好的那些方向」上的投影，剩下的残差必然与它们都正交**——这直接用了 §4.4 的投影和「残差 ⊥ 子空间」。逐步进行：

$$
\begin{aligned}
\mathbf u_1&=\mathbf a_1, &&\mathbf q_1=\mathbf u_1/\|\mathbf u_1\|,\\
\mathbf u_2&=\mathbf a_2-\langle\mathbf a_2,\mathbf q_1\rangle\mathbf q_1, &&\mathbf q_2=\mathbf u_2/\|\mathbf u_2\|,\\
\mathbf u_3&=\mathbf a_3-\langle\mathbf a_3,\mathbf q_1\rangle\mathbf q_1-\langle\mathbf a_3,\mathbf q_2\rangle\mathbf q_2, &&\mathbf q_3=\mathbf u_3/\|\mathbf u_3\|,\\
&\ \ \vdots && \\
\mathbf u_k&=\mathbf a_k-\sum_{j=1}^{k-1}\langle\mathbf a_k,\mathbf q_j\rangle\mathbf q_j, &&\mathbf q_k=\mathbf u_k/\|\mathbf u_k\|.
\end{aligned}
$$

第 $k$ 步：$\mathbf u_k$ = $\mathbf a_k$ 减去它在前 $k-1$ 个 $\mathbf q_j$ 张成的子空间上的投影（$\sum_j\langle\mathbf a_k,\mathbf q_j\rangle\mathbf q_j$ 正是 §4.3.2 的投影公式），残差 $\mathbf u_k$ 自动与所有 $\mathbf q_j$（$j<k$）正交，再归一化即得 $\mathbf q_k$。每一步 $\operatorname{span}\{\mathbf q_1,\dots,\mathbf q_k\}=\operatorname{span}\{\mathbf a_1,\dots,\mathbf a_k\}$（张成的空间始终不变，只是换了正交的「描述」）。

### 4.5.2 Gram–Schmidt 就是 QR 分解（预告第 5 章）

把上面的关系反过来读：每个原向量 $\mathbf a_k$ 都是前 $k$ 个 $\mathbf q$ 的线性组合，组合系数恰是那些内积。整理成矩阵就是

$$
A=QR,\qquad Q=[\mathbf q_1\,\cdots\,\mathbf q_n]\ (Q^TQ=I),\quad R\ \text{上三角},
$$

其中 $R_{jk}=\langle\mathbf a_k,\mathbf q_j\rangle$（$j\le k$），对角元 $R_{kk}=\|\mathbf u_k\|$。这就是大名鼎鼎的 **QR 分解（QR decomposition）**：「把矩阵的列正交化」= 「分解成正交矩阵 $Q$ 乘上三角 $R$」。它是数值求解最小二乘、特征值（QR 算法）的主力工具，第 5 章会正式展开。这里你只需看到：**Gram–Schmidt 和 QR 是同一件事的两种说法**——前者是过程，后者是结果。

> **诚实的边界（经典 Gram–Schmidt 数值不稳，实战用 MGS / Householder）**：上面这个「经典」Gram–Schmidt（classical GS, CGS）在纸上完美，但在有限精度浮点里**数值不稳定**——当向量接近线性相关时，舍入误差会让算出的 $\mathbf q_i$ 逐渐**失去正交性**（$Q^TQ$ 偏离 $I$ 越来越远）。两个修正：(1) **改进的 Gram–Schmidt（modified Gram–Schmidt, MGS）**：把「一次性减掉对所有 $\mathbf q_j$ 的投影」改成「逐个减、每减一个就用更新后的残差算下一个投影」，数学上等价、数值上稳得多。(2) 真正工业级的 QR 用 **Householder 反射**或 Givens 旋转（用正交变换逐步把 $A$ 「打」成上三角），正交性由构造保证、最稳。这条「同一公式、CGS 不稳 / MGS 稳 / Householder 最稳」的故事是第 9 章数值线性代数的经典案例，提醒你：**数学等价 ≠ 数值等价**，写科学计算代码时这条会反复咬你。**实践纪律：要正交基 / QR 时调库（`np.linalg.qr`，内部用 Householder），别手写经典 GS。**

> **给数学 / CS 读者的视角（Gram–Schmidt = 在线「去相关」流式算法）**：把它读成一个流式（streaming）算法——向量一个个来，每来一个就「减去它能被已见向量解释的部分」，只保留**新增的、正交的信息**，再归一化入库。这和增量构造一组不相关特征的思路（如逐步回归里去掉已解释方差、Cholesky 的逐列消元、甚至某些注意力 / 白化操作）同源：**正交化 = 去相关 = 提取增量信息**。复杂度 $O(mn^2)$（$m$ 维、$n$ 个向量），与一次 QR 同阶。理解了这一点，你看 QR、Cholesky、白化（whitening）时会发现它们都是「换一组不相关坐标」的近亲。

> **要点（§4.5）**：Gram–Schmidt 把任意线性无关组**逐个减去在已正交化方向上的投影、再归一化**，得到张成同一空间的标准正交基。它等价于 **QR 分解** $A=QR$（$Q$ 正交、$R$ 上三角，$R_{jk}$ = 内积、$R_{kk}$ = 残差长度），是第 5 章的入口。经典 GS **数值不稳**（会丢正交性），实战用 MGS / Householder（第 9 章）——**数学等价 ≠ 数值等价**。它本质是「去相关 / 提取增量信息」的流式算法。

---

## 4.6 正交矩阵：保内积 = 保距 = 保角 = 旋转与反射

§4.3 里把标准正交基并排得到 $Q^TQ=I$。本节把这种矩阵正名，并揭示它的几何身份——它正是**保持一切几何的变换**，直通第 10 章的刚体旋转。这是暗线三（三维几何与等变性）在本章的接口。

### 4.6.1 定义与等价刻画

一个方阵 $Q\in\mathbb R^{n\times n}$ 叫**正交矩阵（orthogonal matrix）**，若

$$
Q^TQ=QQ^T=I\quad\Longleftrightarrow\quad Q^{-1}=Q^T.
$$

「逆 = 转置」是它最省事的性质（求逆免费！）。$Q^TQ=I$ 的逐元素含义正是 §4.3.1 的 $\langle\mathbf q_i,\mathbf q_j\rangle=\delta_{ij}$——**$Q$ 的列构成一组标准正交基**（行也是）。下面三件事**两两等价**，是正交矩阵的完整画像：

1. $Q^TQ=I$（列标准正交）；
2. **保内积**：$\langle Q\mathbf u,Q\mathbf v\rangle=\langle\mathbf u,\mathbf v\rangle$（变换后内积不变）；
3. **保长度（等距, isometry）**：$\|Q\mathbf v\|=\|\mathbf v\|$（变换后长度不变）。

证 1$\Rightarrow$2：$\langle Q\mathbf u,Q\mathbf v\rangle=(Q\mathbf u)^T(Q\mathbf v)=\mathbf u^TQ^TQ\mathbf v=\mathbf u^T\mathbf v=\langle\mathbf u,\mathbf v\rangle$。2$\Rightarrow$3 取 $\mathbf u=\mathbf v$；3$\Rightarrow$2 由极化恒等式（§4.2.4，长度可恢复内积，故长度不变则内积不变）；最后 2$\Rightarrow$1 把 $\mathbf u=\mathbf e_i,\mathbf v=\mathbf e_j$ 代入即得 $(Q^TQ)_{ij}=\delta_{ij}$。所以**「保距」「保内积」「列正交」是同一件事**——这条等价链请刻进脑子。

由「保内积 + 保长度」自然**也保角度**（$\cos\theta=\langle Q\mathbf u,Q\mathbf v\rangle/(\|Q\mathbf u\|\|Q\mathbf v\|)$ 分子分母都不变，夹角原封不动）——但要小心：**单凭「保角度」并不能反推出正交矩阵**。一个均匀缩放 $Q=cI$（$c\ne\pm1$）会保持任意两向量的夹角（它是「保角 / 共形（conformal）」变换），却把所有长度乘了 $|c|$，根本不是等距。所以「保角」严格**弱于**「保距」：保角度的方阵恰是「正交矩阵乘一个正标量」（相似变换 / 缩放旋转，$Q=cU$，$U$ 正交）。本节后续说正交矩阵「保角」，指的是它作为「保距」的**推论**顺带保角，而不是反过来用保角去定义它。这点在折叠里要紧：刚体叠合要的是**保距**（键长、原子间距离一个都不能变），仅仅「保角」是不够的——缩放会改变绝对尺度。

### 4.6.2 行列式 $\pm1$：旋转与反射

由 $Q^TQ=I$ 取行列式：$\det(Q^T)\det(Q)=(\det Q)^2=\det I=1$，故 $\det Q=\pm1$。这把正交矩阵分成两类：

| $\det Q$ | 名称 | 几何 | 群 |
|---|---|---|---|
| $+1$ | **旋转 / 真旋转（proper rotation）** | 保距 + **保定向**（不翻面） | 特殊正交群 $\mathrm{SO}(n)$ |
| $-1$ | **反射 / 瑕旋转（improper）** | 保距但**翻定向**（镜像、左右手互换） | $\mathrm{O}(n)\setminus\mathrm{SO}(n)$ |

全体正交矩阵构成**正交群 $\mathrm{O}(n)$**，行列式为 $+1$ 的子群是**特殊正交群 $\mathrm{SO}(n)$**（旋转群）。第 3 章 §3.2.2 的二维旋转 $R(\theta)$（$\det=1$）属于 $\mathrm{SO}(2)$，反射（$\det=-1$）属于 $\mathrm{O}(2)$ 但不属于 $\mathrm{SO}(2)$。三维的 $\mathrm{SO}(3)$ 就是刚体旋转的全部，是第 10 章和 AlphaFold frame、等变网络的核心舞台。

> **与你的研究的连接（正交矩阵 = 刚体变换 = 结构叠合唯一合法的「动作」）**：这是本章对折叠最重要的一句话。蛋白结构是 $N\times3$ 坐标矩阵，但**坐标依赖于你把分子摆在哪个参考系**——同一个结构平移、旋转后坐标全变，可它还是同一个结构。所以「比较两个结构」前必须先**叠合（superpose）**，而叠合允许的操作**只能是保距的刚体变换（平移 + 旋转）**，即 $\mathrm{SO}(3)$ 里的旋转加一个平移——因为只有保距变换不改变分子的内部几何（键长、键角、原子间距离全不变，§4.6.1 的「保距」）。(1) **RMSD / Kabsch 算法**：找最优旋转 $R\in\mathrm{SO}(3)$ 使叠合后坐标差的 $\ell_2$ 范数最小。严格说 $\mathrm{RMSD}=\sqrt{\frac1N\sum_i\|\mathbf x_i-\mathbf y_i\|^2}=\frac{1}{\sqrt N}\,\|\operatorname{vec}(X-Y)\|_2$——它就是「坐标差 $\ell_2$ 范数除以 $\sqrt N$」（按原子数归一，让不同大小的分子可比），而那个 $1/\sqrt N$ 是常数，所以**最小化 RMSD 与最小化 $\ell_2$ 范数完全等价**。这是个「正交 Procrustes 问题」，答案由 SVD 给出（第 10 章），约束 $R^TR=I$、$\det R=+1$ 正是为了保证它是真旋转而非镜像。(2) **为什么排除反射（$\det=-1$）**：反射会把分子变成它的**镜像（手性翻转）**——而氨基酸是手性的（天然为 L 型），镜像是另一种分子！所以叠合时必须强制 $\det R=+1$，否则会把一个结构错配到它不存在的镜像上（练习 7）。(3) **AlphaFold 的 frame**：结构模块给每个残基一个 frame = 一个 $\mathrm{SO}(3)$ 旋转 + 平移（一个 $\mathrm{SE}(3)$ 元素，第 10 章），IPA（invariant point attention）对全局刚体变换不变，靠的就是正交矩阵「保距」的代数。(4) **MSA / pair 表示**里也用正交变换做特征旋转。**一句话：在折叠里看到「正交」「保距」「叠合」「frame」「等变」，背后都是同一个 $Q^TQ=I$。**

> **给数学 / CS 读者的视角（正交矩阵 = 无损、稳定的「换坐标」）**：正交变换是数值计算里的「乖宝宝」。因为 $\|Q\mathbf v\|=\|\mathbf v\|$，它**既不放大也不缩小**任何向量——所以它**不放大误差**（条件数 $\kappa(Q)=1$，完美良态，第 9 章），是数值算法（QR、SVD、Householder）爱用它当「工作变换」的根本原因：用正交变换一步步化简矩阵，误差不累积。对比一般可逆变换会扭曲长度、放大噪声。把它类比成「无损的、可逆的、不失真的坐标旋转」——存数据、传数据、迭代计算都安全。这也是深度网络里「正交初始化 / 正交约束权重」能稳住梯度（既不爆炸也不消失）的原因。**正交 = 保距 = 数值上最安全的变换。**

> **CS 读者陷阱（$Q^TQ=I$ 对长方矩阵只给「列正交」，不给 $QQ^T=I$）**：当 $Q$ 是 $m\times n$ 的**长方**矩阵（$m>n$，列正交规范），$Q^TQ=I_n$ 仍成立（列两两正交单位长），但 $QQ^T\ne I_m$！后者是「投影到列空间」的投影矩阵（§4.4，$P=QQ^T$，幂等不满秩），不是单位阵。只有**方阵**正交矩阵才同时有 $Q^TQ=QQ^T=I$。混淆这点会在写投影 / QR 代码时埋 bug：「列正交的长方 $Q$」≠「正交方阵」。判据：先看形状，方阵才谈「正交矩阵」，长方只谈「列正交规范」。

> **要点（§4.6）**：正交矩阵 $Q^TQ=QQ^T=I$（$Q^{-1}=Q^T$），其列 = 标准正交基；它**保内积 = 保长度 = 列正交**（三者等价，由极化恒等式串起），并**顺带保角度**（但反过来「保角」严格更弱——缩放 $cI$ 也保角却非等距，故不能用保角定义正交矩阵）。$\det Q=\pm1$：$+1$ 是**旋转**（$\mathrm{SO}(n)$，保定向）、$-1$ 是**反射**（翻定向 = 镜像 / 手性翻转）。这是折叠里**刚体叠合 / RMSD / Kabsch / AlphaFold frame / 等变**的代数核心——叠合只能用保距的旋转，且须排除反射（手性）。正交变换**条件数 = 1、不放大误差**，是数值算法的「乖宝宝」。长方 $Q$ 只有 $Q^TQ=I$（$QQ^T$ 是投影矩阵），别和正交方阵混。

---

## 4.7 最小二乘的几何：当 $A\mathbf x=\mathbf b$ 无解，求最近的近似解

终于到了把前面所有工具拧成一股绳的地方。**最小二乘（least squares）** 是折叠与一切数据科学里最基础的「最优」问题，而它的灵魂，就是 §4.4 的正交投影。

### 4.7.1 问题：超定方程组「无解」怎么办

实际中你常遇到方程比未知数多的**超定（overdetermined）** 系统 $A\mathbf x=\mathbf b$（$A$ 是 $m\times n$、$m>n$，第 3 章 §3.6.3 的高瘦矩阵）。比如你有 100 个测量点（$m=100$），想拟合一条只有 2 个参数的直线（$n=2$）——除非所有点恰好共线，否则**没有任何 $\mathbf x$ 能让所有方程同时成立**，即 $\mathbf b\notin\operatorname{col}(A)$，方程**无解**。

但「无解」不等于「没办法」。我们退一步问：**哪个 $\mathbf x$ 让等式最接近成立**？用 $\ell_2$ 范数度量「不成立的程度」（残差大小），就得到最小二乘问题：

$$
\hat{\mathbf x}=\arg\min_{\mathbf x}\ \|A\mathbf x-\mathbf b\|^2.
$$

### 4.7.2 几何答案：把 $\mathbf b$ 投影到列空间

这正是 §4.4 的问题换了件衣服！$A\mathbf x$ 遍历整个列空间 $\operatorname{col}(A)$，我们要在列空间里找离 $\mathbf b$ 最近的点——**就是 $\mathbf b$ 在 $\operatorname{col}(A)$ 上的正交投影 $\hat{\mathbf b}=P\mathbf b$**。对应的最优 $\hat{\mathbf x}$ 满足 $A\hat{\mathbf x}=\hat{\mathbf b}$，由「残差 ⊥ 列空间」得到**正规方程**：

$$
A^TA\,\hat{\mathbf x}=A^T\mathbf b\quad\Rightarrow\quad \hat{\mathbf x}=(A^TA)^{-1}A^T\mathbf b\ (\text{当 }A\text{ 列无关}).
$$

ASCII 把这件事画死（最小二乘 = 投影 = 勾股，三位一体）：

```
        b  ●  （真实数据，不在列空间里，方程无解）
          /|
         / |   残差 r = b - A x̂   （我们最小化的 ||r||，且 r ⊥ col(A)）
        /  |
 ------●---●----------  col(A) = { A x : 所有 x }  （模型能表达的一切）
       0   A x̂ = b̂          （最优拟合 = b 在模型空间的影子）
```

「最小二乘解」「正交投影」「正规方程」「勾股定理」在这里是**同一件事的四种语言**：最小化 $\|A\mathbf x-\mathbf b\|^2$ ⟺ 把 $\mathbf b$ 投到列空间 ⟺ 残差正交于列空间 ⟺ 勾股保证投影点最近。你在 `04_projection_least_squares.html` 里拖动数据点时，看到的「拟合线随点移动、残差始终保持垂直」就是这套几何在动。

### 4.7.3 与第 5、8 章的衔接：别用正规方程算

正规方程 $\hat{\mathbf x}=(A^TA)^{-1}A^T\mathbf b$ 漂亮，但**数值上是坏主意**：构造 $A^TA$ 会把条件数**平方**（$\kappa(A^TA)=\kappa(A)^2$，第 9 章），病态被放大，精度崩坏。所以工业实践**绝不**显式算 $A^TA$ 再求逆，而是：

- **QR 分解**（第 5 章）：$A=QR\Rightarrow R\hat{\mathbf x}=Q^T\mathbf b$，上三角回代即得，稳定且快——这是最小二乘的标准解法（`np.linalg.lstsq` 默认走类似路线）。
- **SVD / 伪逆**（第 8 章）：$\hat{\mathbf x}=A^+\mathbf b$，对任何 $A$（包括列相关、欠定）都给出（最小范数）最优解，最稳健、能诊断病态。

注意 $(A^TA)^{-1}A^T$ 正是高瘦满秩矩阵的**伪逆 $A^+$**（第 3 章 §3.6.3 预告的「最小二乘解」在此兑现）——正规方程、QR、SVD 三条路殊途同归，给同一个最优 $\hat{\mathbf x}$，只是数值稳健性递增。

> **与你的研究的连接（最小二乘 = 折叠数据分析的「拟合 / 标定 / 回归」主力）**：(1) **结构叠合 = 带正交约束的最小二乘**：Kabsch / 正交 Procrustes 求 $\min_R\|XR-Y\|_F^2$ s.t. $R^TR=I$——这是「最小二乘」加上「解必须是旋转」的约束版，靠 SVD 解（第 10 章）；普通最小二乘是它去掉正交约束的近亲。(2) **拟合 / 标定**：把实验观测（如 NMR 化学位移、SAXS 散射曲线、cryo-EM 密度）用线性模型拟合到结构参数，是标准最小二乘。(3) **力场参数 / 打分函数拟合**：用大量已知结构 / 能量数据回归出参数，最小化预测与真值的 $\ell_2$ 残差。(4) **接触图 / 距离约束反推坐标**：从一堆距离约束最小二乘地重建坐标（经典距离几何 / MDS 的线性化），也落在这个框架。(5) **回归是统计建模的细胞**：《概率与统计》里高斯噪声下的极大似然估计**恰好等价于最小二乘**（残差平方和 = 负对数似然），这把本章的几何和统计的推断焊在一起——最小二乘是「线代的投影」与「统计的高斯 MLE」的同一枚硬币的两面。**凡是『用模型最好地逼近一堆数据』，第一反应就该是最小二乘 = 投影到模型列空间。**

> **诚实的边界（最小二乘 = 高斯 / 二次损失下最优，离群点 / 非高斯噪声要换）**：最小二乘最小化的是残差的**平方**和，这等价于「假设噪声是独立同方差的高斯」。它的代价：**对离群点（outlier）极其敏感**——一个错得离谱的点（平方后权重巨大）能把整个拟合带偏。折叠数据里坏点不罕见（错误的距离约束、误配的原子、实验伪影）。对策：(1) 用**鲁棒损失**（如 $\ell_1$ / Huber，对大残差不平方放大，但 $\ell_1$ 没有闭式投影、要迭代解）；(2) **加权最小二乘**（已知误差大的点降权，= §4.1.4 的加权内积 $M=\Sigma^{-1}$，对应非等方差高斯）；(3) **RANSAC / 截断**（剔除离群点再拟合）。另外，最小二乘只在**列空间这个线性模型**里找最优——若真实关系是非线性的，再「最优」的投影也只是「最好的线性近似」。**纪律：用最小二乘前，问两件事——噪声像不像高斯（决定要不要鲁棒化），模型够不够线性（决定要不要换模型）。**

> **要点（§4.7）**：超定 $A\mathbf x=\mathbf b$ 无解时，**最小二乘** $\min\|A\mathbf x-\mathbf b\|^2$ = 把 $\mathbf b$ **正交投影**到列空间 = **正规方程** $A^TA\hat{\mathbf x}=A^T\mathbf b$ = 残差 ⊥ 列空间（**勾股**保证最优）——四种语言一件事。$(A^TA)^{-1}A^T$ 就是高瘦满秩的**伪逆**，但数值上别这么算（$A^TA$ 平方条件数），用 **QR**（第 5 章）/ **SVD**（第 8 章）。折叠里它是拟合 / 标定 / 回归 / 距离反推坐标的主力，**与高斯 MLE 等价**（接《概率与统计》），结构叠合是它的正交约束版（Kabsch，第 10 章）。对离群点敏感，按需鲁棒化 / 加权。务必玩 `04_projection_least_squares.html`。

---

## 本章小结

- **内积**（对称 + 双线性 + 正定）一出现就**派生**出长度 $\|\mathbf v\|=\sqrt{\langle\mathbf v,\mathbf v\rangle}$、夹角 $\cos\theta=\langle\mathbf u,\mathbf v\rangle/(\|\mathbf u\|\|\mathbf v\|)$、正交 $\langle\mathbf u,\mathbf v\rangle=0$——这是从「线性结构」到「欧氏几何」的观念跃迁。点积只是一种内积；加权内积（→马氏距离）、函数内积（→傅里叶）同样合法，**换内积 = 换几何**。嵌入相似度 = 余弦 = 归一化内积，注意力打分是缩放内积。（§4.1）
- **范数** = 正定 + 齐次 + 三角。$\ell_2$（欧氏、唯一来自内积、带完整几何）、$\ell_1$（稀疏、graphical lasso）、$\ell_\infty$（最坏分量）；矩阵范数 **Frobenius**（逐元素总能量，比较两接触图 / 两距离矩阵 / 两协方差）、**谱范数**（最大奇异值、最坏放大 / 条件数）。**Cauchy–Schwarz** 保证夹角有定义并推出**三角不等式**；**极化恒等式**从范数恢复内积，但只有满足平行四边形律的 $\ell_2$ 才来自内积。有限维**范数等价**。（§4.2）
- **标准正交基**下**坐标 = 内积**（$\mathbf v=\sum_i\langle\mathbf v,\mathbf q_i\rangle\mathbf q_i$，免求逆），随之有 **Parseval**（内积变坐标点积）与能量守恒——是 PCA / 简正模式 / 傅里叶「换到正交本征坐标后解耦」的统一根基。**正交补** $W\oplus W^\perp=\mathbb R^n$ 对应四个基本子空间的两对正交；正交则**勾股定理**成立（投影最优性的引擎）。（§4.3）
- **正交投影** $\hat{\mathbf b}=P\mathbf b$ = 子空间里离 $\mathbf b$ 最近的点，残差 $\perp$ 子空间（= **正规方程**的几何来源）。$P=\mathbf a\mathbf a^T/(\mathbf a^T\mathbf a)$（投到直线，秩 1 外积）、$P=A(A^TA)^{-1}A^T$（投到列空间）、$P=QQ^T$（正交基，最稳）；投影矩阵**幂等 + 对称**；**勾股定理证明投影最近**。投影 = 折叠里的降维 / 去噪 / 投掉刚体自由度 / 约束到合法子空间。（§4.4）
- **Gram–Schmidt** 逐个减投影 + 归一化，把任意基掰成标准正交基，等价于 **QR 分解** $A=QR$（第 5 章入口）。经典 GS **数值不稳**，实战用 MGS / Householder（**数学等价 ≠ 数值等价**，第 9 章）；本质是「去相关 / 提取增量信息」的流式算法。（§4.5）
- **正交矩阵** $Q^TQ=QQ^T=I$（$Q^{-1}=Q^T$）= **保内积 = 保长 = 保角**（四者等价）。$\det=\pm1$：$+1$ 旋转（$\mathrm{SO}(n)$，保定向）、$-1$ 反射（翻定向 = 镜像 / 手性）。这是**刚体叠合 / RMSD / Kabsch / AlphaFold frame / 等变**的代数核心——叠合只能用保距旋转且须排除反射（手性）。正交变换**条件数 = 1、不放大误差**，数值上最安全。长方 $Q$ 只有 $Q^TQ=I$（$QQ^T$ 是投影阵）。（§4.6，直通第 10 章）
- **最小二乘** $\min\|A\mathbf x-\mathbf b\|^2$ = 把 $\mathbf b$ **投影到列空间** = **正规方程** $A^TA\hat{\mathbf x}=A^T\mathbf b$ = 残差 ⊥ 列空间（**勾股**保证）——四语一事。$(A^TA)^{-1}A^T$ = 高瘦伪逆，但数值上用 **QR/SVD**（别平方条件数）。它是折叠拟合 / 标定 / 回归 / 距离反推坐标的主力，**与高斯 MLE 等价**（接《概率与统计》），Kabsch 是其正交约束版。对离群点敏感，按需鲁棒化 / 加权。（§4.7）

---

## 动手 / 思考小练习

> 编程题用 `numpy`（可选 `matplotlib`）。强烈建议第 4、5、7 题配合 `04_projection_least_squares.html` 一起做。先自己推 / 写，再看答案要点。

**1.（思考·内积定义几何）** 在标准点积下，$\mathbf u=(1,2,2)^T$、$\mathbf v=(2,-1,2)^T$。求 $\|\mathbf u\|$、$\|\mathbf v\|$、$\langle\mathbf u,\mathbf v\rangle$、夹角 $\cos\theta$，判断是否锐角。再换加权内积 $M=\operatorname{diag}(4,1,1)$ 重算 $\cos\theta_M$，体会「换内积 = 换几何」。（巩固 §4.1.3、§4.1.4）

*参考答案*：$\|\mathbf u\|=3$、$\|\mathbf v\|=3$、$\langle\mathbf u,\mathbf v\rangle=2-2+4=4$，$\cos\theta=4/9\approx0.44$（锐角，方向偏相似）。加权：$\langle\mathbf u,\mathbf v\rangle_M=4\cdot1\cdot2+1\cdot2\cdot(-1)+1\cdot2\cdot2=8-2+4=10$，$\|\mathbf u\|_M=\sqrt{4+4+4}=2\sqrt3$，$\|\mathbf v\|_M=\sqrt{16+1+4}=\sqrt{21}$，$\cos\theta_M=10/(2\sqrt3\sqrt{21})\approx0.63$——**同两个向量，换内积后夹角变了**。**为什么**：第一坐标被加权 4 倍，它俩在第一坐标上同号（都正），加权后「更对齐」。*延伸思考*：马氏距离用 $M=\Sigma^{-1}$，正是让「沿数据主轴」的方向被适当加权——这就是「考虑相关性的几何」。

**2.（动手·范数比较与单位球）** 对 $\mathbf v=(3,-4)^T$ 用 numpy 算 $\|\mathbf v\|_1,\|\mathbf v\|_2,\|\mathbf v\|_\infty$。再用 matplotlib 画出三种范数的单位球（$\|\mathbf v\|=1$ 的点集）：$\ell_2$ 是圆、$\ell_1$ 是菱形、$\ell_\infty$ 是方块。观察 $\ell_1$ 的尖角戳在坐标轴上。（巩固 §4.2.1）

*答案要点*：$\|\mathbf v\|_1=7$、$\|\mathbf v\|_2=5$、$\|\mathbf v\|_\infty=4$（验证 $\|\mathbf v\|_\infty\le\|\mathbf v\|_2\le\|\mathbf v\|_1$）。三个单位球：圆 / 菱形（顶点在 $(\pm1,0),(0,\pm1)$）/ 方块（$[-1,1]^2$）。**为什么**：$\ell_1$ 菱形的尖角恰在坐标轴上——当优化问题的「等高线」碰到这个可行域，最先碰到的往往是尖角（某些坐标 = 0），这就是 $\ell_1$ 诱导稀疏的几何直觉。*延伸思考*：画一条斜线（线性约束）去切这三个单位球，看切点：切 $\ell_1$ 菱形多半切在尖角（稀疏解），切 $\ell_2$ 圆切在光滑处（非稀疏）——这就是 LASSO vs 岭回归（graphical lasso 选稀疏精度矩阵的原因）。

**3.（思考·哪些范数来自内积）** 用平行四边形律 $\|\mathbf u+\mathbf v\|^2+\|\mathbf u-\mathbf v\|^2=2\|\mathbf u\|^2+2\|\mathbf v\|^2$ 检验：取 $\mathbf u=(1,0),\mathbf v=(0,1)$，对 $\ell_1$ 和 $\ell_\infty$ 验证它**不**成立，从而说明它们不来自任何内积、没有「正交投影」。（巩固 §4.2.4）

*参考答案*：$\ell_1$：$\|\mathbf u+\mathbf v\|_1=\|(1,1)\|_1=2$、$\|\mathbf u-\mathbf v\|_1=\|(1,-1)\|_1=2$，左边 $=4+4=8$；右边 $=2\cdot1+2\cdot1=4$。$8\ne4$，**不满足**。$\ell_\infty$：$\|(1,1)\|_\infty=1$、$\|(1,-1)\|_\infty=1$，左边 $=1+1=2$，右边 $=4$，也不满足。$\ell_2$：左边 $=2+2=4=$ 右边 ✓。**为什么**：平行四边形律是「范数来自内积」的充要条件；$\ell_1/\ell_\infty$ 违反它，故没有对应内积、谈不上角度 / 正交投影。这就是为什么投影 / 最小二乘 / PCA 必须站在 $\ell_2$ 上。*延伸思考*：所以「$\ell_1$ 正交投影」是无意义的说法——$\ell_1$ 下的「最近点」要靠优化求，没有闭式投影公式。

**4.（动手·投影到直线与子空间）** (a) 把 $\mathbf b=(1,1,1)^T$ 投影到 $\mathbf a=(1,2,2)^T$ 方向，求 $\hat{\mathbf b}$、残差 $\mathbf r$，验证 $\mathbf r\perp\mathbf a$。(b) 取 $A=\begin{bmatrix}1&0\\1&1\\1&2\end{bmatrix}$，用 $P=A(A^TA)^{-1}A^T$ 把 $\mathbf b=(6,0,0)^T$ 投到列空间，验证 $P^2=P$、$P^T=P$、$\mathbf b-P\mathbf b\perp\operatorname{col}(A)$。（巩固 §4.4）

*答案要点*：(a) $c=\mathbf a^T\mathbf b/\mathbf a^T\mathbf a=5/9$，$\hat{\mathbf b}=(5/9)(1,2,2)^T$，$\mathbf r=\mathbf b-\hat{\mathbf b}$，$\mathbf r^T\mathbf a=\mathbf a^T\mathbf b-c\,\mathbf a^T\mathbf a=5-5=0$ ✓。(b) numpy 算出 $P$，验证三条性质（数值上 $\approx$ 成立）。**为什么**：残差垂直是「最近」的充要条件（正规方程的几何）；幂等 + 对称是投影矩阵的指纹。*延伸思考*：把同一个 $A$ 用 `np.linalg.qr` 得 $Q$，验证 $QQ^T$ 和 $A(A^TA)^{-1}A^T$ 数值相等——两条路同一个投影，但 $QQ^T$ 更稳（不构造 $A^TA$）。

**5.（思考·为什么 $A^TA$ 在列无关时可逆）** 证明：若 $A$（$m\times n$）列线性无关，则 $A^TA$（$n\times n$）可逆。提示：考虑 $A^TA\mathbf x=\mathbf 0$，两边左乘 $\mathbf x^T$。（巩固 §4.4.2、§4.7）

*参考答案*：设 $A^TA\mathbf x=\mathbf 0$，左乘 $\mathbf x^T$：$\mathbf x^TA^TA\mathbf x=\|A\mathbf x\|^2=0$，故 $A\mathbf x=\mathbf 0$；因 $A$ 列无关，零空间平凡，$\mathbf x=\mathbf 0$。所以 $A^TA$ 的零空间只有 $\mathbf 0$，方阵零空间平凡即可逆。**为什么**：关键一步是 $\mathbf x^TA^TA\mathbf x=\|A\mathbf x\|^2$（内积的正定性把代数变成几何），把「$A^TA$ 可逆」化归到「$A$ 列无关」。*延伸思考*：这也说明 $A^TA$ 是**对称正定**的（$\mathbf x^TA^TA\mathbf x=\|A\mathbf x\|^2>0$ 对非零 $\mathbf x$），所以能做 Cholesky 分解（第 5 章），且它的特征值全正（第 7 章）——这是协方差 / Gram 矩阵永远半正定的根源。

**6.（动手·Gram–Schmidt 与数值不稳）** (a) 手写 / 编程对 $\mathbf a_1=(1,1,0),\mathbf a_2=(1,0,1),\mathbf a_3=(0,1,1)$ 做经典 Gram–Schmidt，验证结果 $Q^TQ\approx I$。(b) 构造一个接近相关的病态例子（如 $\mathbf a_2=\mathbf a_1+10^{-8}\mathbf e$），用经典 GS 算并检查 $\|Q^TQ-I\|$，再和 `np.linalg.qr` 对比，体会正交性损失。（巩固 §4.5）

*答案要点*：(a) 正常情形 CGS 给出合格的正交基。(b) 病态情形经典 GS 算出的 $Q$ 的 $\|Q^TQ-I\|$ 明显偏离 0（正交性丢失），而 `np.linalg.qr`（Householder）几乎完美。**为什么**：接近相关时残差 $\mathbf u_k$ 很小，舍入误差占比变大，经典 GS「一次性减投影」放大了误差；MGS / Householder 更稳。**数学等价 ≠ 数值等价**。*延伸思考*：这正是第 9 章的主题——同一个数学公式，不同算法的数值稳定性可以差很多，写科学计算代码必须懂这件事。

**7.（动手·正交矩阵保距 + 反射 = 手性翻转）** (a) 造一个 $3\times3$ 随机正交矩阵（对随机矩阵做 `np.linalg.qr` 取 $Q$），验证 $Q^TQ\approx I$、$\|Q\mathbf v\|=\|\mathbf v\|$、$\det Q=\pm1$。(b) 造一个反射 $F=\operatorname{diag}(1,1,-1)$（$\det=-1$），把一个手性物体（如四个不共面点 + 一个标记）变换后，说明它变成了镜像。（巩固 §4.6）

*答案要点*：(a) `Q,_=np.linalg.qr(np.random.randn(3,3))`，验证保距、$\det=\pm1$（QR 的 $Q$ 可能含反射，$\det$ 可正可负）。(b) $F$ 把 $z\to-z$，是镜像反射；作用在手性结构上得到其对映体（手性翻转）。**为什么**：保距变换分旋转（$\det=+1$，保手性）和反射（$\det=-1$，翻手性）两类；折叠叠合必须强制 $\det=+1$ 否则会配到不存在的镜像分子。*延伸思考*：在 Kabsch 算法里，若 SVD 给出的最优正交阵 $\det=-1$，要把最后一个奇异方向取反来强制成真旋转——这正是「排除手性翻转」在算法里的落点（第 10 章）。

**8.（动手·最小二乘拟合 + 与高斯 MLE 等价）** 生成带噪声的线性数据 $y_i=2x_i+1+\varepsilon_i$（$\varepsilon\sim\mathcal N(0,0.3^2)$，$x_i$ 取 0 到 1 共 50 点）。(a) 搭 $A=[\mathbf x\ \ \mathbf 1]$，用 `np.linalg.lstsq` 求斜率 / 截距，对照真值 $(2,1)$。(b) 同时用 $\hat{\mathbf x}=(A^TA)^{-1}A^T\mathbf b$ 算一遍，确认相同。(c) 画出拟合线、数据点、几条残差竖线，并验证残差与 $A$ 的列正交（$A^T\mathbf r\approx\mathbf 0$）。(d) 加入一个离群点 $(0.5, 50)$ 重拟合，看拟合被带偏多少。（巩固 §4.4、§4.7）

*答案要点*：(a)(b) 两法给近似 $(2,1)$ 的结果且彼此一致。(c) `A.T @ r` 接近零向量——残差正交于列空间，正是正规方程 / 投影的体现。(d) 一个离群点把斜率 / 截距显著带偏——最小二乘对离群点敏感（平方放大）。**为什么**：最小二乘 = 投影 = 高斯噪声下的 MLE（残差平方和 = 负对数似然），所以「最优拟合」就是「把观测投影到模型列空间」；离群点在平方损失下权重过大。*延伸思考*：把损失换成 Huber / $\ell_1$（或剔除离群点）再拟合，看鲁棒性改善——这把本章几何接到了《概率与统计》的鲁棒估计。

---

## 深入阅读指引

- **Gilbert Strang,《Introduction to Linear Algebra》（MIT 18.06）第 4 章（Orthogonality）**：本章的「主教材」。Strang 把正交性、投影、最小二乘、Gram–Schmidt、QR 串成一条线讲，几何直觉极强（「投影 = 列空间里离 $\mathbf b$ 最近的点」就是他的招牌），与本课 §4.3–§4.7 完全同源。读完本章后通读这一章巩固手感。
- **3Blue1Brown,《Essence of Linear Algebra》「Dot products and duality」一集**：用对偶视角讲点积「为什么是投影」（§4.1、§4.4 的几何根），动画把「内积 = 投影到一维」讲得入木三分。读 §4.1、§4.4 前后各看一遍，配合 `04_projection_least_squares.html` 拖动。
- **Sheldon Axler,《Linear Algebra Done Right》第 6 章（Inner Product Spaces）**：内积空间、正交、Gram–Schmidt、正交补、最小化问题的最干净现代处理（§4.1–§4.5 的理论底座），且证明优雅、不依赖坐标。想把「内积几何」学到无懈可击时读；其极化恒等式、平行四边形律的讨论正对 §4.2.4。
- **Trefethen & Bau,《Numerical Linear Algebra》Lecture 1–11（尤其 QR、Gram–Schmidt 稳定性、最小二乘）**：把 §4.5 的「CGS vs MGS vs Householder」、§4.7 的「别用正规方程、用 QR/SVD」讲到工业级，是第 9 章的先导。当你开始真写最小二乘 / QR 代码、关心数值稳定性时必读。
- **Golub & Van Loan,《Matrix Computations》（QR、最小二乘、Procrustes 章节）**：QR 分解、最小二乘的各种解法、正交 Procrustes（Kabsch 的数学母题）的权威参考。做结构叠合 / 数值最小二乘时的案头书，呼应 §4.5–§4.7 与第 10 章。
- **Kabsch, W. (1976/1978)「A solution for the best rotation to relate two sets of vectors」**：RMSD 最优旋转的原始论文，正是 §4.6（正交矩阵 = 旋转）+ §4.7（带正交约束的最小二乘）+ SVD 的合体。等你到第 10 章会重逢；现在读它的问题陈述，能看清本章工具如何直接落到「结构叠合」。
- **姊妹课交叉**：《概率与统计》第 5 章（多元高斯 / 协方差 / 精度 / 马氏距离 = 加权内积，§4.1.4）、回归与 MLE 章（最小二乘 = 高斯 MLE，§4.7）、第 11 章（DCA / graphical lasso 的 $\ell_1$ 稀疏，§4.2.1）；《信息论》（KL / 相对熵也可看成某种「广义投影」——信息几何，呼应 §4.4 的投影思想）；《蛋白质折叠》第 7 章（AlphaFold frame / IPA = 正交矩阵保距，§4.6）。

> **下一章预告**：本章我们靠「残差 ⊥ 列空间」推出了正规方程，又反复叮嘱「别显式解它，用 QR/SVD」——但 QR 到底怎么算、怎么用它稳定地解方程和最小二乘？第 5 章《线性方程组、矩阵分解（LU/QR/Cholesky）与最小二乘》就把「解 $A\mathbf x=\mathbf b$」这件事做到底：从高斯消元的矩阵语言（LU 分解，第 3 章剪切型初等矩阵的归宿）、到对称正定矩阵的 Cholesky（$A^TA$、协方差、Hessian 的专属、速度翻倍）、到本章 Gram–Schmidt 的工业版 QR，再把最小二乘用 QR 稳稳解出来。你会看到「分解」是数值线性代数的核心智慧——把一个难矩阵拆成几个易处理（三角 / 正交）的因子，解方程、求逆、求最小二乘全都顺水推舟。这是从「理解几何」走向「真能算、算得稳」的关键一步，直接服务于你将来跑 DCA 协方差求逆、做结构叠合、拟合力场参数的每一次实战。

