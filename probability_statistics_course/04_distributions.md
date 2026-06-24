# 第 4 章 常见分布及其关系网（含方向统计 von Mises）

> **本章定位**：前三章我们打磨的是「关于任意分布的通用工具」——概率空间与随机变量（第 2 章）、期望矩与母函数（第 3 章）。但通用工具像一套扳手，你还得有一柜子值得拧的螺丝。本章就是那柜子螺丝：把概率论里那批**有名字、天天用**的具体分布一一摊开，给出 pmf/pdf、均值方差、典型用途，更重要的是把它们之间盘根错节的**亲缘关系**（极限、特例、和、共轭）织成一张网。这不是「背分布表」的体力活：你会看到二项分布在稀有极限下变泊松、在大样本下变正态（这正是第 6 章中心极限定理的预演）；指数、卡方都只是 Gamma 家族里调了一个参数的特例；Beta 与 Dirichlet 是「概率向量上的分布」（住在单纯形上），它们将在第 8 章贝叶斯推断里作为共轭先验反复登场。本章有一节是折叠研究的**独门特色**：**方向统计（directional statistics）**。蛋白质主链的二面角 $\phi,\psi,\chi$ 活在圆周上（$-180°$ 与 $+180°$ 是同一点），普通高斯在这里**会犯错**，必须改用圆周上的「高斯」——**von Mises 分布**，以及环面上的二元 von Mises 来联合建模 $(\phi,\psi)$，直通折叠课程第 3 章的拉氏图与第 7、9 章的角度生成模型。最后两节拔高视角：**指数族（exponential family）** 把上面绝大多数分布统一成一个形式 $p_\theta(x)=h(x)\exp(\theta\cdot T(x)-A(\theta))$，它的对数配分函数 $A(\theta)$ 凸且「求导=矩」，为第 7 章（极大似然=矩匹配）与信息论课程 ch10（最大熵=指数族）铺路，而它「配分函数往往算不动」一事则通向本课第 10 章的蒙特卡洛/MCMC；**共轭先验（conjugate prior）** 则预告第 8 章的贝叶斯更新——为什么 Beta 配 Binomial、Dirichlet 配 Multinomial 后验还在同一族里。读完本章，你脑中会有一张「分布地图」：知道每个分布住在哪、长什么样、和谁通婚，以及在 MSA、二面角、突变计数、保守性估计这些折叠场景里该掏出哪一个。本章配有交互页 `04_distributions.html`（分布浏览器），强烈建议边读边拖参数看 pmf/pdf 与抽样直方图如何变化。

---

## 4.1 离散分布族：从一次抛硬币到计数

离散分布刻画「取值是可数集（通常是整数）」的随机量。折叠研究里它们无处不在：一个位点是否保守（伯努利）、MSA 某列各氨基酸的计数（多项）、一段基因组的突变个数（泊松）、做几次失败的对接尝试才命中（几何/负二项）。我们按「最简单 → 最复杂」的顺序串起来，注意它们彼此之间的生成关系——这正是 §4.4 关系网的离散半边。

### 4.1.1 伯努利分布：一次抛硬币

**伯努利分布（Bernoulli distribution）** 是最小的随机实验：一次成功/失败试验，成功概率 $p$。记 $X\sim\mathrm{Bernoulli}(p)$，取值 $\{0,1\}$，

$$p_X(1)=p,\qquad p_X(0)=1-p,\qquad \text{即}\quad p_X(x)=p^x(1-p)^{1-x},\ x\in\{0,1\}.$$

- **期望** $\mathbb E[X]=p$（用第 3 章 §3.1 直接求和：$0\cdot(1-p)+1\cdot p=p$）。
- **方差** $\mathrm{Var}(X)=p(1-p)$（因 $\mathbb E[X^2]=p$，故 $\mathrm{Var}=p-p^2$）。注意方差在 $p=1/2$ 处取最大 $1/4$——「最不确定」时散布最大，这与信息论里二元熵 $H_b(p)$ 在 $p=1/2$ 取峰是同一回事（信息论课程 ch2 §2.9）。

伯努利是一切离散分布的「原子」：下面的二项、几何、负二项都由独立伯努利试验拼出来。

> **与你的研究的连接（位点保守与否的指示变量）**：在多序列比对（multiple sequence alignment, MSA）里，问「第 $i$ 列在某条序列中是否就是该列的共识氨基酸」是一个伯努利变量；它的成功概率 $p_i$ 就是该位点的**保守度（conservation）**。把整列看成 $n$ 条序列上的 $n$ 个伯努利，立刻通向二项分布（§4.1.2），而 $p_i$ 的不确定性又通向 Beta 分布（§4.2.5）——这条「伯努利→二项→Beta 后验」的链，是估计「某位点有多保守、且我们对这个估计有多大把握」的标准管线。

### 4.1.2 二项分布：n 次独立抛硬币的成功数

把 $n$ 个**独立同分布**的 $\mathrm{Bernoulli}(p)$ 加起来，得到**二项分布（binomial distribution）** $X=\sum_{i=1}^n I_i\sim\mathrm{Binomial}(n,p)$，它数的是「$n$ 次试验里成功了几次」：

$$p_X(k)=\binom nk p^k(1-p)^{n-k},\qquad k=0,1,\dots,n.$$

- **期望** $\mathbb E[X]=np$，**方差** $\mathrm{Var}(X)=np(1-p)$。两者都用「指示变量法 + 线性性」一行得出（第 3 章 §3.1.2 例 1）：$\mathbb E[X]=\sum\mathbb E[I_i]=np$；因 $I_i$ 独立故方差可加，$\mathrm{Var}=\sum p(1-p)=np(1-p)$。
- **再生性（reproductive property）**：$X\sim\mathrm{Binomial}(n_1,p)$、$Y\sim\mathrm{Binomial}(n_2,p)$ 独立且**同一个 $p$**，则 $X+Y\sim\mathrm{Binomial}(n_1+n_2,p)$。直觉显然（合并两批试验）；用第 3 章 MGF 也能一行证（MGF 是 $(1-p+pe^t)^n$，相乘即指数相加）。**注意：$p$ 必须相同**，不同 $p$ 的二项之和不再是二项。

二项分布还有两条「大样本极限」，都将在第 6 章正式证明，这里先埋下：当 $n$ 很大时，$\mathrm{Binomial}(n,p)$ 近似于均值 $np$、方差 $np(1-p)$ 的**正态分布**（中心极限定理作用在 $X=\sum I_i$ 这个独立同分布之和上，是 de Moivre–Laplace 定理）；而当 $p$ 很小、$np$ 适中时它近似**泊松**（§4.1.5）。两条极限的分界在「$p$ 是否趋于 0」：$p$ 固定看正态、$p\to0$ 看泊松。这是关系网（§4.4）里离散族通向连续族的两座桥。

> **CS 读者陷阱（独立同分布是硬假设）**：二项分布的所有漂亮公式都建立在「$n$ 次试验独立、且成功概率恒为 $p$」之上。现实数据常违反它：MSA 里的序列**不独立**（它们有共同祖先，是一棵进化树上的相关样本），直接把「某列保守次数」当 $\mathrm{Binomial}(n,p)$ 会**高估有效样本量**、给出虚假窄的置信区间。这与第 3 章 §3.6.4 讲的「MD 快照不独立须打折」是同一个病：折叠生信里用**序列权重（sequence reweighting）** 把相关序列降权，得到「有效序列数 $N_{\rm eff}$」再代入，正是这条陷阱的对症下药（折叠课程第 5 章 DCA 必做这一步）。

### 4.1.3 类别分布与多项分布：从二元到 K 元

把伯努利的「两个结局」推广到 $K$ 个结局，就是**类别分布（categorical distribution）**：一次试验落在 $K$ 类之一，概率向量 $\boldsymbol\pi=(\pi_1,\dots,\pi_K)$，$\sum_k\pi_k=1$。常用 one-hot 编码 $X\in\{e_1,\dots,e_K\}$，$\mathbb P(X=e_k)=\pi_k$。它是 softmax 输出层、语言模型下一 token、以及「这个位点是 20 种氨基酸中的哪一种」的基本模型。

把 $n$ 个独立同分布的类别试验加起来，数每类出现几次，得到**多项分布（multinomial distribution）**：计数向量 $\mathbf N=(N_1,\dots,N_K)$，$\sum_k N_k=n$，

$$p(\mathbf N=\mathbf n)=\binom{n}{n_1,\dots,n_K}\prod_{k=1}^K\pi_k^{n_k},\qquad \binom{n}{n_1,\dots,n_K}=\frac{n!}{n_1!\cdots n_K!}.$$

- **边缘**：每个 $N_k\sim\mathrm{Binomial}(n,\pi_k)$，故 $\mathbb E[N_k]=n\pi_k$、$\mathrm{Var}(N_k)=n\pi_k(1-\pi_k)$。
- **协方差**：$\mathrm{Cov}(N_j,N_k)=-n\pi_j\pi_k$（$j\ne k$）——**负相关**，因为总数固定 $n$，一类多了别类就得少（这是「和约束」诱导的负相关，与第 3 章 §3.2 协方差直觉一致）。二项是 $K=2$ 的多项特例。

> **与你的研究的连接（MSA 某列就是一个多项样本）**：把 MSA 第 $i$ 列看成 $n$ 条序列在 20 种氨基酸（+ gap）上的多项计数 $\mathbf N_i=(N_{i,A},N_{i,R},\dots)$，它直接给出该列的**氨基酸频率谱（amino-acid profile）**——这是 PSSM、HMM profile、乃至 AlphaFold 输入 MSA 特征的统计起点。频率向量 $\boldsymbol\pi_i$ 的估计与不确定性，由它的共轭先验 **Dirichlet** 接管（§4.2.6），而「频率为 0 的氨基酸不该真信概率为 0」这件事，就靠 Dirichlet 的**伪计数（pseudocount）** 来救（§4.6）。

### 4.1.4 几何分布与负二项分布：等到第一次（第 r 次）成功

前面数「固定 $n$ 次里成功几次」；现在反过来数「等到成功需要几次」。

**几何分布（geometric distribution）**：独立 $\mathrm{Bernoulli}(p)$ 试验，第一次成功发生在第几次。两种常见约定（务必看清文献用哪种）：
- 「试验次数」版，$X\in\{1,2,\dots\}$：$p_X(k)=(1-p)^{k-1}p$，$\mathbb E[X]=1/p$，$\mathrm{Var}(X)=(1-p)/p^2$。
- 「失败次数」版，$X\in\{0,1,2,\dots\}$：$p_X(k)=(1-p)^kp$，$\mathbb E[X]=(1-p)/p$。

几何分布有著名的**无记忆性（memorylessness）**：$\mathbb P(X>m+n\mid X>m)=\mathbb P(X>n)$——已经失败了 $m$ 次，对「还要再等几次」毫无影响。在**取值于非负整数**（等间隔格点）的分布中，几何分布是**唯一**满足无记忆性的（无记忆性的这条「唯一性」刻画依赖于支撑是等间隔格点；连续世界里这个角色由指数分布扮演，§4.2.3）。

**负二项分布（negative binomial distribution）**：等到第 $r$ 次成功所需的失败次数 $X\in\{0,1,\dots\}$：

$$p_X(k)=\binom{k+r-1}{k}(1-p)^k p^r,\qquad \mathbb E[X]=\frac{r(1-p)}{p},\quad \mathrm{Var}(X)=\frac{r(1-p)}{p^2}.$$

$r=1$ 时退化为几何分布。负二项的现代用途主要不是「数失败」，而是当作**过度离散（overdispersed）的泊松替身**：它的方差**大于**均值（$\mathrm{Var}>\mathbb E$），而泊松的方差**等于**均值。

> **与你的研究的连接（RNA-seq 读段计数的过度离散）**：高通量测序里每个基因的读段计数（read count）名义上像泊松（独立计数事件），但实测的方差**远大于**均值——这叫过度离散，来自生物学重复间的真实变异。所以 DESeq2、edgeR 这些主流差异表达工具**不用泊松、用负二项**来建模计数。这是「负二项 = 带额外散布旋钮的泊松」在计算生物学里最常见的落地。同理，蛋白质家族中某模体的出现次数若也过度离散，负二项往往比泊松拟合更好。

### 4.1.5 泊松分布：稀有事件计数与二项的极限

**泊松分布（Poisson distribution）** 刻画「在固定时间/空间/序列长度内，独立稀有事件发生的次数」，单参数 $\lambda>0$（平均发生率）：

$$p_X(k)=\frac{\lambda^k e^{-\lambda}}{k!},\qquad k=0,1,2,\dots,\qquad \boxed{\ \mathbb E[X]=\mathrm{Var}(X)=\lambda.\ }$$

「均值 = 方差 = $\lambda$」是泊松的**指纹**：实测计数若方差明显超过均值，就该怀疑泊松不够、改用负二项（§4.1.4）。泊松也有再生性：独立 $X\sim\mathrm{Poisson}(\lambda_1)$、$Y\sim\mathrm{Poisson}(\lambda_2)$ 之和服从 $\mathrm{Poisson}(\lambda_1+\lambda_2)$。

**泊松作为二项的稀有极限（law of rare events / Poisson limit）**：这是关系网里最重要的一条极限。

> **定理（泊松极限定理）。** 设 $X_n\sim\mathrm{Binomial}(n,p_n)$，当 $n\to\infty$、$p_n\to0$、且 $np_n\to\lambda$（保持平均次数不变）时，对每个固定 $k$，
> $$\mathbb P(X_n=k)=\binom n k p_n^k(1-p_n)^{n-k}\ \longrightarrow\ \frac{\lambda^k e^{-\lambda}}{k!}.$$

证明思路（一行直觉）：$\binom nk p_n^k=\frac{n!}{k!(n-k)!}\big(\frac{\lambda}{n}\big)^k\to\frac{\lambda^k}{k!}$（因 $\frac{n!}{(n-k)!\,n^k}\to1$），而 $(1-\lambda/n)^{n-k}\to e^{-\lambda}$。所以**当试验次数极多、每次成功概率极小、但期望次数 $np\approx\lambda$ 适中时，二项就坍缩成泊松**。这就是为什么「大群体里的罕见突变数」「长序列里的稀有模体出现数」「单位时间里的稀有结合事件数」都近似泊松：海量机会 × 微小单次概率 = 泊松。

> **与你的研究的连接（突变计数与覆盖度）**：基因组每个碱基每代发生突变的概率极小（$\sim10^{-8}$），但碱基数巨大（$\sim10^9$），二者相乘给出每代每基因组约几十个突变——这正是泊松极限的教科书场景，群体遗传学里突变数、重组事件数都按泊松建模。测序里某位点的覆盖深度（read depth）在理想均匀采样下也近似泊松（均值 = 平均覆盖度）。一旦你看到「海量机会、每次极罕见」，第一反应就该是泊松；看到方差超均值，再升级到负二项。

> **要点**：离散族以**伯努利**为原子。固定次数数成功 → 二项（$\mathbb E=np$，$\mathrm{Var}=np(1-p)$）；$K$ 元推广 → 类别/多项（计数向量负相关 $\mathrm{Cov}=-n\pi_j\pi_k$）；数「等到成功」→ 几何（无记忆，$\mathbb E=1/p$）/ 负二项（过度离散的泊松替身）；稀有事件计数 → 泊松（$\mathbb E=\mathrm{Var}=\lambda$，是二项 $n\to\infty,p\to0,np\to\lambda$ 的极限）。建议打开 `04_distributions.html`，固定 $np=\lambda$ 同时增大 $n$、减小 $p$，亲眼看二项的 pmf 怎么收敛到泊松。（§4.1）

---

## 4.2 连续分布族：从均匀到 Gamma 家族与单纯形上的分布

连续分布刻画「取值在实数（或区间、向量）上」的随机量。折叠研究里：能量、距离、角度、概率向量、检验统计量，处处是连续分布。本节的组织主线是**两个家族**：以**指数分布**为根、**Gamma** 为干、卡方为枝的「正支撑等待时间族」；以及住在**单纯形（simplex）** 上、专门给「概率向量」当分布的 **Beta/Dirichlet** 族。中间穿插正态、$t$、$F$、Laplace 这些统计推断的主力。

### 4.2.1 均匀分布：最大无知的基准

**均匀分布（uniform distribution）** $X\sim\mathrm{Uniform}(a,b)$ 在区间 $[a,b]$ 上密度恒定：

$$f(x)=\frac{1}{b-a}\ (a\le x\le b),\qquad \mathbb E[X]=\frac{a+b}{2},\quad \mathrm{Var}(X)=\frac{(b-a)^2}{12}.$$

它是「在区间上无任何偏好」的最大熵分布（约束只有「固定支撑 $[a,b]$」、无矩约束）。**关于「最大熵」的统一口径**（本章会多次用到）：以下连续分布的最大熵论断都是就**微分熵相对于 Lebesgue 测度**而言的，且最大熵分布的形式完全由「**支撑 + 矩约束（充分统计量的期望）**」决定——均匀对应「固定支撑、无矩约束」、指数对应「$x\ge0$ + 固定均值」、正态对应「全实轴 + 固定均值与方差」、von Mises 对应「圆周 + 固定一阶三角矩」，这正是 §4.5 指数族–最大熵对偶的具体落点（详见信息论课程 ch10）。均匀分布也是**逆变换采样（inverse transform sampling）** 的种子：任何连续分布都能由 $U\sim\mathrm{Uniform}(0,1)$ 经 $F^{-1}(U)$ 生成（第 2 章 §2.5）。在 MCMC（第 10 章）里，均匀随机数是接受/拒绝判定的基础原料。

### 4.2.2 正态分布：中心极限与万物的近似

**正态分布 / 高斯分布（normal / Gaussian distribution）** $X\sim\mathcal N(\mu,\sigma^2)$ 是连续分布之王：

$$f(x)=\frac{1}{\sqrt{2\pi}\,\sigma}\exp\!\Big(-\frac{(x-\mu)^2}{2\sigma^2}\Big),\qquad \mathbb E[X]=\mu,\quad \mathrm{Var}(X)=\sigma^2.$$

它为什么无处不在，有三条独立的理由，每条都深刻：
1. **中心极限定理（central limit theorem, CLT）**：大量独立小扰动之和趋于正态（第 6 章主菜）——这解释了为什么测量误差、众多微观贡献之和近似正态。
2. **最大熵**：在「固定均值与方差」的约束下，正态是微分熵最大的分布（信息论课程 ch2 §2.9、ch8）——「只知道前两阶矩时最诚实的分布」。
3. **解析友好**：自共轭（正态先验 + 正态似然 = 正态后验，§4.6）、线性变换不变（正态的线性组合仍正态）、独立和仍正态（再生性，由第 3 章 MGF $e^{\mu t+\sigma^2t^2/2}$ 相乘即得）。

> **诚实的边界（高斯不是默认值）**：高斯的轻尾（密度按 $e^{-x^2}$ 衰减）让它对**离群值极敏感**，也使它在重尾数据上严重低估极端事件概率（第 3 章柯西那一课）。折叠/生信数据常有重尾或多峰：能量分布在相变附近双峰、距离分布有物理下界（不能为负，应考虑 Gamma/对数正态）、角度数据在圆周上（应考虑 von Mises，§4.3）。**别因为高斯好算就到处假设高斯**——这是初学者最常见的建模错误之一。

### 4.2.3 指数分布：连续世界的无记忆等待

**指数分布（exponential distribution）** $X\sim\mathrm{Exponential}(\lambda)$ 刻画「泊松过程中相邻事件之间的等待时间」，密度

$$f(x)=\lambda e^{-\lambda x}\ (x\ge0),\qquad \mathbb E[X]=\frac1\lambda,\quad \mathrm{Var}(X)=\frac1{\lambda^2}.$$

它是**唯一**无记忆的连续分布：$\mathbb P(X>s+t\mid X>s)=\mathbb P(X>t)$——已经等了 $s$，对「还要再等多久」毫无影响（与几何分布的离散无记忆性互为镜像）。指数分布也是「固定均值、$x\ge0$」下的最大熵分布。它是下一个家族 Gamma 的种子（$r=1$ 个事件的等待时间）。

> **与你的研究的连接（化学反应等待时间与 Gillespie 模拟）**：无记忆性不是数学玩具——它是**马尔可夫过程**的连续时间核心。在 Gillespie 随机模拟算法（用于基因表达、酶反应网络等随机化学动力学）里，**系统「下一次反应」的等待时间**服从指数分布，且其速率是**所有反应倾向之和** $a_0=\sum_j a_j$（不是单个反应的 propensity $a_j$）——抽出等待时间后，再按概率 $a_j/a_0$ 抽中具体是哪条反应发生（这是 Gillespie 直接法）。这也顺带演示了指数分布的一条性质：**多个独立指数的最小值仍是指数，且速率相加**（每条反应各自按 $a_j$ 计时，最先发生的那条的等待时间就是速率 $\sum_j a_j$ 的指数——这与 §4.2.4 Gamma「同 rate 指数之和」恰成对照：取和得 Gamma、取最小得速率相加的指数）。无记忆性保证了「系统当前状态决定一切、不必记住历史」，这正是把化学动力学写成马尔可夫链的前提。等到第 10 章讲 MCMC 的连续时间表亲时，这条无记忆性会再次出场。蛋白质构象在状态间的跳变（两态折叠的逗留时间）在简化模型里也常按指数建模。

### 4.2.4 Gamma 家族：把指数与卡方都收进来

**Gamma 分布（gamma distribution）** $X\sim\mathrm{Gamma}(\alpha,\beta)$（形状 shape $\alpha>0$、速率 rate $\beta>0$）密度

$$f(x)=\frac{\beta^\alpha}{\Gamma(\alpha)}x^{\alpha-1}e^{-\beta x}\ (x\ge0),\qquad \mathbb E[X]=\frac\alpha\beta,\quad \mathrm{Var}(X)=\frac\alpha{\beta^2},$$

其中 $\Gamma(\alpha)=\int_0^\infty t^{\alpha-1}e^{-t}\,\mathrm dt$ 是 Gamma 函数（整数 $\alpha$ 时 $\Gamma(\alpha)=(\alpha-1)!$）。Gamma 的关键身份是**「正支撑分布的母家族」**，它把好几个常见分布收编为特例：

| 取参数 | 得到 | 解释 |
|---|---|---|
| $\alpha=1$ | $\mathrm{Exponential}(\beta)$ | 1 个事件的等待时间 |
| $\alpha=r$（正整数）, 速率 $\beta=\lambda$ | Erlang 分布 | 第 $r$ 个泊松事件的等待时间 |
| $\alpha=\nu/2,\ \beta=1/2$ | $\chi^2_\nu$（卡方，§4.2.7） | $\nu$ 个独立标准正态平方和 |

**形状–速率（shape–rate）与形状–尺度（shape–scale）两套参数化**易混：速率 $\beta$ 与尺度 $\theta=1/\beta$ 互为倒数，`scipy.stats.gamma` 用尺度 `scale`、很多贝叶斯文献用速率。**读代码/论文务必看清用哪套**，否则均值算反。Gamma 也有再生性：同速率的 Gamma 之和形状相加，$\mathrm{Gamma}(\alpha_1,\beta)+\mathrm{Gamma}(\alpha_2,\beta)=\mathrm{Gamma}(\alpha_1+\alpha_2,\beta)$（因 Erlang 正是 $\alpha$ 个**独立同 rate（都等于 $\beta$）** 指数之和，把两组这样的指数拼起来就得到形状相加——再生性要求**速率相同**，与 §4.2.3「指数是 Gamma 的 $\alpha=1$ 种子」前后闭合）。

> **给数学/CS 读者的视角**：把 Gamma 家族想成一个「带两个旋钮的正支撑分布发生器」。形状 $\alpha$ 控制「峰离 0 多远、偏度多大」（$\alpha$ 小则贴着 0 单调衰减像指数，$\alpha$ 大则鼓成钟形并趋于正态——这是 CLT 的又一处体现，因为大 $\alpha$ 的 Gamma 是许多指数之和）；速率/尺度只做水平缩放。记住这张「指数–Erlang–卡方都是 Gamma 切片」的图，你就不必把它们当成三个孤立分布来背。建议在 `04_distributions.html` 里拖 Gamma 的 $\alpha$ 从 0.5 拉到 20，看它从「贴 0 的尖角」长成「对称钟形」。

### 4.2.5 Beta 分布：[0,1] 上的分布、概率的概率

**Beta 分布（beta distribution）** $X\sim\mathrm{Beta}(\alpha,\beta)$ 住在 $[0,1]$ 上，密度

$$f(x)=\frac{x^{\alpha-1}(1-x)^{\beta-1}}{B(\alpha,\beta)}\ (0\le x\le1),\qquad B(\alpha,\beta)=\frac{\Gamma(\alpha)\Gamma(\beta)}{\Gamma(\alpha+\beta)},$$

$$\mathbb E[X]=\frac{\alpha}{\alpha+\beta},\qquad \mathrm{Var}(X)=\frac{\alpha\beta}{(\alpha+\beta)^2(\alpha+\beta+1)}.$$

因为取值在 $[0,1]$，Beta 最自然的角色是**「概率的分布」**——对一个未知的成功概率 $p$ 本身建模。它的形状随两个参数极富弹性，值得记住几个典型：
- $\alpha=\beta=1$：退化为 $\mathrm{Uniform}(0,1)$（对概率「一无所知」）；
- $\alpha=\beta>1$：对称单峰，峰在 $0.5$，$\alpha=\beta$ 越大越尖（越「自信概率在 0.5 附近」）；
- $\alpha>\beta$（且都 $>1$）：单峰偏向 1（更可能成功）；反之偏向 0；
- $\alpha,\beta<1$：两端翘起呈 **U 形**，质量被推向 0 和 1（「要么几乎必成、要么几乎必败」的双模信念）。

直觉记法：$\alpha$ 像「伪成功数」、$\beta$ 像「伪失败数」，均值 $\frac{\alpha}{\alpha+\beta}$ 是「成功比例」，而 $\alpha+\beta$ 是「伪样本量」——越大方差越小、对 $p$ 越笃定（方差里分母含 $\alpha+\beta+1$，正是这个意思）。这套「伪计数」直觉在 §4.6 共轭更新里会变成字面公式。Beta 与 Gamma 有漂亮联系：若 $U\sim\mathrm{Gamma}(\alpha,1)$、$V\sim\mathrm{Gamma}(\beta,1)$ 独立，则 $\frac{U}{U+V}\sim\mathrm{Beta}(\alpha,\beta)$——「两个 Gamma 的归一化比例」就是 Beta，这也是 Dirichlet 构造的种子（§4.2.6）。建议在 `04_distributions.html` 里拖 Beta 的 $(\alpha,\beta)$，亲眼看它在「均匀—钟形—U 形—偏峰」之间连续变形，体会「一个分布族覆盖对概率的各种信念」。

### 4.2.6 Dirichlet 分布：单纯形上的分布、概率向量的分布

把 Beta 从「$[0,1]$ 上的一个数」推广到「和为 1 的 $K$ 维概率向量」，就是 **Dirichlet 分布（Dirichlet distribution）**。它住在 **$(K-1)$ 维单纯形（simplex）** $\Delta^{K-1}=\{\boldsymbol\pi:\pi_k\ge0,\sum_k\pi_k=1\}$ 上，参数 $\boldsymbol\alpha=(\alpha_1,\dots,\alpha_K)$，密度

$$f(\boldsymbol\pi)=\frac{1}{B(\boldsymbol\alpha)}\prod_{k=1}^K\pi_k^{\alpha_k-1},\qquad B(\boldsymbol\alpha)=\frac{\prod_k\Gamma(\alpha_k)}{\Gamma(\sum_k\alpha_k)}.$$

记 $\alpha_0=\sum_k\alpha_k$，则

$$\mathbb E[\pi_k]=\frac{\alpha_k}{\alpha_0},\qquad \mathrm{Var}(\pi_k)=\frac{\alpha_k(\alpha_0-\alpha_k)}{\alpha_0^2(\alpha_0+1)},\qquad \mathrm{Cov}(\pi_j,\pi_k)=\frac{-\alpha_j\alpha_k}{\alpha_0^2(\alpha_0+1)}.$$

$K=2$ 时 Dirichlet 退化为 Beta。**关键直觉**：$\alpha_0=\sum\alpha_k$ 是「集中度/伪样本量」——$\alpha_0$ 越大，分布越集中在均值 $\boldsymbol\alpha/\alpha_0$ 附近（对概率向量越「自信」）；$\alpha_0$ 小则分散，对称情形 $\alpha_k<1$ 时质量被推向单纯形的**边界（棱、面乃至顶点）**——倾向于让**一部分类别概率接近 0** 的「稀疏」概率向量（$K=2$ 即 Beta 的 U 形退化到两个顶点；$K\ge3$ 时更常见的是落到某些低维面/棱上，即部分类别为 0 而非只有单个类别独大）。这一点让 Dirichlet 成为「主题模型 LDA」「贝叶斯混合模型」里的标配先验——小 $\alpha$ 正是 LDA 里「鼓励每篇文档只用少数几个主题」的机制。

> **与你的研究的连接（氨基酸频率谱的 Dirichlet 先验与伪计数）**：MSA 某列的氨基酸计数是多项样本（§4.1.3），其频率向量 $\boldsymbol\pi$ 的不确定性正由 Dirichlet 接管。后验均值（§4.6）是 $\hat\pi_k=\frac{N_k+\alpha_k}{n+\alpha_0}$——分母分子里的 $\alpha_k$ 就是**伪计数（pseudocount）**：它保证「即使某氨基酸在这 $n$ 条序列里一次没出现，估计概率也不为 0」，从而避免 PSSM/HMM 在新序列上遇到未见氨基酸时打出 $-\infty$ 分数。BLAST 的 PSSM、HMMER 的 profile、乃至很多打分矩阵，背后都是「Dirichlet 先验 + 多项似然」的贝叶斯估计，更精细的版本用 **Dirichlet 混合先验（Dirichlet mixture）** 来编码「氨基酸的物化类别（疏水/带电/…）」的先验知识。这是本章「单纯形上的分布」在折叠生信里最高频的用途。

### 4.2.7 卡方、Student-t、F：正态衍生的检验三剑客

这三个分布几乎只在「从正态样本做推断」时出现，是第 9 章假设检验的主力。它们都由正态经「平方和 / 比值」构造而来——这是 §4.4 关系网右半边的核心。

**卡方分布（chi-square distribution）** $\chi^2_\nu$：$\nu$ 个独立标准正态的平方和 $\sum_{i=1}^\nu Z_i^2$，自由度 $\nu$。它是 $\mathrm{Gamma}(\alpha=\nu/2,\beta=1/2)$ 的别名，故 $\mathbb E=\nu$、$\mathrm{Var}=2\nu$。用于方差的推断、拟合优度检验（goodness-of-fit）、似然比检验的渐近分布。

**Student-t 分布（Student's t-distribution）** $t_\nu$：若 $Z\sim\mathcal N(0,1)$ 与 $W\sim\chi^2_\nu$ 独立，则 $T=\frac{Z}{\sqrt{W/\nu}}\sim t_\nu$。它比正态**尾巴更重**，而「重尾」的精确含义是：$t_\nu$ 的尾按**幂律** $\sim|t|^{-(\nu+1)}$ 衰减，而正态是 $e^{-t^2/2}$ 的**指数衰减**——多项式比指数衰减慢得多，这才是 t 比正态重尾的本质。自由度 $\nu$ 越小、尾越重、矩也越早发散：**$\nu\le1$ 时连期望都不存在**（$\nu=1$ 即**柯西分布**，第 3 章那个无期望的反例！），**$1<\nu\le2$ 时期望为 $0$ 但方差无穷**，**$\nu>2$ 才有有限方差 $\mathrm{Var}=\frac{\nu}{\nu-2}$**；$\nu\to\infty$ 时趋于标准正态（尾从幂律退回指数）。所以「t 总有均值」是个误解——别默认它的矩都存在。t 分布的诞生就是为了在「样本量小、用样本方差估计总体方差」时做出诚实（更宽）的置信区间——这是第 9 章 t 检验的根。它的重尾也使它成为**鲁棒回归**里替代高斯噪声的常用选择。

**F 分布（F-distribution）** $F_{d_1,d_2}$：两个独立卡方各除以自由度后的比值 $\frac{W_1/d_1}{W_2/d_2}$。它的**均值** $\mathbb E=\frac{d_2}{d_2-2}$（$d_2>2$，否则均值不存在）、**方差** $\mathrm{Var}=\frac{2d_2^2(d_1+d_2-2)}{d_1(d_2-2)^2(d_2-4)}$（$d_2>4$，否则方差无穷）——注意均值只依赖分母自由度 $d_2$ 且恒略大于 1（分子分母都在估同一方差时比值应在 1 附近）。用于比较两个方差、方差分析（ANOVA）、回归的整体显著性检验。

> **CS 读者陷阱（自由度不是玄学，是「用掉的约束」）**：自由度 $\nu$ 常让初学者困惑。一个干净的理解：从 $n$ 个数据估计方差时，你先要估均值 $\bar X$，这「用掉一个自由度」，剩下 $n-1$ 个独立的偏差信息，故样本方差除以 $n-1$（无偏）、对应 $\chi^2_{n-1}$。每估一个参数就「用掉」一个自由度。把自由度当成「数据点数减去已估参数数」，多数情形就对了。这条会在第 7 章（充分统计量）和第 9 章（检验）反复用到。

### 4.2.8 Laplace 分布：尖峰重尾与 L1 正则的影子

**Laplace 分布（Laplace distribution，又称双指数）** 密度 $f(x)=\frac{1}{2b}\exp(-|x-\mu|/b)$，$\mathbb E=\mu$、$\mathrm{Var}=2b^2$。它在中心**比高斯更尖**、尾巴**比高斯更重**（按 $e^{-|x|}$ 而非 $e^{-x^2}$ 衰减）。它最重要的身份是「L1 的概率对应物」：

> **给数学/CS 读者的视角**：高斯负对数似然是 $\frac{(x-\mu)^2}{2\sigma^2}+\text{const}$，即 **L2 损失**；Laplace 负对数似然是 $\frac{|x-\mu|}{b}+\text{const}$，即 **L1 损失**。所以「假设噪声是 Laplace」 ⟺ 「用绝对值损失」，「Laplace 先验」 ⟺ 「L1 正则（LASSO）」——这就是为什么 L1 正则会产生**稀疏**解：Laplace 先验在 0 处有尖峰，把弱系数往 0 压。把分布族和损失函数/正则项对应起来（高斯↔L2、Laplace↔L1、t↔鲁棒损失），是数学/CS 读者理解「概率建模 = 选损失」的总钥匙（呼应第 3 章 §3.4「换损失则最优预测从均值变中位数」）。

> **要点**：连续族两大家族——**正支撑等待时间族**（指数 →（Erlang/Gamma）→ 卡方，Gamma 把它们统一为两旋钮发生器）与**单纯形上的概率向量族**（Beta 是 $[0,1]$ 上「概率的分布」，Dirichlet 是单纯形上「概率向量的分布」，$\alpha_0$ 是集中度/伪样本量）。正态是 CLT/最大熵双重产物但轻尾、勿滥用；卡方/t/F 由正态的平方和/比值构造，是检验三剑客（t 的 $\nu=1$ 即柯西、$\nu\to\infty$ 即正态）；Laplace↔L1、高斯↔L2、t↔鲁棒损失。（§4.2）

---

## 4.3 方向 / 圆周统计：二面角为什么不能用普通高斯

这一节是折叠研究的独门内容，也是本章最该「慢读」的部分。它解决一个具体而尖锐的问题：**蛋白质主链的二面角 $\phi,\psi$（以及侧链 $\chi$）该用什么分布建模？** 答案不是高斯——而原因牵涉到一个常被忽视的几何事实：**这些角活在圆周上，不是直线上。**

### 4.3.1 圆周的拓扑：0° 与 360° 是同一点

折叠课程第 3 章告诉我们：每个二面角住在一个圆 $\mathbb S^1$ 上，角度是**周期的**——$-180°$ 与 $+180°$（等价地 $0°$ 与 $360°$）是**物理上同一个构象**。这个看似无害的事实，让「在直线上工作」的普通统计量全部失效。看三个翻车现场：

**翻车 1：算平均角。** 两个角 $350°$ 和 $10°$，物理上几乎一样（都在 $0°$ 附近）。算术平均 $(350+10)/2=180°$ —— 跑到了完全相反的方向！正确的「圆周均值」应该在 $0°$ 附近。错的根源是：算术平均把圆「剪开摊平成直线」，而剪口处的两点其实紧挨着。

**翻车 2：算方差/标准差。** 同理，一组都聚在 $0°$ 附近、但有的记成 $359°$ 有的记成 $1°$ 的角，普通标准差会算出巨大的散布（接近 $180°$），而它们其实非常集中。

**翻车 3：拟合高斯。** 高斯的支撑是整条实轴、是**单峰且无界**的；它无法表达「绕一圈回到原点」的周期结构，也无法把质量正确地「接」在 $\pm180°$ 的缝合处。强行在 $(-180°,180°]$ 上拟合高斯，会在边界产生人为的密度断裂。

> **CS 读者陷阱（角度别直接喂给欧氏算法）**：任何「假设输入住在欧氏空间」的算法——求均值、算欧氏距离、K-means、普通高斯混合、线性回归、标准 PCA——直接喂角度都会在缝合点 $\pm180°$ 出错。两个常用的补救：(1) 把角 $\theta$ 编码成单位向量 $(\cos\theta,\sin\theta)\in\mathbb R^2$（嵌入到平面圆上，缝合点自动连续），很多神经网络处理二面角就这么做——例如 AlphaFold2 预测**侧链扭转角 $\chi$** 时正是以 $(\sin,\cos)$ 形式输出来绕开角度的周期性（折叠课程第 7 章 §7.5.2）；不过要注意 AF2 的**主链 $\phi/\psi$ 并不是直接当 $(\sin,\cos)$ 角度回归的，而是由每个残基的刚体 frame（$\mathrm{SE}(3)$）导出**（§7.5.1），别一概而论成「AF2 把所有角都用 $(\sin,\cos)$」；(2) 直接改用圆周上的分布与统计量——这就是**方向统计（directional statistics）**，下面登场。

### 4.3.2 圆周均值与集中度：合向量长度 R

方向统计的第一步是重新定义「均值」和「散布」。把每个角 $\theta_i$ 看成单位向量 $(\cos\theta_i,\sin\theta_i)$，求它们的**矢量和**：

$$C=\sum_i\cos\theta_i,\quad S=\sum_i\sin\theta_i,\qquad \bar R=\frac1n\sqrt{C^2+S^2}\in[0,1].$$

- **圆周均值（circular mean）**：$\bar\theta=\mathrm{atan2}(S,C)$——合向量的方向。它对 $350°,10°$ 给出正确的 $\approx0°$。
- **平均合向量长度（mean resultant length）** $\bar R\in[0,1]$ 度量集中度：所有角都相同时 $\bar R=1$（完全集中）；角均匀散布一整圈时 $\bar R\approx0$（无方向性）。**圆周方差**定义为 $1-\bar R$。注意：圆周方差 $1-\bar R\in[0,1]$ 是一个**无量纲**的集中度度量，**不能直接拿去和直线方差比**（直线方差的量纲是「角度$^2$」，如度$^2$ 或弧度$^2$）；若想要一个量纲为弧度的散布量，另有**圆周标准差** $\sqrt{-2\log\bar R}$（$\bar R\to1$ 时它 $\to0$、$\bar R\to0$ 时发散）。

这套「先嵌入到平面、再取矢量和」的思路，是方向统计的统一引擎。下面的 von Mises 分布正是它的「参数分布」版本。

### 4.3.3 von Mises 分布：圆周上的「高斯」

**von Mises 分布（von Mises distribution）** 是圆周上对应高斯地位的分布，记 $\theta\sim\mathrm{vM}(\mu,\kappa)$，密度（关于角度，定义在 $(-\pi,\pi]$，自动周期延拓）

$$f(\theta)=\frac{1}{2\pi I_0(\kappa)}\exp\big(\kappa\cos(\theta-\mu)\big),$$

其中 $\mu$ 是**位置参数（mean direction）**（峰所在角度），$\kappa\ge0$ 是**集中度参数（concentration）**，$I_0(\kappa)$ 是零阶第一类修正贝塞尔函数（modified Bessel function），作归一化常数。读法：
- $\kappa=0$ 时 $f$ 退化为圆周上的均匀分布（无方向偏好）；
- $\kappa$ 越大，质量越集中在 $\mu$ 附近，分布越尖；
- **大 $\kappa$ 极限下，von Mises 近似于以 $\mu$ 为中心、方差 $1/\kappa$ 的高斯**（把 $\cos(\theta-\mu)\approx1-\frac12(\theta-\mu)^2$ 代入指数即得 $\exp(-\frac\kappa2(\theta-\mu)^2)$）——所以 $\kappa$ 扮演「精度（precision，方差倒数）」的角色，这是「圆周上的高斯」这一称呼的精确含义。

von Mises 也是**最大熵分布**：在「固定圆周一阶三角矩 $\mathbb E[\cos\theta],\mathbb E[\sin\theta]$」（等价于固定平均方向与集中度）的约束下，圆周上熵最大的分布就是 von Mises（与信息论课程 ch10 的最大熵–指数族框架完全一致；事实上 von Mises 是指数族成员，§4.5）。它的圆周均值就是 $\mu$，集中度由 $\kappa$（通过 $\bar R=I_1(\kappa)/I_0(\kappa)$）刻画。

**怎么从数据估 $(\mu,\kappa)$**？极大似然给出极漂亮的结果（也是矩匹配，§4.5）：用 §4.3.2 的合向量，$\hat\mu=\mathrm{atan2}(S,C)$（平均方向），而 $\hat\kappa$ 由解方程 $A(\hat\kappa)=\bar R$ 得到，其中 $A(\kappa)=I_1(\kappa)/I_0(\kappa)$ 是单调函数（实践用数值反解或现成近似公式）。直觉：**平均合向量越长（$\bar R$ 越接近 1），数据越集中，估出的 $\kappa$ 越大**。这把「圆周均值/集中度」（描述统计）与「von Mises 参数」（生成模型）严丝合缝地接了起来，正是因为 $(\cos\theta,\sin\theta)$ 同时是 von Mises 的充分统计量（§4.5.1 表）。注意小样本时 $\bar R$ 对 $\kappa$ 有正偏，需要偏差校正——这与第 7 章 MLE 的有限样本偏差是同一类问题。

> **诚实的边界（von Mises 的限度）**：von Mises 是**单峰、对称**的圆周分布。真实的二面角分布常**多峰**（拉氏图里 $\alpha$ 螺旋区、$\beta$ 折叠区、左手螺旋区是分开的几团），单个 von Mises 无法描述——要用 **von Mises 混合（mixture of von Mises）**。此外它不是「环面上的二元高斯的精确类比」：$(\phi,\psi)$ 的联合建模需要下面的二元 von Mises，且它的角度相关结构比平面高斯更微妙。

### 4.3.4 环面上的二元 von Mises：联合建模 (φ, ψ)

主链每个残基的 $(\phi,\psi)$ 是**两个角**，它们的联合取值空间是**环面（torus）** $\mathbb T^2=\mathbb S^1\times\mathbb S^1$（两个圆的乘积，像甜甜圈表面）——折叠课程第 3 章 §3.3.1 已点明这个拓扑。要建模 $(\phi,\psi)$ 的联合分布（捕捉它们之间的相关，比如某些 $\phi$ 偏好搭配某些 $\psi$），就要**环面上的二元 von Mises 分布（bivariate von Mises / sine 或 cosine 变体）**。常用的「sine 变体」密度形如

$$f(\phi,\psi)\propto\exp\big(\kappa_1\cos(\phi-\mu_1)+\kappa_2\cos(\psi-\mu_2)+\lambda\sin(\phi-\mu_1)\sin(\psi-\mu_2)\big),$$

其中 $(\mu_1,\mu_2)$ 是两个角的中心，$(\kappa_1,\kappa_2)$ 是各自集中度，$\lambda$ 控制两角之间的**圆周相关**。$\lambda=0$ 时退化为两个独立 von Mises 的乘积。

> **与你的研究的连接（拉氏图、构象采样与生成模型里的角度建模）**：拉氏图（Ramachandran plot，折叠课程第 3 章）就是 $(\phi,\psi)$ 在环面上的经验密度——那些允许区（$\alpha$ 螺旋、$\beta$ 折叠、左手螺旋）正是密度的几个峰。要把它写成概率模型，标准做法是 **von Mises 混合 / 二元 von Mises 混合**（每个允许区一个分量），这正是诸如 TorusDBN、以及现代蛋白质生成/采样模型里二面角分布的数学骨架。这条思路也直接进入**生成式扩散模型**：一般性的原理是——**只要你把噪声加在二面角/环面这种非欧支撑上，就必须用环面/圆周上的扩散过程**（如 wrapped 正态或 von Mises 形式的扩散核），而不能直接套欧氏高斯扩散，否则会在 $\pm180°$ 缝合处出错。在**内坐标/二面角空间**做扩散的工作（如分子构象生成里的 torsional diffusion）正是这么干的。

> **避免张冠李戴**：要注意，折叠课程第 9 章的 **RFdiffusion / FrameDiff 这一类骨架生成器走的并不是「二面角环面扩散」这条路**，而是另一条——它们在刚体框架流形 $\mathrm{SE}(3)=\mathrm{SO}(3)\times\mathbb R^3$ 上扩散：平移分量在 $\mathbb R^3$ 上做普通高斯扩散，旋转分量在旋转群 $\mathrm{SO}(3)$ 上做旋转扩散（见折叠课程第 9 章 §9.3.2 与本课第 10 章 §10.7）。二者的共同思想是「**在非欧流形上做扩散**」，但具体流形不同（环面 $\mathbb T^2$ vs. 框架流形 $\mathrm{SE}(3)$），别把 von Mises/wrapped 环面扩散核安到 RFdiffusion 头上。

换句话说：你这一节学的「角度活在圆周/环面上，要用 von Mises 而非高斯」，到折叠前沿会**字面**决定在角度空间做生成时该怎么设计噪声与似然——而当生成对象换成刚体框架时，则换成 $\mathrm{SE}(3)$ 上的扩散，思想同源、流形有别。建议在 `04_distributions.html` 里把 von Mises 的 $\kappa$ 从 0 拖到大，看它如何从「均匀一圈」收紧成「圆周上的钟形」，再对照拉氏图想象多峰版本。

> **要点**：二面角 $\phi,\psi,\chi$ 活在**圆周 $\mathbb S^1$**（$\pm180°$ 是同一点），普通均值/方差/高斯在缝合处全部翻车；正确工具是**方向统计**——圆周均值 $\mathrm{atan2}(S,C)$、集中度 $\bar R$、以及圆周上的「高斯」**von Mises 分布** $f\propto e^{\kappa\cos(\theta-\mu)}$（$\mu$ 位置、$\kappa$ 集中度，大 $\kappa$ 时近似方差 $1/\kappa$ 的高斯，是圆周最大熵分布）。联合 $(\phi,\psi)$ 住在**环面** $\mathbb T^2$，用二元/混合 von Mises 建模，直通拉氏图与角度生成模型。神经网络常用 $(\sin\theta,\cos\theta)$ 嵌入绕开缝合问题。（§4.3）

---

## 4.4 分布关系网：一张图把谁是谁连起来

前三节的分布不是孤岛，它们由极限、特例、求和、比值、共轭等关系编成一张网。把这张网装进脑子，你就能「由一个分布想到一串」，而不是孤立地背几十张表。下面这张 ASCII 关系图是本章的「地图」，请对照前面各节回看每条箭头。

```
                         离散半边                                            连续半边
   ┌──────────────────────────────────────────┐    ┌────────────────────────────────────────────────┐
   │                                            │    │                                                  │
   │   Bernoulli(p) ──[n 个独立同分布求和]──► Binomial(n,p)        Exponential(λ)=Gamma(1,λ)            │
   │        │                                   │    │        │                                         │
   │        │[K 元推广]                          │    │        │[r 个独立指数求和]                        │
   │        ▼                                   │    │        ▼                                         │
   │   Categorical(π) ──[n 个求和]──► Multinomial(n,π)         Gamma(α,β) ──[α=ν/2,β=1/2]──► χ²_ν       │
   │                                            │    │           ▲                                      │
   │   NegBinom(r,p) ──[r=1 退化]──► Geometric(p)              │[U/(U+V), U,V~Gamma]                   │
   │                                            │    │           ▼                                      │
   │   Binomial(n,p) ──[n→∞,p→0,np→λ]──► Poisson(λ)           Beta(α,β) ──[K 维推广]──► Dirichlet(α)   │
   │                                            │    │       (住在 [0,1])            (住在单纯形 Δ^{K-1})│
   └────────────────────┬───────────────────────┘    └──────────────────────────┬───────────────────────┘
                        │                                                        │
                        │  [n→∞，CLT，第 6 章]                                    │  [大 α / 大 ν，CLT]
                        ▼                                                        ▼
                ┌────────────────────────────  Normal / Gaussian  N(μ,σ²)  ◄──────────────┐
                │                          （CLT 的共同归宿、最大熵）                        │
                │   Z~N(0,1) 的平方与比值生成检验三剑客：                                    │
                │      ∑_{i=1}^ν Z_i²  =  χ²_ν                                              │
                │      Z / √(W/ν)  =  t_ν      （t_1=Cauchy 无期望； t_∞=N(0,1)）            │
                │      (W₁/d₁)/(W₂/d₂)  =  F_{d₁,d₂}                                        │
                └──────────────────────────────────────────────────────────────────────────┘

   圆周/方向半边（折叠特色）：
       Uniform on circle  =  von Mises(μ, κ=0)
       von Mises(μ, κ)  ──[大 κ]──►  近似 N(μ, 1/κ)        （圆周上的高斯）
       两个 von Mises  ──[环面 T²=S¹×S¹ 联合]──►  bivariate von Mises（建模 (φ,ψ)）

   共轭关系（先验—似然，→ 后验同族，第 8 章）：
       Beta —— Binomial          Dirichlet —— Multinomial
       Gamma —— Poisson          Normal —— Normal
```

读这张图的几条主干：
1. **离散主干**：伯努利是原子；求和给二项/多项；稀有极限给泊松；大样本给正态（CLT）。
2. **连续 Gamma 主干**：指数是 $\alpha=1$ 的 Gamma；卡方是 $\alpha=\nu/2,\beta=1/2$ 的 Gamma；Beta 是两个 Gamma 的归一化比例；Dirichlet 是 Beta 的多维推广。
3. **正态枢纽**：CLT 让离散与连续两半的「大样本极限」都汇到正态；而正态的平方和/比值又生成 $\chi^2/t/F$ 三剑客——所以正态既是「终点」（极限），又是「起点」（衍生检验分布）。（速查图把 $t$ 简记成「$t_1=$Cauchy 无期望、$t_\infty=N(0,1)$」，但别只记这两端：完整结论见 §4.2.7——$t_\nu$ 尾按幂律 $\sim|t|^{-(\nu+1)}$ 衰减，$\nu\le1$ 无均值、$1<\nu\le2$ 方差无穷、$\nu>2$ 才有 $\mathrm{Var}=\frac{\nu}{\nu-2}$。）
4. **圆周半边**：von Mises 在 $\kappa=0$ 是圆周均匀、大 $\kappa$ 近似高斯，环面上拼成二元 von Mises。
5. **共轭虚线**：Beta↔Binomial、Dirichlet↔Multinomial、Gamma↔Poisson、Normal↔Normal——这是第 8 章的入口（§4.6 预告）。

> **给数学/CS 读者的视角**：这张网背后有一个更深的统一原理——**绝大多数箭头的起点和终点都是指数族成员**（§4.5）。「独立同分布求和保持族」（二项、泊松、Gamma、正态的再生性）本质是「指数族的充分统计量可加」；「极限到正态」是 CLT 作用在充分统计量上；「共轭关系」是指数族似然配上「指数族共轭先验」必然后验同族（§4.6）。所以你不必把这几十条关系当孤立事实记——它们大多是「指数族 + 几条通用定理」的推论。下一节就把这个统一框架摆出来。

> **要点**：分布不是孤岛。离散主干 伯努利→二项/多项→（稀有极限）泊松；连续 Gamma 主干 指数→Gamma→卡方、Beta→Dirichlet；正态是 CLT 的共同归宿，又经平方和/比值衍生出 $\chi^2/t/F$；圆周半边 von Mises（$\kappa=0$ 均匀、大 $\kappa$ 近高斯）→环面二元 von Mises；共轭四对 Beta-Binomial / Dirichlet-Multinomial / Gamma-Poisson / Normal-Normal 通向第 8 章。这张网大半是「指数族 + 通用定理」的推论。（§4.4）

---

## 4.5 指数族：把分布动物园装进一个框架

前面的分布看似各异，但它们里的绝大多数（伯努利、二项、泊松、几何、正态、指数、Gamma、Beta、Dirichlet、卡方、von Mises……）都属于同一个大家庭——**指数族（exponential family）**。掌握这个统一框架，你对「极大似然为什么等于矩匹配」（第 7 章）、「最大熵为什么给出指数族」（**信息论课程 ch10**，注意不是本课第 10 章）、「共轭先验为什么存在」（§4.6、第 8 章）会一通百通。

### 4.5.1 标准形式与四个零件

一族分布称为**指数族**，若其 pmf/pdf 可写成

$$\boxed{\ p_\theta(x)=h(x)\,\exp\big(\theta^\top T(x)-A(\theta)\big).\ }$$

四个零件各司其职：
- **自然参数（natural / canonical parameter）** $\theta$：把原参数（如 $p$、$\lambda$、$\mu,\sigma^2$）重参数化后的「干净」参数。
- **充分统计量（sufficient statistic）** $T(x)$：数据中「关于 $\theta$ 的全部信息」的浓缩——给定 $T(x)$ 后，数据其余部分与 $\theta$ 无关（第 7 章正式讲）。
- **底测度 / 载体（base measure）** $h(x)$：与 $\theta$ 无关的部分。
- **对数配分函数（log-partition / cumulant function）** $A(\theta)=\log\int h(x)e^{\theta^\top T(x)}\,\mathrm dx$：归一化项，保证积分为 1。

举几个把熟悉分布写成此形式的例子（务必自己验算一遍，这是真正「懂」指数族的方式）。**记号约定**：本节起 $\theta$ 专指**自然参数**（向量）；数据点统一记作 $x$。这与 §4.3 不同——那里 $\theta$ 是「角度随机变量」，到这里角度数据要写成 $x$（即下表 von Mises 行里 $T(x)=(\cos x,\sin x)$ 的那个 $x$，就是 §4.3 的角度）。表里特地补上 $h(x)$ 一列：**只有 $h,T,A$ 三者自洽（使 $\int h\,e^{\theta^\top T}=e^{A}$）时，分布才真的归一化为 1**，所以验算时务必把 $h(x)$ 一起代进去，否则正态、von Mises 这种 $h$ 非平凡的行会对不上。

| 分布 | 底测度 $h(x)$ | 自然参数 $\theta$ | 充分统计量 $T(x)$ | $A(\theta)$ |
|---|---|---|---|---|
| $\mathrm{Bernoulli}(p)$ | $1$ | $\log\frac{p}{1-p}$（log-odds） | $x$ | $\log(1+e^\theta)$ |
| $\mathrm{Poisson}(\lambda)$ | $1/x!$ | $\log\lambda$ | $x$ | $e^\theta$ |
| $\mathcal N(\mu,\sigma^2)$ | $1/\sqrt{2\pi}$ | $(\mu/\sigma^2,\,-1/2\sigma^2)$ | $(x,x^2)$ | $\frac{\mu^2}{2\sigma^2}+\log\sigma$ |
| $\mathrm{Gamma}(\alpha,\beta)$ | $1$ | $(\alpha-1,-\beta)$ | $(\log x, x)$ | $\log\Gamma(\alpha)-\alpha\log\beta$ |
| $\mathrm{vM}(\mu,\kappa)$ | $1$ | $(\kappa\cos\mu,\kappa\sin\mu)$ | $(\cos x,\sin x)$ | $\log\big(2\pi I_0(\kappa)\big)$ |

（验算提示：正态那行只有取 $h(x)=1/\sqrt{2\pi}$ 时 $A=\frac{\mu^2}{2\sigma^2}+\log\sigma$ 才正确——若硬取 $h\equiv1$，则 $A$ 须补一个常数 $\tfrac12\log(2\pi)$ 写成 $\frac{\mu^2}{2\sigma^2}+\tfrac12\log(2\pi\sigma^2)$；泊松那行的 $h=1/x!$ 也不能漏，否则 $A=e^\theta$ 凑不出来。Bernoulli/Gamma/von Mises 的 $h$ 平凡或被 $T$ 吸收，取 $1$ 即可。）

注意 von Mises 也在表里——它的充分统计量正是 $(\cos x,\sin x)$（即 §4.3 角度的余弦/正弦），与 §4.3.2 的圆周矩矢量和完全对应，这解释了「为什么圆周统计的核心量是合向量」。再注意正态那一行：它的充分统计量是 $(x,x^2)$，正好对应「均值和方差就能完全刻画高斯」这件事——**充分统计量的维度，告诉你这个分布需要几个数才能确定**（伯努利/泊松一个、正态/Gamma 两个、von Mises 两个）。

**著名的非指数族**（务必记住反例，别误以为「所有分布都是指数族」）：
- **均匀分布** $\mathrm{Uniform}(0,\theta)$：支撑 $[0,\theta]$ **依赖参数 $\theta$**，无法写成「$h(x)$ 与 $\theta$ 无关」的形式；
- **Student-t 分布**：它是「正态除以卡方根」的比值（§4.2.7），是高斯的连续混合，不在指数族里——这也是 t 分布尾重、对离群值鲁棒的另一面；
- **混合分布**（如 von Mises 混合、高斯混合）：有限混合一般不是指数族（虽然每个分量是），这正是混合模型要靠 EM 而非闭式 MLE 求解的根源（第 7 章）。

### 4.5.2 对数配分函数的凸性与「求导=矩」

指数族的全部魔力，集中在 $A(\theta)$ 的两条性质上。

> **定理（$A$ 凸，且其导数生成累积量）。** 在自然参数的定义域上，$A(\theta)$ 是**凸函数**，且
> $$\nabla A(\theta)=\mathbb E_\theta[T(X)]\quad(\text{均值}),\qquad \nabla^2 A(\theta)=\mathrm{Cov}_\theta\big(T(X)\big)\succeq0\quad(\text{协方差}).$$

证明思路（一行）：直接对 $A(\theta)=\log\int h\,e^{\theta^\top T}\,\mathrm dx$ 求导，$\nabla A=\frac{\int T\,h\,e^{\theta^\top T}}{\int h\,e^{\theta^\top T}}=\mathbb E_\theta[T]$；再求一次导得协方差矩阵，协方差矩阵半正定故 $A$ 凸。这与第 3 章 §3.6 的「$\log Z$ 是累积量母函数 CGF、一阶导给均值二阶导给方差」**是同一件事**——$A(\theta)$ 就是把自然参数当母函数变量的 CGF（信息论 ch10 §10.2 从最大熵方向得到同一结论）。

这条「**$A$ 的导数 = 充分统计量的矩**」是后续两大支柱的发动机：
- **第 7 章（极大似然 = 矩匹配）**：在指数族上做极大似然估计，对数似然是 $\theta^\top\big(\sum_i T(x_i)\big)-nA(\theta)+\text{const}$，对 $\theta$ 求导令零得 $\nabla A(\hat\theta)=\frac1n\sum_i T(x_i)$，即「**模型的充分统计量期望 = 数据的经验充分统计量**」——MLE 就是矩匹配。又因 $A$ 凸，这个优化问题是凸的、解唯一（极小参数化下）。
- **信息论课程 ch10（最大熵 = 指数族）**：在「固定充分统计量的期望（矩约束）」下最大化熵，拉格朗日法直接吐出指数族 $p\propto e^{\theta^\top T}$，$A(\theta)$ 是配它的对数配分函数。（注意：本课**第 10 章是蒙特卡洛/MCMC**，并不讲最大熵；指数族与本课第 10 章的真正接点是另一条线——配分函数 $A(\theta)$ 算不动时只能靠采样，见下方「诚实的边界」。）

> **诚实的边界（有闭式不等于算得动）**：指数族「有干净的解析形式」常被误读为「好算」。陷阱在 $A(\theta)$ 里的那个积分/求和：对蛋白质共进化的 Potts 模型（一个充分统计量为「单点 + 成对」的指数族，信息论 ch2 §2.9、ch10 §10.3、折叠课程第 5 章），$A(\theta)$ 是对 $20^L$ 个构型求和，**计算上是 #P-难的**，算不动。所以实践用伪似然（pseudo-likelihood）、MCMC（第 10 章）、变分近似绕开 $A$。**「有闭式分布」与「能精确计算」是两回事**——这是指数族最重要的边界，做折叠生信必须时刻记得。

> **要点**：指数族 $p_\theta(x)=h(x)\exp(\theta^\top T(x)-A(\theta))$ 把本章绝大多数分布统一（伯努利/泊松/正态/Gamma/Beta/Dirichlet/von Mises 皆是；均匀、$t$、混合不是）。$\theta$ 自然参数、$T(x)$ 充分统计量、$A(\theta)$ 对数配分函数。核心定理：$A$ 凸、$\nabla A=\mathbb E[T]$、$\nabla^2A=\mathrm{Cov}(T)$（即 CGF，与第 3 章 §3.6 同源）。由此 MLE = 矩匹配 + 凸优化（第 7 章）、最大熵 = 指数族（**信息论课程 ch10**，非本课第 10 章）。但 $A$ 中的配分函数对大状态空间算不动（Potts 模型 #P-难），这才是通向本课第 10 章蒙特卡洛/MCMC 的那条线。（§4.5）

---

## 4.6 共轭先验：贝叶斯更新的「闭合」预告

最后一节预告第 8 章的核心机制——**共轭先验（conjugate prior）**。它回答一个实际问题：当你有先验信念、又观察到数据、想用贝叶斯定理更新成后验时，**后验长什么样、好不好算？**

### 4.6.1 共轭的定义与直觉

贝叶斯定理（第 8 章正式展开）：$\underbrace{p(\theta\mid x)}_{\text{后验}}\propto\underbrace{p(x\mid\theta)}_{\text{似然}}\cdot\underbrace{p(\theta)}_{\text{先验}}$。一般而言后验是个怪分布、要靠数值方法。但若先验选得「般配」，后验会**落回与先验同一族**，只是参数变了——这就是共轭：

> **定义（共轭先验）。** 给定似然族 $p(x\mid\theta)$，若先验 $p(\theta)$ 取自某族 $\mathcal F$，使得后验 $p(\theta\mid x)$ 仍属于 $\mathcal F$，则称 $\mathcal F$ 是该似然的**共轭先验族**。一句话：**共轭 = 后验与先验同族**。

共轭不是「正确先验」，只是「数学上方便、有闭式更新公式」的先验。它的存在不是巧合：**指数族似然总存在一个形式上的共轭先验族**（先验也取指数族形式、把自然参数 $\theta$ 线性嵌入、以 $(\theta,-A(\theta))$ 为充分统计量构造），这是 §4.5 框架的又一推论。需提醒一句：这个先验族在**合适的超参数下是正常（proper、可归一化）先验**，但某些超参数取值会让它退化为非正常先验——「是否 proper」要逐情形检查，这一点留到第 8 章正式处理。本章关系网图里的四对共轭，正是四种常见似然各自的共轭先验：

| 似然（数据模型） | 共轭先验 | 后验（同族） |
|---|---|---|
| Binomial / Bernoulli（成功比例 $p$） | Beta | Beta |
| Multinomial / Categorical（概率向量 $\boldsymbol\pi$） | Dirichlet | Dirichlet |
| Poisson（发生率 $\lambda$） | Gamma | Gamma |
| Normal（已知方差，均值 $\mu$） | Normal | Normal |

### 4.6.2 Beta–Binomial：一个具体的更新公式

把最重要的一对落到字面。设你想估计某事件的成功概率 $p$（如「某位点是保守的」的概率）。

- **先验**：$p\sim\mathrm{Beta}(\alpha,\beta)$（用伪成功数 $\alpha$、伪失败数 $\beta$ 编码先验信念，§4.2.5）。
- **似然**：观察到 $n$ 次独立试验中 $k$ 次成功，$k\mid p\sim\mathrm{Binomial}(n,p)$。
- **后验**：由贝叶斯定理，$p(p\mid k)\propto p^k(1-p)^{n-k}\cdot p^{\alpha-1}(1-p)^{\beta-1}=p^{(\alpha+k)-1}(1-p)^{(\beta+n-k)-1}$，恰好是

$$\boxed{\ p\mid k\ \sim\ \mathrm{Beta}(\alpha+k,\ \beta+n-k).\ }$$

更新规则美得不可思议：**先验的伪成功数加上观察到的成功数、伪失败数加上观察到的失败数**。后验均值

$$\mathbb E[p\mid k]=\frac{\alpha+k}{\alpha+\beta+n}$$

是「先验 + 数据」的加权折中：数据少时偏向先验均值 $\frac{\alpha}{\alpha+\beta}$，数据多时趋于经验比例 $\frac kn$（先验被数据淹没）。这正是 §4.2.6 伪计数公式的来源（Dirichlet–Multinomial 是它的多维版：后验 $\mathrm{Dirichlet}(\boldsymbol\alpha+\mathbf N)$，后验均值 $\frac{\alpha_k+N_k}{\alpha_0+n}$）。

> **与你的研究的连接（保守比例的不确定性 + 伪计数的贝叶斯出身）**：「某位点保守吗」这件事不该只报一个点估计 $\hat p=k/n$，而该报一个**分布**——Beta 后验 $\mathrm{Beta}(\alpha+k,\beta+n-k)$ 同时给出点估计（后验均值）和不确定性（后验方差/可信区间）。当某列只有 $n=5$ 条序列时，$k/n$ 极不可靠，而 Beta 后验会诚实地给出宽区间。更重要的是：§4.2.6 里救命的**伪计数**，在这里露出了它的贝叶斯本质——它就是 Dirichlet/Beta 先验的参数 $\alpha_k$。所以 PSSM/profile 里「加伪计数」不是工程 hack，而是「用共轭先验做贝叶斯估计」的正规操作；选不同伪计数 = 选不同先验信念（均匀先验 vs. 编码氨基酸物化类别的 Dirichlet 混合先验）。这条线把 §4.1（多项）→§4.2（Dirichlet）→§4.5（指数族共轭）→§4.6（更新公式）在「估计 MSA 保守性」这一个折叠任务上完全打通，第 8 章会把推断、可信区间、层次模型继续展开。

> **要点**：共轭 = 后验与先验同族，是「贝叶斯更新有闭式」的方便先验（指数族似然必有共轭先验）。四对：Beta–Binomial、Dirichlet–Multinomial、Gamma–Poisson、Normal–Normal。Beta–Binomial 更新极简洁——$\mathrm{Beta}(\alpha,\beta)\xrightarrow{k\text{ 成功},n-k\text{ 失败}}\mathrm{Beta}(\alpha+k,\beta+n-k)$，后验均值是先验与经验比例的加权折中；这正是 MSA 伪计数的贝叶斯出身。正式展开见第 8 章。（§4.6）

---

## 4.7 本章小结

- **离散族以伯努利为原子**：固定次数数成功 → 二项（$\mathbb E=np$，再生性需同 $p$）；$K$ 元 → 类别/多项（和约束诱导负协方差）；数等待 → 几何（无记忆）/ 负二项（过度离散的泊松替身，RNA-seq 计数主力）；稀有计数 → 泊松（$\mathbb E=\mathrm{Var}=\lambda$，是二项 $n\to\infty,p\to0,np\to\lambda$ 的极限，建模突变/覆盖度）。（§4.1）
- **连续族两大家族**：正支撑等待时间族（指数=Gamma(1)、Erlang、卡方=Gamma($\nu/2,1/2$)，Gamma 是两旋钮发生器）；单纯形上的概率向量族（Beta 是 $[0,1]$ 上「概率的分布」，Dirichlet 是单纯形上「概率向量的分布」，$\alpha_0$ 是集中度/伪样本量）。（§4.2）
- **正态是 CLT 与最大熵的双重产物**，自共轭、线性变换不变，但轻尾、对离群值敏感、勿默认套用；卡方/t/F 由正态平方和/比值构造（t 的 $\nu=1$ 即柯西、$\nu\to\infty$ 即正态），是检验三剑客；Laplace↔L1、高斯↔L2、t↔鲁棒损失。（§4.2）
- **二面角活在圆周/环面上**：$\pm180°$ 是同一点，普通均值/方差/高斯在缝合处翻车；用方向统计——圆周均值 $\mathrm{atan2}(S,C)$、集中度 $\bar R$、von Mises 分布 $f\propto e^{\kappa\cos(\theta-\mu)}$（圆周上的高斯，大 $\kappa$ 近似方差 $1/\kappa$ 的高斯，圆周最大熵）。（§4.3）
- **$(\phi,\psi)$ 联合住在环面 $\mathbb T^2$**，用二元/混合 von Mises 建模，直通拉氏图与角度生成/扩散模型；神经网络常用 $(\sin\theta,\cos\theta)$ 嵌入绕开缝合问题。（§4.3）
- **分布关系网**：离散主干、连续 Gamma 主干、正态枢纽（既是 CLT 极限又衍生 $\chi^2/t/F$）、圆周半边、共轭四对——大半是「指数族 + 通用定理」的推论，不必孤立背诵。（§4.4）
- **指数族 $p_\theta=h(x)\exp(\theta^\top T(x)-A(\theta))$ 统一动物园**（伯努利/泊松/正态/Gamma/Beta/Dirichlet/von Mises 皆是；均匀/t/混合不是）；$A$ 凸、$\nabla A=\mathbb E[T]$、$\nabla^2A=\mathrm{Cov}(T)$（即 CGF），通向 MLE=矩匹配（第 7 章）与最大熵=指数族（**信息论课程 ch10**，非本课第 10 章）；但配分函数对大状态空间算不动（Potts #P-难），这条算不动才是通向本课第 10 章蒙特卡洛的线。（§4.5）
- **共轭先验 = 后验与先验同族**（指数族似然必有）；四对 Beta-Binomial / Dirichlet-Multinomial / Gamma-Poisson / Normal-Normal；Beta–Binomial 更新 $\mathrm{Beta}(\alpha+k,\beta+n-k)$，后验均值是先验与经验比例的加权折中，正是 MSA 伪计数的贝叶斯出身（详见第 8 章）。（§4.6）

---

## 4.8 动手 / 思考小练习

**1（动手·编程，泊松是二项的极限）。** 仅用 `numpy/scipy`：固定 $\lambda=3$。对 $n=10,50,500,5000$ 各取 $p=\lambda/n$，画出 $\mathrm{Binomial}(n,p)$ 的 pmf（`scipy.stats.binom.pmf`）并叠加 $\mathrm{Poisson}(\lambda)$ 的 pmf。观察 $n$ 增大时二项如何收敛到泊松，并计算两者在 $k=0,\dots,15$ 上的最大绝对差随 $n$ 的变化。
（*提示*：`from scipy.stats import binom, poisson`；`k=np.arange(0,16)`；`max(abs(binom.pmf(k,n,3/n)-poisson.pmf(k,3)))` 随 $n$ 单调减小到接近 0。这实证了 §4.1.5 的泊松极限定理：海量机会 × 微小单次概率 = 泊松。）

**2（思考·辨析，过度离散与分布选择，与研究相关）。** 你统计某蛋白质家族 1000 个同源序列里某短模体（motif）的出现次数，得到样本均值 $\bar x=4.0$、样本方差 $s^2=12.0$。(a) 这个数据支持用泊松还是负二项建模？给出判据。(b) 若误用泊松，对「某序列出现 $\ge15$ 次」这种尾事件概率的估计会偏高还是偏低？为什么这在筛选异常序列时危险？
（*提示*：(a) 泊松要求 $\mathrm{Var}=\mathbb E$，这里 $s^2=12\gg\bar x=4$ 是明显**过度离散**，应选**负二项**（§4.1.4）。(b) 泊松尾比真实负二项尾**轻**，故会**低估**尾概率——把本不算稀奇的高计数误判为「极显著异常」，造成假阳性。这呼应 §4.1.4 RNA-seq 用负二项不用泊松的同一理由。）

**3（动手·编程，圆周均值 vs 算术均值，与研究相关）。** 仅用 `numpy`：给定一组二面角（度）`angles = np.array([350, 355, 5, 10, 359, 2])`。(a) 算普通算术平均，看它给出什么。(b) 按 §4.3.2 算圆周均值 `atan2(mean(sin), mean(cos))`（注意先转弧度），看它给出什么。(c) 解释两者为何天差地别，并说明这对在拉氏图上做聚类意味着什么。
（*提示*：(a) 算术平均 $\approx 180°$（完全错，因为剪口在 $0°$）。(b) 圆周均值 $\approx 0°$（正确，这些角都聚在 $0°$ 附近）。(c) 直接对角度跑 K-means/求均值会在 $\pm180°$ 缝合处把同一团角拆散或把均值甩到对侧——故拉氏图聚类要么用方向统计/von Mises 混合，要么先做 $(\sin,\cos)$ 嵌入再聚类，§4.3.1。）

**4（动手·编程，Beta–Binomial 后验与不确定性，与研究相关）。** 仅用 `numpy/scipy`：某 MSA 列在 $n$ 条序列里有 $k$ 条是共识氨基酸。取均匀先验 $\mathrm{Beta}(1,1)$。(a) 对 $(k,n)=(4,5)$ 和 $(k,n)=(80,100)$ 分别画出后验 $\mathrm{Beta}(1+k,1+n-k)$ 的 pdf，比较两条后验的宽窄。(b) 报告各自的后验均值与 95% 可信区间（`scipy.stats.beta.ppf` 取 0.025 和 0.975 分位）。(c) 解释为什么 $n=5$ 时即使 $k/n=0.8$ 也不能下结论「该位点高度保守」。
（*提示*：(a)(b) 后验 `beta(1+k,1+n-k)`；$(4,5)$ 给宽区间（如约 $[0.36,0.96]$），$(80,100)$ 给窄区间（约 $[0.71,0.87]$）。(c) $n=5$ 时数据极少、后验极宽，点估计 $0.8$ 的不确定性巨大——这正是 §4.6.2 说的「数据少时该报分布而非点估计」，也是为什么 MSA 深度（序列条数）直接决定保守性结论的可信度。）

**5（思考·证明，把分布写成指数族）。** (a) 把 $\mathrm{Poisson}(\lambda)$ 写成指数族标准形 $h(x)\exp(\theta T(x)-A(\theta))$，指出 $\theta,T(x),A(\theta)$，并验证 $A'(\theta)=\mathbb E[X]=\lambda$。(b) 用 $\nabla^2A=\mathrm{Cov}(T)$ 这条定理，不直接积分地说明泊松「均值=方差」。
（*提示*：(a) $p(x)=\frac1{x!}e^{x\log\lambda-\lambda}$，故 $h(x)=1/x!$、$\theta=\log\lambda$、$T(x)=x$、$A(\theta)=e^\theta=\lambda$；$A'(\theta)=e^\theta=\lambda=\mathbb E[X]$（§4.5.2）。(b) $T(x)=x$ 故 $\mathrm{Var}(X)=A''(\theta)=e^\theta=\lambda=A'(\theta)=\mathbb E[X]$——一阶导=均值、二阶导=方差且二者都等于 $\lambda$，无需做泊松求和。这展示「$A$ 求导=矩」的威力。）

**6（思考·辨析，von Mises 的限度与多峰）。** 有人用单个 von Mises 拟合一段蛋白质所有残基的 $\phi$ 角，发现拟合很差。(a) 用 §4.3.3 解释为什么单个 von Mises 注定拟合不好真实主链 $\phi$ 分布。(b) 应该改用什么模型？(c) 如果任务是联合建模 $(\phi,\psi)$ 而非单独 $\phi$，又该用什么、为什么不能简单地用两个独立 von Mises 相乘？
（*提示*：(a) 真实 $\phi$（更明显的是 $(\phi,\psi)$ 联合）是**多峰**的（$\alpha$ 螺旋区、$\beta$ 折叠区等几团），而单个 von Mises 是**单峰对称**的，§4.3.3 诚实边界。(b) **von Mises 混合**，每个允许区一个分量。(c) 用**环面上的二元 von Mises（或其混合）**，§4.3.4；两个独立 von Mises 相乘假设 $\phi\perp\psi$，会丢掉 $(\phi,\psi)$ 之间的相关（拉氏图允许区不是矩形而是有倾斜/相关的形状），需要 $\lambda\ne0$ 的相关项才能捕捉。）

---

## 4.9 深入阅读指引

**标准概率/统计教材（分布族与关系网）**
- **Wasserman, *All of Statistics*（2004），第 2 章及附录的分布表**：与本课定位最契合，常见分布的 pmf/pdf、均值方差、关系一表打尽——对应本章 §4.1–4.2、§4.4，做随手速查。
- **Blitzstein & Hwang, *Introduction to Probability*（第 2 版），离散与连续分布两章 + 「分布之间的联系」**：把伯努利→二项→泊松、指数→Gamma→卡方、Beta↔Dirichlet 这些关系用 story proofs 讲得直觉极好——对应本章 §4.1–4.4，补直觉首选。
- **Casella & Berger, *Statistical Inference*（第 2 版），第 3 章「常见分布族」**：分布族 + 指数族 + 充分统计量的标准研究生处理，含指数族凸性与矩——对应本章 §4.2、§4.5，想把指数族学扎实看它。

**指数族与共轭先验（深入 §4.5–4.6）**
- **Wainwright & Jordan, *Graphical Models, Exponential Families, and Variational Inference*（2008），第 3 章**：指数族、$A(\theta)$ 凸性与对偶、充分统计量的现代权威处理，直通图模型与变分推断——对应本章 §4.5，做表示/推断方向必读（也呼应第 5 章高斯图模型）。
- **Gelman et al., *Bayesian Data Analysis*（第 3 版），第 2–3 章**：共轭先验（Beta-Binomial、Dirichlet-Multinomial、Gamma-Poisson、Normal-Normal）逐一展开 + 弱信息先验的实务——对应本章 §4.6，第 8 章的主教材，提前读这两章。

**方向 / 圆周统计与蛋白质二面角（对应 §4.3）**
- **Mardia & Jupp, *Directional Statistics*（2000）**：方向统计的权威专著，圆周均值、von Mises、环面上的二元 von Mises、混合模型一应俱全——对应本章 §4.3，要认真建模二面角必备。
- **Boomsma et al., *"A generative, probabilistic model of local protein structure"*（PNAS, 2008，TorusDBN）**：用环面上的隐马尔可夫 + 二元 von Mises 建模主链 $(\phi,\psi)$ 的开创性工作——把本章 §4.3.4 的二元 von Mises 落到真实蛋白质主链，是「方向统计 × 折叠」的桥梁文献。
- **折叠课程第 3 章《结构的数学语言》§3.3**：$\phi,\psi,\omega,\chi$ 二面角的几何定义与「角度住在环面上」的拓扑——本章 §4.3 的几何前提，务必对照。

> **下一章预告**：本章把一维（与少数低维）分布逐一摊开，并用 von Mises 处理了圆周这个「非欧」支撑。但折叠研究里最核心的对象——多个原子的坐标、多个残基的接触、整条链的构象——是**高维且变量间强相关**的。第 5 章《多元分布、协方差与多元高斯》就来处理「向量随机变量」：协方差矩阵如何刻画整组变量的耦合、**多元高斯**为什么是高维世界的默认建模语言、以及它的逆——**精度矩阵（precision matrix）** 如何编码「条件独立」从而给出**高斯图模型（Gaussian graphical model）**。那张图模型正是把本课程 §4.5 的指数族、信息论 ch2/ch10 的最大熵 + 成对约束、与折叠课程第 5 章的共进化/DCA 接通的关键一跳——本章 §4.5 埋下的「Potts 模型是指数族」那条线，将在第 5 章的高斯版本里第一次显出全貌。
