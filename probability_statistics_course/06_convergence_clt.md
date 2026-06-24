# 第 6 章 收敛、大数定律与中心极限定理

> **本章定位**：到上一章为止，我们已经把「单个分布」的工具备齐了——概率空间（第 2 章）、期望与矩与母函数（第 3 章）、一柜子有名字的分布（第 4 章）、以及向量随机变量与多元高斯（第 5 章）。本章是全课程的**枢纽章**：它回答一个看似平凡、实则撑起整门统计学的问题——**为什么「重复测量很多次再平均」是可信的？** 答案分两层。第一层是**大数定律（law of large numbers, LLN）**：样本均值会收敛到真均值，所以「多测几次取平均」不是迷信，而是定理。第二层更深也更实用——**中心极限定理（central limit theorem, CLT）**：不仅均值收敛，连它**收敛过程中的随机涨落**都有确定的形状，那形状几乎总是**高斯**。这两条定理是误差棒、置信区间、标准误（standard error）、$p$ 值的共同出处；没有它们，第 7 章的极大似然标准误、第 9 章的假设检验与置信区间、第 10 章蒙特卡洛误差的估计都将无从谈起。本章的技术内核是**收敛模式（modes of convergence）**：几乎必然、依概率、$L^p$、依分布——它们强弱有别，这套语言让我们能精确地说出「$X_n$ 趋近 $X$」到底是什么意思（信息论课程 ch2 §2.7 只点到弱/强大数定律，本章把整张蕴含图讲全）。我们会用**特征函数**（承接第 3 章 §3.3.3）一行勾出 CLT 的证明骨架，给出非同分布版本的 **Lindeberg/Lyapunov 条件**与**多元 CLT**，并诚实地用 **Berry–Esseen 定理**告诉你「$n$ 要多大高斯近似才够准」——这依赖偏度与尾巴，绝不是「$n>30$ 就万事大吉」。然后是把 CLT 推广到「估计量的非线性函数」的实用工具 **Delta 方法**（第 7、9 章求标准误的主力），以及高维统计与机器学习泛化界的引擎——**sub-Gaussian / Bernstein / McDiarmid** 等进阶集中不等式。读完本章，你会真正理解：MD 轨迹与 MCMC 样本的时间平均为何收敛到系综平均、误差棒里的 $1/\sqrt n$ 从哪来、为什么「有效独立样本数」才是 CLT 里真正的 $n$。配套交互页 `06_clt_lln.html`：选一个基分布，亲眼看样本均值如何收敛、以及均值的抽样分布如何从五花八门趋于高斯。

---

## 6.1 收敛模式：四种「趋近」的精确含义

在确定性数学里，「数列 $a_n\to a$」只有一个意思：$\forall\varepsilon>0,\exists N,\forall n>N,|a_n-a|<\varepsilon$。可一旦 $X_n$ 是**随机变量**，「$X_n$ 趋近 $X$」就不再唯一——因为 $X_n$ 在每个样本点 $\omega$ 上取不同的值，你可以问「每条样本路径都收敛吗」，也可以问「偏离很大的概率趋于零吗」，还可以问「均方误差趋于零吗」，甚至只问「分布形状趋于一致吗」。这四个问题对应四种**收敛模式**，强弱分明。把它们分清，是读懂 LLN（用依概率/几乎必然）与 CLT（用依分布）的前提。

### 6.1.1 四个定义

几乎必然、依概率、$L^p$ 这三种收敛都要求 $X_n$ 与 $X$ 定义在**同一概率空间**上——因为它们都要谈 $X_n-X$ 或逐点之差 $X_n(\omega)-X(\omega)$，没有共同的 $\Omega$ 这些差根本无从定义。**依分布收敛是唯一不需要这个前提的**：它只比较分布（CDF），$X_n$ 与 $X$ 可以分别住在不同的概率空间里（见下文 §6.1.1 末尾）。

> **定义（几乎必然收敛，almost sure convergence，记 $X_n\xrightarrow{a.s.}X$）。** $\mathbb P\big(\{\omega:\lim_{n\to\infty}X_n(\omega)=X(\omega)\}\big)=1$。即：除去一个概率为 0 的「坏样本集」，**每一条样本路径都逐点收敛**。

> **定义（依概率收敛，convergence in probability，记 $X_n\xrightarrow{P}X$）。** 对任意 $\varepsilon>0$，$\mathbb P(|X_n-X|>\varepsilon)\to 0$（$n\to\infty$）。即：对任何固定容差 $\varepsilon$，「偏离超过 $\varepsilon$」这件事的概率随 $n$ 消失。

> **定义（$L^p$ 收敛，convergence in $L^p$，$p\ge1$，记 $X_n\xrightarrow{L^p}X$）。** $\mathbb E[|X_n-X|^p]\to 0$。$p=2$ 时称**均方收敛（mean-square convergence）**，是最常用的版本。

> **定义（依分布收敛，convergence in distribution / 弱收敛，weak convergence，记 $X_n\xrightarrow{d}X$）。** 对 CDF $F_n,F$，在 $F$ 的**每个连续点** $x$ 处 $F_n(x)\to F(x)$。等价地（更现代的陈述）：对每个有界连续函数 $g$，$\mathbb E[g(X_n)]\to\mathbb E[g(X)]$。

四者的「问法」一字排开：

```
 a.s. :  每条样本路径最终都钉到 X            （问轨迹）
  P   :  任给容差 ε，越界的概率 → 0          （问越界概率）
 L^p  :  平均第 p 次方误差 → 0               （问误差矩）
  d   :  分布形状（CDF）逐点趋同             （问分布，不问取值）
```

注意最后一个**只关心分布、不关心取值**：依分布收敛甚至不要求 $X_n$ 与 $X$ 在同一个概率空间上（CLT 的极限 $\mathcal N(0,1)$ 是一个「理想分布」，跟你的样本均值并不住在同一个 $\Omega$ 里）。这正是为什么 CLT 的结论写成 $\sqrt n(\bar X_n-\mu)\xrightarrow{d}\mathcal N(0,\sigma^2)$——我们只断言「左边的分布趋近右边」，不断言它「趋近某个具体随机变量的取值」。

### 6.1.2 强弱关系：一张蕴含图

这四种收敛不是平起平坐的，它们的强弱关系是本节的核心。先给结论图，再逐条解释：

```
   a.s.       ──────────►┐
   （几乎必然）           │     ┌────────────────────────┐
                         ├────►│   依概率  X_n →P X      ├────► 依分布  X_n →d X
   L^p (p≥1)  ──────────►┘     └────────────────────────┘        （最弱）
   （含均方 L^2）

   ↑ a.s. 与 L^p 各有一支独立箭头指向「依概率」；二者之间【没有横向箭头】
     ——a.s. 与 L^p 互不蕴含（见 §6.1.3 反例）。

   特例：若 X_n →d c（极限是常数），则反推 X_n →P c        （唯一的逆向箭头）
```

**蕴含一：a.s. ⟹ P。** 几乎必然收敛是最强的「逐点」收敛，它强过依概率。直觉：如果几乎每条路径最终都钉在 $X$ 上，那么「在某个固定 $n$ 越界」的概率当然趋于零。**反之不成立**（见 §6.1.3 的滑动窗口反例）。

**蕴含二：$L^p$ ⟹ P。** 这是 Markov 不等式（第 3 章 §3.5.1）一行的事：
$$\mathbb P(|X_n-X|>\varepsilon)=\mathbb P(|X_n-X|^p>\varepsilon^p)\le\frac{\mathbb E[|X_n-X|^p]}{\varepsilon^p}\to 0.$$
所以「平均误差趋零」自动给出「越界概率趋零」。**反之不成立**：依概率收敛允许「极小概率的极大偏离」拖垮期望（见 §6.1.3 的高尖峰反例）。

**蕴含三：P ⟹ d。** 依概率收敛比依分布强：取值都贴近了，分布形状当然趋同。**反之一般不成立**，但有一个**重要特例**：若极限是**常数** $c$（即 $X_n\xrightarrow{d}c$，退化分布），则可反推 $X_n\xrightarrow{P}c$。这个特例在统计里极有用——很多「一致性（consistency）」证明就是先证依分布收敛到常数、再升级成依概率。

**a.s. 与 $L^p$ 互不蕴含。** 这是初学者最容易想当然的地方：几乎必然收敛**不**蕴含 $L^p$ 收敛（路径都收敛了，但期望可能被罕见的巨大值毁掉），$L^p$ 收敛也**不**蕴含几乎必然收敛（平均误差趋零，但可以有「无穷多次小概率越界」让路径永不稳定）。两个反例在下一小节给出。它们唯一的公共下界是「依概率收敛」——这也是为什么**弱大数定律（用依概率）比强大数定律（用 a.s.）更容易证**：它只要 $L^2$ 收敛（方差趋零）再 Markov 一下即可。

> **给数学/CS 读者的视角**：把这张图记成「**a.s. 与 $L^p$ 是两座并立的高峰，都俯瞰着『依概率』这块平原，而平原又俯瞰着『依分布』这片海**」。证 LLN 你站在平原或高峰上（依概率/几乎必然），证 CLT 你只需站到海平面（依分布）——这解释了一个常被忽略的事实：**CLT 不能直接给你「样本均值的某条具体轨迹会怎样」**，它只描述分布。想要轨迹层面的保证（如「时间平均几乎必然收敛」），你需要的是 SLLN 或遍历定理（§6.2、第 10 章），不是 CLT。

### 6.1.3 三个分清强弱的反例

反例是这套语言真正「咬合」的地方。下面三个反例分别证明三条「反向不成立」。一律取概率空间 $\Omega=[0,1]$ 配均匀测度（即 $\mathbb P$ = Lebesgue 长度）。

**反例 A（依概率 ⟹̸ 几乎必然）：滑动窗口（typewriter sequence）。** 把 $[0,1]$ 切成越来越细的区间，让一个「指示函数窗口」反复扫过。具体地，对 $n=2^k+j$（$0\le j<2^k$），令 $X_n=\mathbf 1_{[j/2^k,\,(j+1)/2^k]}$。则窗口宽度 $1/2^k\to0$，故对任意 $\varepsilon\in(0,1)$，$\mathbb P(X_n>\varepsilon)=1/2^k\to0$，即 $X_n\xrightarrow{P}0$。但**对每个固定的 $\omega\in[0,1]$**，窗口会无穷多次扫过它（每一层 $k$ 都有一个 $j$ 让 $\omega$ 落在窗口里），所以 $X_n(\omega)$ 在 0 和 1 之间反复横跳、**永不收敛**。故 $X_n$ **不**几乎必然收敛到任何东西。寓意：依概率收敛只管「单个时刻越界的概率」，管不住「无穷多次越界」这件长程的事。

**反例 B（依概率 ⟹̸ $L^p$，且 a.s. ⟹̸ $L^p$）：高而窄的尖峰。** 令 $X_n=n\cdot\mathbf 1_{[0,1/n]}$。则 $\mathbb P(X_n\ne0)=1/n\to0$，故 $X_n\xrightarrow{P}0$；而且对每个 $\omega>0$，当 $n>1/\omega$ 后 $X_n(\omega)=0$，所以 $X_n\xrightarrow{a.s.}0$ 也成立。**但** $\mathbb E[X_n]=n\cdot\frac1n=1\not\to0$，更不用说 $\mathbb E[X_n^2]=n^2\cdot\frac1n=n\to\infty$。所以 $X_n$ **既不** $L^1$ 收敛**也不** $L^2$ 收敛。寓意：罕见但巨大的值可以让期望/方差爆掉，而完全不影响「依概率」与「几乎必然」——这就是为什么 LLN 要么假设方差有限（弱 LLN 走 $L^2$），要么单独假设 $\mathbb E|X|<\infty$（强 LLN 走 a.s.）。

**反例 C（$L^p$ ⟹̸ 几乎必然）：又是滑动窗口。** 反例 A 里的窗口序列 $X_n=\mathbf 1_{[j/2^k,(j+1)/2^k]}$ 还满足 $\mathbb E[X_n^p]=1/2^k\to0$，即 $X_n\xrightarrow{L^p}0$（任意 $p$）。但前已证它**不**几乎必然收敛。所以 $L^p$ 收敛也推不出几乎必然收敛。

把三个反例填回蕴含图，每一条「反向箭头不成立」都有了具体证人。

> **诚实的边界（同一个反例，多重身份）**：注意反例 A/C 是同一个序列，它同时见证「P ⟹̸ a.s.」和「$L^p$ ⟹̸ a.s.」；反例 B 同时见证「P ⟹̸ $L^p$」和「a.s. ⟹̸ $L^p$」。这不是巧合：a.s. 与 $L^p$ 各自缺的那块短板——前者怕「无穷多次小概率越界」，后者怕「罕见巨大值」——本就是两类不同的「坏事」，所以分别需要不同构造的反例。读到这里若你能凭直觉说出「该用滑动窗口还是高尖峰」，说明你已经把这套语言内化了。

> **要点**：四种收敛模式按强弱排成「a.s. 与 $L^p$ 并立 ⟹ 依概率 ⟹ 依分布」；唯一的逆向特例是「依分布到常数 ⟹ 依概率」。a.s. 与 $L^p$ 互不蕴含（滑动窗口 vs 高尖峰两反例）。LLN 活在「依概率/几乎必然」层（§6.2），CLT 活在「依分布」层（§6.3）。把这张图刻进脑子，后面所有渐近陈述你都能立刻判断「这是哪种收敛、有多强」。（§6.1）

---

## 6.2 大数定律：平均会收敛到真值

现在用上节的语言精确陈述并证明：把独立同分布（independent and identically distributed, i.i.d.）的样本平均，会收敛到它们的共同期望 $\mu$。有两个版本，分别对应两种收敛。

### 6.2.1 弱大数定律：两行切比雪夫证明

> **定理（弱大数定律，weak law of large numbers, WLLN）。** 设 $X_1,X_2,\dots$ 不相关（pairwise uncorrelated 即可，不必独立），同均值 $\mathbb E[X_i]=\mu$、同方差 $\mathrm{Var}(X_i)=\sigma^2<\infty$。记 $\bar X_n=\frac1n\sum_{i=1}^nX_i$。则
> $$\bar X_n\xrightarrow{P}\mu,\qquad\text{即}\quad\forall\varepsilon>0,\ \mathbb P(|\bar X_n-\mu|>\varepsilon)\to0.$$

**证明（两行）。** 由不相关性，方差可加（第 3 章 §3.2.3），故
$$\mathrm{Var}(\bar X_n)=\frac1{n^2}\sum_{i=1}^n\mathrm{Var}(X_i)=\frac{\sigma^2}{n}.$$
对 $\bar X_n$ 用切比雪夫不等式（第 3 章 §3.5.2，它需要的正是方差有限）：
$$\mathbb P(|\bar X_n-\mu|>\varepsilon)\le\frac{\mathrm{Var}(\bar X_n)}{\varepsilon^2}=\frac{\sigma^2}{n\varepsilon^2}\to0.\qquad\blacksquare$$

这个证明短得近乎透明，却包含了整个误差棒理论的种子：**样本均值的方差以 $\sigma^2/n$ 速率缩小，标准差以 $\sigma/\sqrt n$ 缩小**。那个 $1/\sqrt n$ 就是你在论文里画误差棒、报标准误时反复看到的「采样越多、误差越小、但只按平方根速率减小」的来源。想把误差减半，得把样本量翻**四**倍——这条经济学般冷酷的事实，直接出自这两行。

注意定理的假设有多弱：**只要不相关 + 方差有限**，连独立都不必、连同分布都可放松（不同方差时把上界换成 $\frac1{n^2}\sum\sigma_i^2$，只要它 $\to0$ 即可）。这种「robust」正是 WLLN 在实践里好用的原因。

### 6.2.2 强大数定律：路径几乎必然收敛

> **定理（强大数定律，strong law of large numbers, SLLN，Kolmogorov 版）。** 设 $X_1,X_2,\dots$ **独立同分布**，且 $\mathbb E|X_1|<\infty$（**只要一阶绝对矩有限，不需要方差有限！**）。则
> $$\bar X_n\xrightarrow{a.s.}\mu=\mathbb E[X_1].$$

SLLN 比 WLLN 强在两处：(1) 收敛模式从「依概率」升到「几乎必然」——它断言**整条样本路径**最终钉在 $\mu$ 上，而不只是「每个固定时刻越界概率小」；(2) 假设反而更弱——它**不要求方差有限**，只要 $\mathbb E|X_1|<\infty$（代价是要求严格的 i.i.d.，不能只「不相关」）。这个「假设更弱、结论更强」的反差，是 Kolmogorov 1930 年代的深刻成果，证明远超弱版的「两行」（需要 Kolmogorov 不等式 + 截断技巧 + Borel–Cantelli），这里只陈述、不证。

两个版本的实践含义对照：

| | 弱 LLN（WLLN） | 强 LLN（SLLN） |
|---|---|---|
| 收敛模式 | 依概率 $\xrightarrow{P}$ | 几乎必然 $\xrightarrow{a.s.}$ |
| 关键假设 | 不相关 + $\sigma^2<\infty$ | i.i.d. + $\mathbb E|X|<\infty$ |
| 证明 | 切比雪夫，两行 | Kolmogorov，需截断 + Borel–Cantelli |
| 通俗陈述 | 「大 $n$ 时偏离很可能小」 | 「这一条路径最终一定稳定」 |
| 哪里用 | 误差棒、矩估计一致性 | 蒙特卡洛、遍历性（单条轨迹的时间平均，第 10 章） |

> **CS 读者陷阱（柯西没有大数定律）**：SLLN 的假设 $\mathbb E|X_1|<\infty$ 不是装饰。柯西分布（第 3 章 §3.1.4、§3.3.4）连一阶绝对矩都发散，于是**它的样本均值不收敛**——我们在第 3 章用特征函数算过：$n$ 个标准柯西的样本均值仍是标准柯西，多采样毫无改善。如果你在代码里对一个重尾量（比如某些没有有限均值的网络度分布、或除以接近零的随机分母得到的比值）做 `np.mean` 并指望它「越来越稳」，你可能正撞在这堵墙上：均值看似在跳动收敛，实则永远在跳。**做实验前先问：这个量的均值存在吗？** 这是 LLN 的隐形前提。

### 6.2.3 经验分布与经验 CDF：不止均值收敛

LLN 不仅让样本均值收敛——它让**整个经验分布**收敛。给定 i.i.d. 样本 $X_1,\dots,X_n$，定义**经验 CDF（empirical CDF）**
$$\hat F_n(x)=\frac1n\sum_{i=1}^n\mathbf 1\{X_i\le x\}.$$
对**每个固定的 $x$**，$\mathbf 1\{X_i\le x\}$ 是 i.i.d. 的伯努利变量，期望 $F(x)$，于是 SLLN 直接给 $\hat F_n(x)\xrightarrow{a.s.}F(x)$——即「用频率估计概率」在每点都对。但能不能**一致地**（对所有 $x$ 同时）收敛？能：

> **定理（Glivenko–Cantelli，「统计学基本定理」）。** $\sup_x|\hat F_n(x)-F(x)|\xrightarrow{a.s.}0$。

一句话：经验 CDF **一致地**收敛到真 CDF（不只逐点，是上确界范数下整体收敛）。这条定理常被称为「统计学的基本定理」，因为它保证了**频率派统计的根基**——从有限样本估计的整个分布，会均匀地逼近真相。它也是 §6.6 自助法（bootstrap）的理论靠山：既然 $\hat F_n\approx F$，那么「从 $\hat F_n$ 里抽样」就近似于「从 $F$ 里抽样」。

> **与信息论的连接（AEP 是大数定律的信息论化身）**：把 LLN 用到一个特别的随机变量——**自信息** $-\log p(X)$——上，你就得到信息论里的**渐近均分性（asymptotic equipartition property, AEP）**。具体地，对 i.i.d. 样本 $X_1,\dots,X_n$，样本平均 $-\frac1n\log p(X_1,\dots,X_n)=\frac1n\sum_i[-\log p(X_i)]$ 由 LLN 依概率收敛到它的期望——而这个期望正是**熵** $H=\mathbb E[-\log p(X)]$。于是「绝大部分概率质量集中在约 $2^{nH}$ 条彼此近似等概率的『典型序列』上」（信息论课程 ch5 §5.1：「期望是承诺，AEP 是兑现」）。一句话：**AEP 就是把弱大数定律套在 $-\log p(X)$ 上得到的；它把『熵』从一个平均量，升级成『几乎必然会发生的事实』，恰如本节 LLN 把『真均值』从一个数字升级成『样本均值一定趋近的目标』。** 两门课的这条 LLN，在信息论里兑现成了压缩极限的根本（信息论 ch5–6）。

> **与你的研究的连接（MSA 某列的氨基酸频率谱）**：把 MSA（multiple sequence alignment）第 $i$ 列看成 $n$ 条序列在 20 种氨基酸上的样本，那一列的**经验频率向量**就是这个离散分布的经验估计——它正是 PSSM、HMM profile、AlphaFold 输入 MSA 特征的统计起点（与第 4 章 §4.1.3、信息论课程 ch2 §2.8 同源）。Glivenko–Cantelli（的离散/多项版本）保证：序列越多，这个频率谱越逼近该位点真实的氨基酸偏好。但请记住信息论 ch2 §2.4.3 的警告：MSA 序列因共同祖先（沿系统发育树逐代突变的马尔可夫结构）而**相关**（系统发育噪声 phylogenetic noise），不是 i.i.d.，所以这里的「$n$」不是序列条数，而是**有效独立样本数** $N_{\rm eff}$（信息论 ch2 §2.7.1 在 LLN 框架下也讨论了有效样本数）——这件事 §6.3.4 还要从 CLT 角度再算一次账。

> **要点**：弱 LLN（依概率，切比雪夫两行，只需不相关 + 方差有限）与强 LLN（几乎必然，需 i.i.d. + $\mathbb E|X|<\infty$）共同保证样本均值收敛到真均值，误差按 $\sigma/\sqrt n$ 缩小。Glivenko–Cantelli 把这一保证从「均值」升级到「整个经验 CDF 一致收敛」，是频率派统计与自助法的根基。柯西这种无一阶矩的分布是 LLN 的反例——别对它求平均。（§6.2）

---

## 6.3 中心极限定理：涨落服从高斯

LLN 说 $\bar X_n\to\mu$，但它没说**怎么趋近**——围绕 $\mu$ 的随机涨落长什么样？答案是本章皇冠上的明珠：**把涨落放大 $\sqrt n$ 倍后，它依分布收敛到高斯**，而且这个高斯几乎不依赖原分布长什么样。这就是为什么高斯无处不在、为什么误差棒默认画成「均值 ± 标准误」并按高斯解读。打开 `06_clt_lln.html`，选一个奇形怪状的基分布（指数、双峰、均匀都行），把 $n$ 从 1 拖到大，你会**亲眼看到**均值的抽样分布如何从原分布的怪样子，逐渐塑成同一口钟形——这一节就是这个动画的数学。

### 6.3.1 经典 CLT（i.i.d. 版）与特征函数证明骨架

> **定理（经典中心极限定理，i.i.d. 版）。** 设 $X_1,X_2,\dots$ i.i.d.，$\mathbb E[X_i]=\mu$、$\mathrm{Var}(X_i)=\sigma^2\in(0,\infty)$（**均值与方差都有限，且方差非零**）。记 $\bar X_n=\frac1n\sum X_i$。则标准化后的样本均值
> $$Z_n:=\frac{\bar X_n-\mu}{\sigma/\sqrt n}=\frac{\sum_{i=1}^n X_i-n\mu}{\sigma\sqrt n}\xrightarrow{d}\mathcal N(0,1).$$
> 等价写法：$\sqrt n(\bar X_n-\mu)\xrightarrow{d}\mathcal N(0,\sigma^2)$。

两点要先咬清楚。第一，**标准化是必须的**：$\bar X_n$ 本身趋于常数 $\mu$（退化分布），其涨落以 $\sigma/\sqrt n$ 速率消失；你得用恰好 $\sqrt n$ 这个放大率去「显微」那个正在塌缩的涨落，才能看到非退化的极限。放大率小了（如 $n^{1/3}$）极限仍是 0，大了（如 $n$）极限会发散。$\sqrt n$ 是唯一让涨落「既不塌缩也不发散」的临界标度。第二，**极限分布不含原分布的任何形状信息**，只剩 $\mu,\sigma^2$——这就是 CLT 惊人的「普适性（universality）」：无论 $X_i$ 是抛硬币、是指数等待时间、还是双峰怪物，标准化均值都流向同一口标准正态钟。

**证明骨架（特征函数法，承接第 3 章 §3.3.3）。** 不失一般性设 $\mu=0,\sigma=1$（否则先中心化、标准化每个 $X_i$）。设 $X_i$ 的特征函数为 $\varphi(t)=\mathbb E[e^{itX_1}]$。因 $\mathbb E[X_1]=0,\mathbb E[X_1^2]=1$，对 $\varphi$ 在 0 处做二阶 Taylor 展开（第 3 章 §3.3.3 的「求矩」规则给出导数值）：
$$\varphi(t)=1+i\,\mathbb E[X_1]\,t-\tfrac12\mathbb E[X_1^2]\,t^2+o(t^2)=1-\tfrac{t^2}{2}+o(t^2).$$
$Z_n=\frac1{\sqrt n}\sum_{i=1}^nX_i$，由独立性，特征函数相乘（第 3 章 §3.3.1）：
$$\varphi_{Z_n}(t)=\Big[\varphi\big(\tfrac t{\sqrt n}\big)\Big]^n=\Big[1-\frac{t^2}{2n}+o\big(\tfrac1n\big)\Big]^n.$$
取 $n\to\infty$，用经典极限 $(1+a_n/n)^n\to e^a$（这里 $a=-t^2/2$）：
$$\varphi_{Z_n}(t)\to e^{-t^2/2},$$
而 $e^{-t^2/2}$ 正是 $\mathcal N(0,1)$ 的特征函数。由 **Lévy 连续性定理**（第 3 章 §3.3.3：特征函数逐点收敛 $\iff$ 依分布收敛；完整版还需**极限特征函数在 $t=0$ 处连续**，此处 $e^{-t^2/2}$ 显然满足），得 $Z_n\xrightarrow{d}\mathcal N(0,1)$。$\blacksquare$

这个证明的美在于它把「抽象的分布收敛」彻底化约成「一个初等极限 $(1-\frac{t^2}{2n})^n\to e^{-t^2/2}$」。高斯之所以是吸引子，根子就在 $e^{-t^2/2}$ 是这个极限运算的不动点——**任何均值方差有限的分布，其特征函数在原点都长成 $1-\frac{\sigma^2t^2}{2}+\dots$ 这个二次形状，而二次形状在「$n$ 次方 + 缩放」下唯一地流向高斯**。这就是普适性的代数根源。

> **给数学/CS 读者的视角（CLT 是一个不动点定理）**：把「取两个独立同分布的标准化平均」看成作用在「均值 0、方差 1 的分布」空间上的一个算子 $T$（renormalization 群的离散版）。CLT 等价于说：$T$ 的迭代有唯一吸引不动点，就是 $\mathcal N(0,1)$。这个视角解释了两件事：(1) 为什么高斯在卷积/求和下「自我复制」——它是不动点；(2) 为什么有「另类 CLT」——当方差无限（重尾、幂律尾指数 $\alpha<2$）时，吸引子不再是高斯，而是 **$\alpha$-稳定分布（$\alpha$-stable distribution）**，柯西（$\alpha=1$）就是其中一个。所以「均值方差有限」不是技术性废话，它正是「落入高斯吸引域」的入场券。

### 6.3.2 非同分布：Lindeberg 与 Lyapunov 条件

经典 CLT 要求 i.i.d.，但现实里求和的项常常**不同分布**（不同方差、不同形状）。例如，回归残差、加权平均、不同测量精度的实验汇总。此时还能不能 CLT？能，只要「没有任何一项主导整个和」——这就是 Lindeberg 条件要管的事。

设 $X_1,X_2,\dots$ 独立（**不必同分布**），$\mathbb E[X_i]=0$、$\mathrm{Var}(X_i)=\sigma_i^2<\infty$，记 $s_n^2=\sum_{i=1}^n\sigma_i^2$。

> **定理（Lindeberg–Feller CLT）。** 若对每个 $\varepsilon>0$ 满足 **Lindeberg 条件**
> $$\frac1{s_n^2}\sum_{i=1}^n\mathbb E\!\left[X_i^2\,\mathbf 1\{|X_i|>\varepsilon s_n\}\right]\xrightarrow{n\to\infty}0,$$
> 则 $\dfrac1{s_n}\sum_{i=1}^nX_i\xrightarrow{d}\mathcal N(0,1)$。

Lindeberg 条件长得吓人，但意思很朴素：**总方差 $s_n^2$ 里，来自「异常大的单项」（超过 $\varepsilon s_n$ 的那部分）的贡献，占比要趋于零**。换句话说，**没有哪一项独大**，每一项对总和的影响都是「无穷小」的。一个直接推论是 Lindeberg 条件蕴含 $\max_i\sigma_i^2/s_n^2\to0$（单项方差占比消失）——这正是「群体效应、无个体主导」的精确表述。CLT 的普适性在这里看得更清楚：高斯之所以涌现，是因为**大量微小独立扰动的叠加**，没有一个扰动能在尺度上压倒其余。

> **充要性（这才是「Feller」冠名的来历）**：上面把 Lindeberg 条件当**充分**条件用，但 Lindeberg–**Feller** 是一个**充要**结果——在**渐近可忽略**前提 $\max_i\sigma_i^2/s_n^2\to0$（即 **Feller 条件**）下，标准化和 $\frac1{s_n}\sum X_i\xrightarrow{d}\mathcal N(0,1)$ **当且仅当** Lindeberg 条件成立。也就是说，正文里我们说「Lindeberg $\Rightarrow$ Feller 条件」只是其一半；Feller 补上的另一半是：在渐近可忽略的前提下，CLT 一旦成立，Lindeberg 条件就**必定**成立——这把「无个体主导」从一个方便的充分条件，提升为「i.i.d. 之外 CLT 成立的精确刻画」。

Lindeberg 条件常不好直接验证，于是有一个**更强但更好查**的充分条件：

> **定理（Lyapunov CLT）。** 若存在 $\delta>0$ 使
> $$\frac1{s_n^{2+\delta}}\sum_{i=1}^n\mathbb E\big[|X_i|^{2+\delta}\big]\xrightarrow{n\to\infty}0\qquad(\text{Lyapunov 条件}),$$
> 则同样的结论成立。

Lyapunov 条件（典型取 $\delta=1$，用三阶绝对矩）蕴含 Lindeberg 条件，但只需算一个「高阶矩之和 / 总方差的高次幂」的比值，实践中好用。它直白地说：**只要三阶矩之和相对总方差不爆，CLT 成立**。

> **与你的研究的连接（不等精度测量的汇总）**：在计算结构生物学里你常做「加权平均」。比如把同一个量（某个二面角的偏好、某对残基的接触概率、多次 MD 运行各自的自由能估计）从**不同来源、不同噪声水平**汇总成一个估计：$\hat\theta=\sum_i w_i Y_i$。各 $Y_i$ 方差不同（测量精度不同），不是 i.i.d.。Lindeberg/Lyapunov CLT 告诉你：**只要没有某一次测量的权重 × 噪声压倒所有其他项**，汇总估计的误差仍近似高斯，你照样能画高斯误差棒、给置信区间。反过来——若某一项主导（例如一次低温 MD 给出方差极大的自由能估计、却被赋了大权重），高斯近似失效，置信区间不可信。这条「无个体主导」是你信任任何加权汇总的隐形前提。

### 6.3.3 多元 CLT

把 CLT 抬到向量随机变量（第 5 章），结论几乎一字不变，只是高斯换成**多元高斯**、方差换成**协方差矩阵**。

> **定理（多元中心极限定理，multivariate CLT）。** 设 $\mathbf X_1,\mathbf X_2,\dots\in\mathbb R^d$ 为 i.i.d. 随机向量，均值 $\boldsymbol\mu=\mathbb E[\mathbf X_1]$、协方差 $\boldsymbol\Sigma=\mathrm{Cov}(\mathbf X_1)$（各分量方差有限）。则
> $$\sqrt n(\bar{\mathbf X}_n-\boldsymbol\mu)\xrightarrow{d}\mathcal N_d(\mathbf 0,\boldsymbol\Sigma).$$

证明的精巧之处是**化多为一**：用 **Cramér–Wold 器件（Cramér–Wold device）**——「$\sqrt n(\bar{\mathbf X}_n-\boldsymbol\mu)\xrightarrow{d}\mathcal N_d(\mathbf 0,\boldsymbol\Sigma)$」当且仅当对**每一个固定方向** $\mathbf a\in\mathbb R^d$，一维投影 $\mathbf a^\top\sqrt n(\bar{\mathbf X}_n-\boldsymbol\mu)\xrightarrow{d}\mathcal N(0,\mathbf a^\top\boldsymbol\Sigma\mathbf a)$。而后者是**一维 CLT** 套在标量 i.i.d. 序列 $\mathbf a^\top\mathbf X_i$ 上的直接结果。一句话：**向量的依分布收敛 = 它所有一维投影的依分布收敛**——这把多元问题彻底归约成已解决的一元问题，是高维概率里反复出现的招数（与第 5 章「多元高斯的任意线性组合仍是高斯」同根）。

多元 CLT 是后续一切「多参数估计量渐近正态」的母定理：极大似然估计向量的渐近正态（第 7 章，协方差是 Fisher 信息逆）、多元 Delta 方法（§6.4）、卡方检验统计量的极限分布（第 9 章，由多元高斯的平方和给出卡方），全都从它生发。

### 6.3.4 收敛速率：Berry–Esseen 定理，以及「$n$ 要多大」

CLT 只说「$n\to\infty$ 时收敛」，它**没说收敛多快**。实践中你永远用**有限的** $n$，于是真问题是：「用高斯近似我的样本均值，误差有多大？$n$ 要多大才够？」答案来自一个常被入门课跳过、却极其重要的定理。

> **定理（Berry–Esseen）。** 设 $X_1,\dots,X_n$ i.i.d.，$\mathbb E[X_i]=\mu$、$\mathrm{Var}=\sigma^2$、$\rho=\mathbb E|X_i-\mu|^3<\infty$（三阶绝对中心矩有限）。记 $Z_n=\frac{\bar X_n-\mu}{\sigma/\sqrt n}$、$\Phi$ 为标准正态 CDF。则存在绝对常数 $C$（已知 $C<0.4748$）使
> $$\sup_x\big|\,\mathbb P(Z_n\le x)-\Phi(x)\,\big|\le\frac{C\,\rho}{\sigma^3\sqrt n}.$$

读懂它的三层含义：

1. **收敛速率是 $O(1/\sqrt n)$**。CLT 的近似误差（用 CDF 的上确界范数即 Kolmogorov 距离衡量）以 $1/\sqrt n$ 速率消失——这是一般情况下**不可改进**的速率。

2. **误差由三阶绝对矩 $\rho/\sigma^3$ 控制**。决定常数的是 $\rho/\sigma^3=\mathbb E|X-\mu|^3/\sigma^3$，它是**绝对偏度**的量级（第 3 章 §3.2.4）。**注意它不等于带符号的偏度** $\gamma=\mathbb E[(X-\mu)^3]/\sigma^3$：对称分布有 $\gamma=0$，但 $\rho/\sigma^3>0$ 依旧非零，所以 Berry–Esseen 界对对称分布也是非平凡的。一般规律：**分布越偏斜、越重尾，$\rho/\sigma^3$ 越大，达到给定近似精度所需的 $n$ 越大。** 对称分布（如均匀、对称三角）的高斯近似实际收敛更快，但要把「为什么更快」说精确：Berry–Esseen 给的 Kolmogorov 距离上界 $\frac{C\rho}{\sigma^3\sqrt n}$ 本身**仍是 $\Theta(1/\sqrt n)$ 量级**（这个 $1/\sqrt n$ 速率一般不可改进，见第 1 点），但更细的 **Edgeworth 展开**里那个**正比于带符号偏度 $\gamma$ 的 $1/\sqrt n$ 领头修正项**，会因 $\gamma=0$（奇阶矩抵消）而**整项消失**——于是对称分布的领头近似误差降到 $O(1/n)$。强偏斜分布（如 $\lambda$ 小的指数、$p$ 极端的二项）则 $\gamma$ 大、$1/\sqrt n$ 修正项不消失，收敛很慢。

3. **「$n>30$ 就够」是民间传说，不是定理**。这条流传甚广的经验法则**对偏斜或重尾分布是错的**。一个被赋了重权的指数分布、一个 $p=0.01$ 的二项分布，可能 $n$ 要到几百上千高斯近似才靠谱。Berry–Esseen 把这件事量化了：你该看的不是「$n$ 是否大于某个魔数」，而是 $\frac{\rho}{\sigma^3\sqrt n}$ 是否足够小。

> **诚实的边界（CLT 管中间、不管极端尾部）**：Berry–Esseen 衡量的是**整体**（上确界范数）的近似误差，它在分布**中心**附近最准。但你做假设检验时关心的常是**极端尾部**（比如 $p$ 值 $=10^{-6}$ 处）。在尾部，高斯近似的**相对误差**可能巨大——真实尾概率与高斯尾概率之比可以差好几个数量级（这是「大偏差（large deviations）」的领域，CLT 的「中等偏差」精度管不到那里）。后果很实在：**用 CLT/高斯近似算出来的极小 $p$ 值往往不可信**，尾部该用精确分布、置换检验（permutation test）或大偏差/鞍点近似（saddlepoint approximation）。第 9 章讲多重检验时这一点会再次咬人——当你在全基因组上做百万次检验、关心的阈值在 $10^{-8}$ 量级时，高斯尾近似的偏差足以制造或淹没「显著」。**CLT 给你钟的腰，不给你钟的脚趾。**

> **与你的研究的连接（有效样本数 $N_{\rm eff}$ 才是 CLT 里真正的 $n$）**：CLT 的所有公式里的 $n$ 都假设样本**独立**。但折叠研究的两大数据源——MSA 和 MD/MCMC 轨迹——样本都**相关**。(1) MSA：序列因系统发育而相关（信息论 ch2 §2.4.3 的马尔可夫链结构、本课 §6.2.3），所以共进化统计量的「有效样本数」$N_{\rm eff}$ 远小于序列条数，CLT 里的 $\sqrt n$ 应换成 $\sqrt{N_{\rm eff}}$，误差棒据此放宽。(2) MD/MCMC 轨迹：相邻快照高度自相关。这里要先把「相关时间」的两种约定钉死，否则会差一个 $\sqrt2$ 因子（这是计算化学统计里反复出现的混淆源）：
>
> - **集成自相关时间** $\tau_{\rm int}:=1+2\sum_{k\ge1}\rho_k$（$\rho_k$ 是滞后 $k$ 的自相关），它**已经含了那个因子 2**。用它时，有效样本数与标准误是
>   $$N_{\rm eff}=\frac{L}{\tau_{\rm int}},\qquad \mathrm{se}(\bar Y)=\sigma\sqrt{\frac{\tau_{\rm int}}{L}}.$$
> - **指数（弛豫）相关时间** $\tau_{\rm exp}$（$\rho_k\approx\phi^k=e^{-k/\tau_{\rm exp}}$）。对几何衰减的相关，$\tau_{\rm int}\approx 2\tau_{\rm exp}$，于是也可写成 $N_{\rm eff}\approx L/(2\tau_{\rm exp})$、$\mathrm{se}=\sigma\sqrt{2\tau_{\rm exp}/L}$（这正是本课 ch3 §3.6.4 用的 $n_{\rm eff}\approx n/(2\tau)$ 约定，其中 $\tau=\tau_{\rm exp}$）。
>
> 两套写法在 $\tau_{\rm int}=2\tau_{\rm exp}$ 下**完全等价**——关键是别把两个 $\tau$ 混用。无论用哪套，结论都一样：时间平均的标准误是 $\sigma/\sqrt{N_{\rm eff}}$ 而非 $\sigma/\sqrt L$，天真地用 $\sqrt L$ 会把误差棒画得**虚假地窄**，这是计算化学里最常见的统计错误之一（第 10 章 §「自相关与有效样本数」会沿用同一约定正式推导 $\tau$ 与 $N_{\rm eff}$）。**练习 4 用的就是集成自相关时间 $\tau_{\rm int}=(1+\phi)/(1-\phi)$、$N_{\rm eff}=L/\tau_{\rm int}$。** 结论：**先估有效样本数，再代进 CLT**。这一句话，能救你一篇论文的统计审稿。

> **与你的研究的连接（pLDDT 校准 = 一个大样本频率问题）**：AlphaFold 的 pLDDT、PAE 这类「置信分数」要可信，关键看它**校准（calibration）**得好不好——即「模型说置信约 $p$ 的那些位点里，是否真的约有比例 $p$ 折叠正确」。这本质上就是本章的 **LLN + CLT**：把预测置信落在某区间（bin）的位点聚成一组，该组的**经验正确率**（一堆 0/1 指示量的平均）由 LLN 收敛到该组的**真实正确率**，而它的误差棒由（二项）CLT 给出 $\sqrt{\hat q(1-\hat q)/m}$（$m$ 是该 bin 的位点数、$\hat q$ 是经验正确率）。校准曲线画的就是「每个 bin 的经验正确率 ± CLT 误差棒」对「该 bin 的标称置信」；校准好就是这条曲线贴着对角线、且在误差棒内。于是「AlphaFold 置信分数到底可不可信」这个看似工程的问题，**精确地落在本章 LLN（经验频率收敛）+ CLT（误差棒）+ §6.4 Delta 方法（若先把原始分数非线性地映成概率，还要传一道误差）之上**。（注意：每个 bin 内的位点若来自同一结构、空间相邻，也会相关——又是 $N_{\rm eff}$ 问题，$m$ 要打折。）

> **要点**：CLT 说标准化样本均值 $\xrightarrow{d}\mathcal N(0,1)$（i.i.d. + 均值方差有限），证明就是「特征函数 $\to e^{-t^2/2}$」一个初等极限，普适性源于高斯是「求和+缩放」算子的不动点。非同分布版（Lindeberg/Lyapunov）要求「无个体主导」；多元版（Cramér–Wold 化为一维）给多元高斯。Berry–Esseen 诚实地告诉你收敛速率是 $O(\rho/(\sigma^3\sqrt n))$，由偏度控制——「$n>30$」是传说，偏斜/重尾要大得多的 $n$，且尾部 $p$ 值别信高斯。真实数据里 $n$ 应换成有效独立样本数 $N_{\rm eff}$。（§6.3）

---

## 6.4 Delta 方法：把 CLT 传给非线性函数

CLT 给的是**样本均值** $\bar X_n$ 的渐近正态。但你真正想要标准误的，常是 $\bar X_n$ 的某个**非线性函数** $g(\bar X_n)$——比如把均值取对数、取倒数、做比值、过 logit。直接对 $g(\bar X_n)$ 写 CLT 不成立（$g(\bar X_n)$ 不是独立项之和）。**Delta 方法（delta method）** 就是解决这个问题的标准工具：用一阶 Taylor 展开把非线性「线性化」，再把 CLT 借过来。它是第 7 章求 MLE 标准误、第 9 章构造置信区间的日常主力。

### 6.4.1 一阶 Delta 方法

> **定理（一阶 Delta 方法）。** 设 $\sqrt n(T_n-\theta)\xrightarrow{d}\mathcal N(0,\sigma^2)$（$T_n$ 是任意渐近正态的估计量，通常就是 $\bar X_n$，$\theta=\mu$）。设 $g$ 在 $\theta$ 处可微且 $g'(\theta)\ne0$。则
> $$\sqrt n\big(g(T_n)-g(\theta)\big)\xrightarrow{d}\mathcal N\!\big(0,\ [g'(\theta)]^2\sigma^2\big).$$

**证明（一行 Taylor）。** 在 $\theta$ 处一阶 Taylor：$g(T_n)=g(\theta)+g'(\theta)(T_n-\theta)+R_n$，余项 $R_n=o_P(T_n-\theta)$（可微性给出：$R_n/(T_n-\theta)\xrightarrow{P}0$）。要点是把余项乘 $\sqrt n$ 后可忽略这一步写清楚：
$$\sqrt n\,R_n=\underbrace{\big[\sqrt n(T_n-\theta)\big]}_{=O_P(1)\ (\text{渐近正态})}\cdot\underbrace{\big[R_n/(T_n-\theta)\big]}_{=o_P(1)\ (\text{可微性余项})}=O_P(1)\cdot o_P(1)=o_P(1),$$
即 $\sqrt n\,R_n\xrightarrow{P}0$，由 Slutsky 定理（见下）不影响极限分布。于是
$$\sqrt n\big(g(T_n)-g(\theta)\big)\approx g'(\theta)\cdot\sqrt n(T_n-\theta)\xrightarrow{d}g'(\theta)\cdot\mathcal N(0,\sigma^2)=\mathcal N\big(0,[g'(\theta)]^2\sigma^2\big).\qquad\blacksquare$$

直觉一句话：**把非线性函数在估计点处当成一条直线，斜率是 $g'(\theta)$；CLT 的高斯涨落经过这条直线被放大 $|g'(\theta)|$ 倍**（标准差乘 $|g'|$，方差乘 $g'^2$，这正是第 2/3 章「常数倍随机变量方差乘平方」的延续）。所以 $g(\bar X_n)$ 的渐近标准误是 $|g'(\hat\theta)|\cdot\sigma/\sqrt n$——把这条记牢，你就掌握了 90% 的「估计量函数的标准误」怎么算。

证明里用到的 **Slutsky 定理（Slutsky's theorem）** 值得单列，它是 CLT 实战中的胶水：若 $Y_n\xrightarrow{d}Y$ 且 $A_n\xrightarrow{P}a$（常数），则 $A_nY_n\xrightarrow{d}aY$、$A_n+Y_n\xrightarrow{d}a+Y$。它的用处是「把依概率收敛到常数的扰动项无害地吸收掉」——比如把未知 $\sigma$ 换成一致估计 $\hat\sigma$（$\hat\sigma/\sigma\xrightarrow{P}1$）不改变极限分布，这就是为什么你能在置信区间里**用估计的标准差代替真标准差**。

**例（对数变换稳定比例的方差）。** 设 $\hat p$ 是某事件频率，$\sqrt n(\hat p-p)\xrightarrow{d}\mathcal N(0,p(1-p))$（伯努利的 CLT）。取 $g(p)=\log\frac{p}{1-p}$（logit），$g'(p)=\frac1{p(1-p)}$。Delta 方法给
$$\sqrt n\big(\mathrm{logit}(\hat p)-\mathrm{logit}(p)\big)\xrightarrow{d}\mathcal N\Big(0,\ \tfrac1{[p(1-p)]^2}\cdot p(1-p)\Big)=\mathcal N\Big(0,\ \tfrac1{p(1-p)}\Big).$$
有意思的是：原来比例 $\hat p$ 的方差 $p(1-p)$ 在 $p\to0,1$ 时趋于 0（边界处方差被压扁，正态近似在那里最差）；而 logit 尺度的方差 $\frac1{p(1-p)}$ 在边界处**变大但更稳定地正态**。这就是为什么在比例接近 0 或 1 时，**在 logit（或对数）尺度上构造置信区间、再变换回去**，比直接在原尺度上 $\hat p\pm1.96\,\mathrm{se}$ 靠谱得多（后者可能给出超过 $[0,1]$ 的荒唐区间）。这类**方差稳定变换（variance-stabilizing transformation）** 是 Delta 方法最实用的副产品。

### 6.4.2 二阶 Delta 方法：当一阶导为零

一阶 Delta 方法在 $g'(\theta)=0$ 时**失效**——线性项消失，极限退化成常数 0，需要看二次项。

> **定理（二阶 Delta 方法）。** 若 $g'(\theta)=0$ 但 $g''(\theta)\ne0$，且 $\sqrt n(T_n-\theta)\xrightarrow{d}\mathcal N(0,\sigma^2)$，则**标度变成 $n$（而非 $\sqrt n$）**：
> $$n\big(g(T_n)-g(\theta)\big)\xrightarrow{d}\frac12 g''(\theta)\,\sigma^2\,\chi^2_1.$$

这里极限不再是高斯，而是**卡方**（$\chi^2_1$ = 一个标准正态的平方，第 4 章 §4.2）。原因清楚：当一阶导为 0，二阶 Taylor 给 $g(T_n)-g(\theta)=\frac12 g''(\theta)(T_n-\theta)^2+R_n$，而 $(T_n-\theta)^2$ 经 $n$ 放大后是「高斯的平方」=卡方。三阶 Taylor 余项 $R_n=o_P((T_n-\theta)^2)$，乘 $n$ 后为 $n\cdot(T_n-\theta)^2\cdot o_P(1)=O_P(1)\cdot o_P(1)=o_P(1)$（因 $n(T_n-\theta)^2=O_P(1)$），故**不影响极限**，主导项恰为 $\frac12 g''(\theta)(T_n-\theta)^2$。注意标度从 $\sqrt n$ 升到 $n$——**收敛更快**（涨落更小），但**形状变偏斜**（卡方非对称）。

> **与你的研究的连接（标准误传播 = 实验室里的 Delta 方法）**：你在折叠/结构计算里报告的几乎每个「派生量」都需要 Delta 方法传播误差。例：(1) 从 MD 估计**结合自由能** $\Delta G=-k_BT\log K$，其中 $K$ 是某个比值/概率的估计——$\Delta G$ 的标准误就是对 $g(K)=-k_BT\log K$ 用一阶 Delta 方法，$\mathrm{se}(\Delta G)=k_BT\cdot\mathrm{se}(K)/K$。(2) 从接触频率估计**耦合强度的对数几率**、从两个能量估计算**它们的比值或差**、把 pLDDT 这类原始分数**校准成概率**再传误差——全是 Delta 方法。物理/工程里管这叫「误差传播公式（error propagation）」，$\mathrm{se}(g)\approx|g'|\cdot\mathrm{se}(x)$；它和统计学的 Delta 方法是**同一条 Taylor 展开**，只是一个从渐近分布角度、一个从微分角度叙述。多元版（下一段）让你能同时传播多个相关输入的误差，这在「多个拟合参数共同决定一个派生量」时是必需的。

**多元 Delta 方法（一句话）。** 若 $\sqrt n(\mathbf T_n-\boldsymbol\theta)\xrightarrow{d}\mathcal N_d(\mathbf 0,\boldsymbol\Sigma)$，$g:\mathbb R^d\to\mathbb R$ 可微、梯度 $\nabla g(\boldsymbol\theta)\ne\mathbf 0$，则
$$\sqrt n\big(g(\mathbf T_n)-g(\boldsymbol\theta)\big)\xrightarrow{d}\mathcal N\big(0,\ \nabla g(\boldsymbol\theta)^\top\boldsymbol\Sigma\,\nabla g(\boldsymbol\theta)\big).$$
这就是「多个相关估计量的函数」的渐近方差公式，$\nabla g^\top\boldsymbol\Sigma\nabla g$ 把输入的**协方差结构**（相关的误差会互相放大或抵消）正确地传进输出——比逐个独立传播（忽略相关）准确得多。第 7 章 MLE 的标准误、第 9 章比值/差的置信区间都靠它。

> **要点**：Delta 方法用一阶 Taylor 把非线性函数线性化，把 CLT 借给 $g(\bar X_n)$：渐近方差是 $[g'(\theta)]^2\sigma^2/n$（标准误 $\approx|g'|\cdot\mathrm{se}$）。Slutsky 定理是它的胶水（把估计的方差换进去不改变极限）。$g'=0$ 时退化为二阶版，极限变卡方、标度升到 $n$。方差稳定变换（如 logit、log）让边界附近的区间更可靠。多元版用 $\nabla g^\top\Sigma\nabla g$ 正确传播相关误差。这是第 7、9 章一切标准误/置信区间的计算引擎。（§6.4）

---

## 6.5 进阶集中不等式：指数尾是高维统计的引擎

第 3 章 §3.5 给过基础集中不等式（Markov、Chebyshev、Hoeffding、Chernoff）。本节升级到现代机器学习与高维统计真正依赖的那批工具。核心动机：CLT 是**渐近**的（$n\to\infty$），但理论分析常需要**有限样本、非渐近（non-asymptotic）** 的保证——「在这个具体的 $n$ 下，偏离超过 $t$ 的概率至多是多少」。这正是**集中不等式（concentration inequality）** 的领地。而其中最关键的结构是**sub-Gaussian 性**——「尾巴衰减得至少和高斯一样快」，它是泛化界、随机矩阵、压缩感知、$\varepsilon$-net 论证的统一引擎。

### 6.5.1 sub-Gaussian 随机变量

> **定义（sub-Gaussian 随机变量）。** 称均值为 0 的随机变量 $X$ 是 $\sigma^2$-**sub-Gaussian** 的，若其 MGF 被高斯的 MGF 控制：
> $$\mathbb E[e^{\lambda X}]\le e^{\lambda^2\sigma^2/2}\qquad\forall\lambda\in\mathbb R.$$

这个看似抽象的定义有一个直观的刻画：**尾巴指数衰减、且不慢于高斯**——$\mathbb P(|X|>t)\le 2e^{-t^2/(2\sigma^2)}$。换句话说，sub-Gaussian 就是「尾巴最多和 $\mathcal N(0,\sigma^2)$ 一样胖」的那一类随机变量。需要点明的是：这两个刻画**在相差一个普适常数因子的意义下等价**——MGF 界 $\Rightarrow$ 尾界是 Chernoff 一行（取 $\lambda=t/\sigma^2$）；反向（尾界 $\Rightarrow$ MGF 界）也成立，但参数 $\sigma^2$ 可能相差一个绝对常数倍，并非严格守恒同一个 $\sigma^2$。这正是高维概率里「sub-Gaussian 范数」那几个等价定义的来源——它们都「等价到普适常数」，但常数不必相同。哪些是 sub-Gaussian？

- **任何有界变量**：若 $X\in[a,b]$（中心化后），则它是 $\frac{(b-a)^2}{4}$-sub-Gaussian（这是 Hoeffding 引理，第 3 章 §3.5.7 用过）。所以伯努利、有界的指示函数、归一化的特征都自动 sub-Gaussian。
- **高斯本身**：$\mathcal N(0,\sigma^2)$ 是 $\sigma^2$-sub-Gaussian（取等号）。
- **不是 sub-Gaussian 的**：指数分布、泊松（尾巴只是指数衰减但比高斯胖，属于更宽的 **sub-exponential** 类）、任何重尾分布。

sub-Gaussian 类有一条极好用的封闭性：**独立 sub-Gaussian 之和仍 sub-Gaussian，参数相加**。若 $X_i$ 独立、$\sigma_i^2$-sub-Gaussian，则 $\sum X_i$ 是 $(\sum\sigma_i^2)$-sub-Gaussian，于是
$$\mathbb P\Big(\Big|\sum_{i=1}^n X_i\Big|>t\Big)\le 2\exp\!\Big(-\frac{t^2}{2\sum\sigma_i^2}\Big).$$
这一条直接重新证出 Hoeffding 不等式，并把它推广到任何 sub-Gaussian 项。**指数尾 $e^{-t^2/(2\sigma^2)}$ 才是高维统计的引擎**：它让你能对**很多个**事件同时用 union bound（第 3 章 §3.5.5）而代价只是对数级——$M$ 个 sub-Gaussian 量同时越界的概率 $\le 2M e^{-t^2/(2\sigma^2)}$，要把它压到 $\delta$ 只需 $t=\sigma\sqrt{2\log(2M/\delta)}$，**阈值只随 $\log M$ 增长**。这就是为什么高维统计能在「变量数 $M$ 远大于样本数 $n$」时仍给出有意义的界——指数尾让你「便宜地」控制指数级多的事件。

### 6.5.2 Bernstein 不等式：用上方差，对低概率事件更紧

Hoeffding 只用「有界」，没用「方差」。当变量有界**且方差小**时，Hoeffding 太松——它假设最坏情况的方差。**Bernstein 不等式**把方差用进来，对「事件罕见（小方差）」的情形给出紧得多的界。

> **定理（Bernstein 不等式）。** 设 $X_1,\dots,X_n$ 独立、均值 0、$|X_i|\le M$、$\frac1n\sum\mathrm{Var}(X_i)=\sigma^2$。则
> $$\mathbb P\Big(\frac1n\sum_{i=1}^n X_i>t\Big)\le\exp\!\Big(-\frac{n t^2}{2\sigma^2+\tfrac23 M t}\Big).$$

上式是**单边**界（对偏离的一侧）；要双边 $\mathbb P(|\frac1n\sum X_i|>t)$ 时右端**乘 2**。另外注意**比较口径**：这条 Bernstein 是对**均值** $\frac1n\sum X_i$ 偏离 $t$ 写的，而 §6.5.1 的 sub-Gaussian 界 $2\exp(-t^2/(2\sum\sigma_i^2))$ 是对**和** $\sum X_i$ 偏离 $t$ 写的——拿两条界比紧度时，先统一成同一对象（如都换成均值），别把「和的 $t$」与「均值的 $t$」直接相减。

读这个分母（都站在「均值偏离 $t$」的口径下看）：**当 $t$ 小（关心适中偏离）时，$2\sigma^2$ 主导，指数是 $-\frac{nt^2}{2\sigma^2}$**——这是「高斯型/方差型」尾，比同口径下 Hoeffding 的 $-\frac{2nt^2}{(2M)^2}=-\frac{nt^2}{2M^2}$ 紧（因为通常 $\sigma^2\ll M^2$）。**当 $t$ 大时，$\frac23Mt$ 主导，指数是 $-\frac{3nt}{2M}$**——退化为「指数型/有界型」尾。Bernstein 漂亮地在两个 regime 间插值：小偏离吃方差红利（像 CLT 的高斯），大偏离吃有界保护（像 Hoeffding）。**对稀有事件（小 $\sigma^2$）的频率估计，Bernstein 常比 Hoeffding 紧一两个数量级**——这在估计「某罕见构象的出现概率」「某低频突变的频率」时直接节省大量样本。

### 6.5.3 McDiarmid（有界差分）不等式：超越「均值」的集中

前面的不等式都管「独立项之和」。但机器学习里你关心的常不是简单和，而是样本的**复杂函数**——泛化误差、经验风险的上确界、$U$-统计量、图的某个性质。McDiarmid 不等式（也叫**有界差分不等式，bounded differences inequality**）把集中推广到**任意满足「改一个输入只能小幅改变输出」的函数**。

> **定理（McDiarmid / 有界差分不等式）。** 设 $X_1,\dots,X_n$ 独立，函数 $f(x_1,\dots,x_n)$ 满足**有界差分条件**：对每个 $i$，固定其余坐标、只改第 $i$ 个坐标，函数变化不超过 $c_i$：
> $$\sup_{x_1,\dots,x_n,x_i'}\big|f(\dots,x_i,\dots)-f(\dots,x_i',\dots)\big|\le c_i.$$
> 则（单边）
> $$\mathbb P\big(f-\mathbb E[f]\ge t\big)\le\exp\!\Big(-\frac{2t^2}{\sum_{i=1}^n c_i^2}\Big).$$
> 同样的界对 $\mathbb P(f-\mathbb E[f]\le-t)$ 也成立（对 $-f$ 重复论证）；要**双边** $\mathbb P(|f-\mathbb E[f]|\ge t)$ 时右端**乘 2**。

威力在于 $f$ **可以是任意复杂的函数**，只要它「不被单个样本绑架」（改一个样本只能小幅扰动它）。Hoeffding 是它在 $f=\frac1n\sum x_i$（此时 $c_i=(b-a)/n$）下的特例。McDiarmid 的杀手级应用是**统计学习理论的泛化界**：把 $f$ 取成「训练误差与测试误差之差的上确界」$\sup_{h\in\mathcal H}|\hat R(h)-R(h)|$，改一个训练样本最多改变它 $1/n$ 量级，于是 McDiarmid 给出「这个上确界以高概率接近其期望」，再配合期望的对称化/Rademacher 复杂度界，就得到「**训练误差 ≈ 测试误差**，误差 $O(\sqrt{(\log|\mathcal H|)/n})$」的经典泛化保证。**指数尾 + 有界差分 = 现代泛化理论的两大支柱之一**。

> **与你的研究的连接（深度学习折叠模型为何能泛化、采样估计要多少样本）**：(1) AlphaFold、ProteinMPNN、RFdiffusion 这类模型「在训练蛋白上学到的、能迁移到没见过的蛋白」——这种泛化能力的理论刻画，骨架正是 McDiarmid + 复杂度界（折叠课程第 6/7 章的 ML 视角）。改动训练集中一条蛋白对学到的模型只有有限影响（有界差分），是泛化界成立的前提。(2) 更直接地，sub-Gaussian/Bernstein 回答「我做多少次 MD 采样 / 多少条 MCMC 样本，才能把某个观测量的估计误差以 $95\%$ 把握压到阈值 $t$」——这是第 3 章 §3.6.4「要多少独立采样」的延续，但 Bernstein 让你在「目标事件稀有（小方差）」时用**少得多**的样本达到同样精度。记住：所有这些界里的 $n$ 仍须是**有效独立**样本数（§6.3.4），相关样本要先打折。（3) McDiarmid 还直接给出「交叉验证误差作为复杂统计量也集中」的保证，让你敢用 CV 估计来选模型。

> **要点**：sub-Gaussian = 尾巴不胖于高斯（$\mathbb E e^{\lambda X}\le e^{\lambda^2\sigma^2/2}$，等价于 $e^{-t^2/2\sigma^2}$ 尾），有界变量与高斯都属此类，独立和参数相加——指数尾让你用 union bound 廉价控制指数级多事件（阈值随 $\log M$ 增长），这是高维统计的引擎。Bernstein 把方差用进来，在「小偏离吃方差、大偏离吃有界」间插值，对稀有事件远紧于 Hoeffding。McDiarmid（有界差分）把集中推广到任意「不被单样本绑架」的复杂函数，是统计学习泛化界的支柱。（§6.5）

---

## 6.6 重采样预告：Bootstrap 的思想

本章最后埋一颗在第 9 章引爆的种子。CLT + Delta 方法给的是估计量的**渐近**抽样分布（高斯），但有两类情形它们不够：(1) 统计量太复杂、$g$ 难求导（中位数、相关系数、各种排序统计量），Delta 方法算不动；(2) $n$ 不够大，高斯近似不准（Berry–Esseen 警告过的偏斜/重尾情形）。**自助法（bootstrap）** 用一个简单到近乎作弊的想法绕过这两难。

核心思想一句话：**既然真分布 $F$ 未知、但经验分布 $\hat F_n$ 由 Glivenko–Cantelli（§6.2.3）保证逼近 $F$，那就用 $\hat F_n$ 冒充 $F$，从它里反复重抽样，直接「实验」出统计量的抽样分布。**

具体地，估计量 $\hat\theta=T(X_1,\dots,X_n)$ 的抽样分布想知道却看不见（你只有一份样本）。Bootstrap 的做法：

```
原始样本  {X_1,...,X_n}  ──(从经验分布 F̂_n 有放回抽样 n 个)──►  自助样本 {X*_1,...,X*_n}
        重复 B 次（如 B=2000）
        ┌──────────────────────────────────────────────────────────┐
        │  每个自助样本算一次统计量  θ̂*^(1), θ̂*^(2), ..., θ̂*^(B)    │
        └──────────────────────────────────────────────────────────┘
        这 B 个 θ̂* 的经验分布   ≈   θ̂ 的真实抽样分布
        ↓                              ↓
   它们的标准差 = θ̂ 的标准误      分位数 = θ̂ 的置信区间
```

为什么有效？因为「有放回从 $\hat F_n$ 抽 $n$ 个」模拟了「从 $F$ 抽 $n$ 个」这件你本无法重复的事（这一步默认样本 **i.i.d.**——正是 §6.2.3/§6.3.4 反复打折的那个假设：相关数据下 $\hat F_n$ 仍逼近**边缘**分布 $F$，但样本间的依赖结构没了，抽样分布的方差会被**低估**，故需 block bootstrap，见下文「诚实的边界」）——你用**计算**换取了本需**更多数据**才能得到的抽样分布。它把「抽样分布」这个抽象对象变成一个**可以蒙特卡洛出来的直方图**，因而能处理任何复杂统计量、不需要任何求导或高斯假设。这是 CLT/Delta 方法（解析、渐近）的**计算式补充**：前者快、有公式、需大 $n$ 与可微性；后者慢（要重抽 $B$ 次）、无公式、但通用且对中等 $n$ 更稳。

这里只点到思想；置信区间的具体构造（百分位法、BCa 法）、何时 bootstrap 会失效（极值统计量、相关数据需 block bootstrap）、与置换检验的关系，全部留到第 9 章正式展开。

> **诚实的边界（bootstrap 不是魔法）**：bootstrap 默认样本 **i.i.d.**——对相关数据（MD 轨迹、系统发育相关的 MSA）直接 bootstrap 会**严重低估**误差（又是「有效样本数」问题！）。相关数据需要 **block bootstrap**（按相关长度成块重抽，第 9/10 章）。此外，bootstrap 对依赖**极端尾**的统计量（如样本最大值）系统性失败，因为经验分布在尾部根本没有 $F$ 的信息（你没抽到的极端值，重抽样里也变不出来）。它逼近的是 $F$ 在「你采样到的区域」的样子——这与 Glivenko–Cantelli 的「一致收敛」并不矛盾：收敛快慢与统计量对尾部的敏感度是两回事。

> **要点**：Bootstrap 用经验分布 $\hat F_n$（由 Glivenko–Cantelli 保证逼近真分布）替代未知真分布，有放回重抽样 $B$ 次、每次重算统计量，用这 $B$ 个值的经验分布近似真实抽样分布——从而对**任意复杂统计量**用计算换出标准误与置信区间，无需求导或高斯假设。它是 CLT/Delta（解析、渐近）的计算式补充。但默认 i.i.d.，相关数据须用 block bootstrap，极值统计量会失败。正式用法见第 9 章。（§6.6）

---

## 6.7 本章小结

- **四种收敛模式**按强弱排成「几乎必然 与 $L^p$ 并立 ⟹ 依概率 ⟹ 依分布」，唯一逆向特例是「依分布到常数 ⟹ 依概率」；a.s. 与 $L^p$ 互不蕴含，由滑动窗口（怕无穷多次越界）与高尖峰（怕罕见巨值）两反例见证。（§6.1）
- **弱大数定律**（依概率）只需「不相关 + 方差有限」，切比雪夫两行即得，并揭示样本均值标准差以 $\sigma/\sqrt n$ 缩小——这是一切误差棒里 $1/\sqrt n$ 的出处；**强大数定律**（几乎必然）需 i.i.d. + $\mathbb E|X|<\infty$，保证单条轨迹的时间平均收敛（蒙特卡洛/遍历性的根基）。（§6.2）
- **Glivenko–Cantelli** 把 LLN 从「均值收敛」升级到「经验 CDF 一致收敛」，是频率派统计与 bootstrap 的根基；柯西（无一阶矩）是 LLN 的反例。把 LLN 套在自信息 $-\log p(X)$ 上即得信息论的**渐近均分性（AEP）**——「样本熵依概率收敛到熵 $H$」，是 LLN 的信息论化身（信息论 ch5）。（§6.2）
- **中心极限定理**：标准化样本均值 $\xrightarrow{d}\mathcal N(0,1)$（i.i.d. + 均值方差有限），证明是「特征函数 $\to e^{-t^2/2}$」一个初等极限，普适性源于高斯是「求和+缩放」算子的不动点（方差无限时吸引子换成 $\alpha$-稳定分布）。（§6.3.1）
- 非同分布的 **Lindeberg/Lyapunov CLT** 要求「无个体主导」；**多元 CLT** 经 Cramér–Wold 化为一维投影，给出多元高斯，是一切多参数估计量渐近正态的母定理。（§6.3.2–3）
- **Berry–Esseen** 给出收敛速率 $O(\rho/(\sigma^3\sqrt n))$，由三阶绝对矩 $\rho/\sigma^3$（绝对偏度量级，不等于带符号偏度 $\gamma$）控制——「$n>30$」是民间传说，偏斜/重尾要大得多的 $n$（对称分布因 Edgeworth 的 $\gamma\propto1/\sqrt n$ 项消失而快到 $O(1/n)$）；且 CLT 管钟的腰不管脚趾，极端尾部 $p$ 值别信高斯近似。真实数据里 $n$ 应换成有效独立样本数 $N_{\rm eff}$。（§6.3.4）
- **Delta 方法**用一阶 Taylor 把 CLT 借给非线性函数 $g(\bar X_n)$，渐近标准误 $\approx|g'|\cdot\mathrm{se}$；Slutsky 定理是其胶水；$g'=0$ 时退化为二阶版（极限变卡方）；多元版 $\nabla g^\top\Sigma\nabla g$ 正确传播相关误差。这是第 7、9 章一切标准误的引擎。（§6.4）
- **进阶集中不等式**：sub-Gaussian（尾不胖于高斯）的指数尾让 union bound 廉价控制指数级多事件（阈值随 $\log M$ 增长），是高维统计的引擎；Bernstein 用方差在「小偏离/大偏离」间插值，对稀有事件远紧于 Hoeffding；McDiarmid（有界差分）把集中推广到任意「不被单样本绑架」的复杂函数，是泛化界的支柱。（§6.5）
- **Bootstrap** 用经验分布替代真分布、重抽样 $B$ 次，对任意复杂统计量计算出抽样分布/标准误/置信区间，是 CLT/Delta 的计算式补充；但默认 i.i.d.，相关数据须 block bootstrap，极值统计量会失败。（§6.6）

---

## 6.8 动手 / 思考小练习

**1（动手·编程，亲眼看 CLT 的普适性与速率）。** 仅用 `numpy/scipy`：取三个形状迥异的基分布——(i) 均匀 $U(0,1)$（对称、轻尾）、(ii) 指数 $\mathrm{Exp}(1)$（强右偏）、(iii) 一个双峰分布（如 $0.5\mathcal N(-2,0.5)+0.5\mathcal N(2,0.5)$）。对每个分布与每个 $n\in\{1,2,5,30,100\}$，抽 $M=20000$ 组、每组 $n$ 个样本，算 $M$ 个标准化样本均值 $Z=\frac{\bar X-\mu}{\sigma/\sqrt n}$，画直方图叠加 $\mathcal N(0,1)$ 的 pdf。再对每个 $(分布, n)$ 用 KS 距离 `scipy.stats.kstest(Z, 'norm')` 量化与高斯的差距。观察：哪个分布收敛最慢？它与 §6.3.4 的偏度预测一致吗？（提示：指数分布偏度大，收敛最慢，$n=30$ 时直方图仍明显右偏，KS 距离明显大于均匀分布的；这正是 Berry–Esseen 里 $\rho/\sigma^3$ 大的后果——「$n=30$」对它远远不够。）

**2（思考·辨析，收敛模式的强弱）。** (a) 举例：一个序列 $X_n\xrightarrow{P}0$ 但**不**几乎必然收敛。(b) 举例：一个序列 $X_n\xrightarrow{a.s.}0$ 但**不** $L^1$ 收敛。(c) 判断真假并说明：「若 $X_n\xrightarrow{d}X$ 且 $X$ 是常数 $c$，则 $X_n\xrightarrow{P}c$。」（提示：(a) §6.1.3 滑动窗口 $\mathbf 1_{[j/2^k,(j+1)/2^k]}$；(b) §6.1.3 高尖峰 $n\mathbf 1_{[0,1/n]}$；(c) 真——这是唯一的逆向蕴含，因为收敛到常数时「分布贴近」与「取值贴近」等价，可直接验证 $\mathbb P(|X_n-c|>\varepsilon)\to0$。）

**3（动手·编程，Delta 方法 vs 蒙特卡洛真值，误差传播）。** 仅用 `numpy`：设 $\hat p$ 是 $n=200$ 次伯努利试验（真 $p=0.1$）的频率，关心派生量 $\hat\theta=\log\frac{\hat p}{1-\hat p}$（logit）。(a) 用 §6.4.1 的 Delta 方法解析地给出 $\hat\theta$ 的渐近方差 $\frac1{np(1-p)}$，代入数值。(b) 用蒙特卡洛：重复整个实验 $M=50000$ 次，每次算 $\hat\theta$，求其样本方差，与 (a) 比较。(c) 把 $p$ 改成 $0.01$ 重做，解释为什么 Delta 近似在 $p$ 极端时变差（提示：(a) $\frac1{200\cdot0.1\cdot0.9}\approx0.0556$；(b) 应接近 (a)，但 $n p=20$ 不算大，会略有出入；(c) $p=0.01$ 时 $np=2$，$\hat p$ 常取到 0 使 logit 发散、CLT/Delta 的高斯近似在边界处差——印证 §6.3.4 偏度与 §6.4.1 边界方差稳定变换的讨论。）

**4（动手·编程，与研究相关：有效样本数毁掉天真误差棒）。** 仅用 `numpy`：用 AR(1) 过程 $Y_t=\phi Y_{t-1}+\sqrt{1-\phi^2}\,\varepsilon_t$（$\varepsilon_t\sim\mathcal N(0,1)$，$\phi=0.95$）模拟一条**自相关**的「MD 轨迹」，长度 $L=10000$（边缘方差为 1）。(a) 算时间平均 $\bar Y$ 的天真标准误 $1/\sqrt L$。(b) 用**集成自相关时间** $\tau_{\rm int}\approx\frac{1+\phi}{1-\phi}$（注意这是含因子 2 的那种约定，故下面直接用 $L/\tau_{\rm int}$ 而非 $L/(2\tau)$，对照 §6.3.4 两套约定）估有效样本数 $N_{\rm eff}=L/\tau_{\rm int}$，给出**修正**标准误 $1/\sqrt{N_{\rm eff}}$。(c) 重复模拟整条轨迹 $M=2000$ 次，求 $\bar Y$ 的真实标准差，看它更接近 (a) 还是 (b)。（提示：$\phi=0.95$ 时 $\tau_{\rm int}\approx39$，$N_{\rm eff}\approx256$，修正标准误约 $0.0625$ 是天真值 $0.01$ 的 6 倍多；蒙特卡洛真值会贴近 (b)。这就是 §6.3.4 警告的「相关样本让 $\sqrt L$ 把误差棒画得虚假地窄」——做 MD/MCMC 统计前必须估 $N_{\rm eff}$。）

**5（思考·证明，二阶 Delta 方法为什么是卡方）。** 设 $\sqrt n(T_n-\theta)\xrightarrow{d}\mathcal N(0,\sigma^2)$，$g'(\theta)=0$、$g''(\theta)\ne0$。用二阶 Taylor $g(T_n)-g(\theta)\approx\frac12 g''(\theta)(T_n-\theta)^2$ 论证 $n(g(T_n)-g(\theta))\xrightarrow{d}\frac12 g''(\theta)\sigma^2\chi^2_1$，并解释为什么标度从 $\sqrt n$ 升到了 $n$。（提示：令 $W_n=\sqrt n(T_n-\theta)/\sigma\xrightarrow{d}\mathcal N(0,1)$，则 $(T_n-\theta)^2=\sigma^2 W_n^2/n$，故 $n\cdot\frac12 g''\sigma^2 W_n^2/n=\frac12 g''\sigma^2 W_n^2\xrightarrow{d}\frac12 g''\sigma^2\chi^2_1$，用连续映射定理 $W_n^2\xrightarrow{d}\chi^2_1$；标度升高是因为一阶项消失后，主导项是二次的、塌缩更快，需更大放大率才能看到非退化极限。）

**6（思考·辨析，与研究相关：CLT 误用清单）。** 判断下列做法对错并说明：(a) 「我有 50000 条 MSA 序列，所以频率估计的高斯误差棒里 $n=50000$」。(b) 「我做了全基因组 $10^6$ 次检验，最小 $p$ 值是 $3\times10^{-9}$，是用样本均值的高斯近似算的，所以这个关联极显著」。(c) 「我的统计量是 200 个有界量之和，$n=200>30$，所以一定能用高斯近似」。（提示：(a) 错——序列因系统发育相关，应用 $N_{\rm eff}\ll50000$（§6.2.3、§6.3.4）；(b) 错——CLT 在极端尾部不可靠（Berry–Esseen 只管整体/中心，大偏差区高斯相对误差可达数量级，§6.3.4 诚实边界），如此小的 $p$ 应用精确/置换检验；(c) 不一定——若这些有界量强偏斜（如几乎总是 0、偶尔很大），偏度大，$n=200$ 可能仍不够，要看 Berry–Esseen 的 $\rho/\sigma^3$ 而非魔数 30。）

---

## 6.9 深入阅读指引

**标准概率/统计教材（收敛、LLN、CLT 的严格来源）**
- R. Durrett, *Probability: Theory and Examples*（5th ed., Cambridge, 2019）。—— 收敛模式、弱/强大数定律、特征函数与 CLT（含 Lindeberg–Feller）的严格标准来源；§6.1–6.3 想看完整证明与测度论细节就查它。
- A. W. van der Vaart, *Asymptotic Statistics*（Cambridge, 1998）。—— 把依分布收敛、Slutsky、Delta 方法、Cramér–Wold 写成统计学家最常用的形式；§6.3.3–6.4 的标准参考，第 7、9 章求渐近分布也反复用它。
- P. Billingsley, *Convergence of Probability Measures*（2nd ed., Wiley, 1999）。—— 弱收敛理论的经典专著，想真正理解「依分布收敛 = 弱收敛」的拓扑学根基（§6.1.1 依分布收敛/弱收敛的定义）读它。

**集中不等式与高维统计（§6.5 的现代引擎）**
- S. Boucheron, G. Lugosi & P. Massart, *Concentration Inequalities: A Nonasymptotic Theory of Independence*（Oxford, 2013）。—— sub-Gaussian、Bernstein、McDiarmid 的现代权威，§6.5 的母书，理解「为什么指数尾撑起高维统计」。
- R. Vershynin, *High-Dimensional Probability: An Introduction with Applications in Data Science*（Cambridge, 2018）。—— 把 sub-Gaussian/sub-exponential 讲得最直观、最贴近机器学习，§6.5 想看应用（随机矩阵、压缩感知）读它。

**收敛速率与渐近展开（§6.3.4 的深化）**
- W. Feller, *An Introduction to Probability Theory and Its Applications, Vol. II*（2nd ed., Wiley, 1971）。—— Berry–Esseen 与 Edgeworth 展开的经典出处，想搞清「CLT 近似的下一阶修正长什么样、偏度如何进入」读它。

**与蛋白质/计算生物学研究的桥梁**
- D. Frenkel & B. Smit, *Understanding Molecular Simulation*（2nd ed., Academic Press, 2002），关于「误差估计、自相关时间、block averaging」的章节。—— 把 §6.3.4 的「有效样本数」落到 MD/MCMC 实践，告诉你怎样为模拟观测量画**诚实**的误差棒，直通第 10 章。
- A. Sokal, “Monte Carlo Methods in Statistical Mechanics: Foundations and New Algorithms”（Functional Integration, Springer, 1997）。—— 集成自相关时间 $\tau$ 与有效样本数的经典讲义，练习 4 的理论背景；做 MCMC 统计必读。

---

> **下一章预告**：本章建立了「估计量会收敛、且涨落服从高斯」这套渐近机器，但我们一直回避一个根本问题——**面对数据，到底该怎么构造估计量？** 凭什么用样本均值估总体均值、用频率估概率？第 7 章《点估计：极大似然、充分统计量、Fisher 信息与 EM》给出系统答案：**极大似然估计（MLE）** 是「让数据最可能」的普适配方，它的渐近正态与标准误正是本章 CLT + Delta 方法的直接果实（你会看到 MLE 的渐近方差恰是 **Fisher 信息**的逆）；**充分统计量**告诉你数据里哪部分才是「信息的全部」；当有隐变量（如混合模型、HMM）时，**EM 算法**给出迭代求解的钥匙。换句话说，本章告诉你估计量「收敛得多好」，第 7 章告诉你「最优的估计量从哪来、为什么 MLE 渐近最优」。
