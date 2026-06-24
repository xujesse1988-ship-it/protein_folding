# 线性代数深度入门课程 · 中英文术语对照表（GLOSSARY）

> 速查约定：每条格式为 **中文名 | English | 一句话精准解释（注所在章）**。
> 分组大致按课程主题；同一术语跨章出现时归入其最核心的主题组，必要时在解释里指出其它出场章。
> 缩写在领域内通用时随术语标注（如 SVD、PCA、LU、QR、SPD、PSD、RMSD、NMA、ENM、DCA、MSA、SO(3)、SE(3)、IPA、FAPE、GEMM、CG）。

---

## 目录

1. [向量空间与基础](#1-向量空间与基础)
2. [矩阵与线性变换](#2-矩阵与线性变换)
3. [内积·范数·正交·投影](#3-内积范数正交投影)
4. [线性方程组与矩阵分解](#4-线性方程组与矩阵分解)
5. [行列式·特征值·特征向量](#5-行列式特征值特征向量)
6. [谱定理·对称矩阵·二次型](#6-谱定理对称矩阵二次型)
7. [SVD 与 PCA](#7-svd-与-pca)
8. [数值线性代数](#8-数值线性代数)
9. [几何·旋转·SE(3)·等变性](#9-几何旋转se3等变性)
10. [蛋白质折叠 / 计算生物学应用](#10-蛋白质折叠--计算生物学应用)
11. [人物·文献·里程碑](#11-人物文献里程碑)

---

## 1. 向量空间与基础

| 中文名 | English | 解释 |
|---|---|---|
| 向量空间 | Vector space | 满足 8 条公理（加法 4 条 + 数乘 4 条）的集合，是「能加能数乘」这套行为的抽象接口（第 2 章）。 |
| 域 | Field | 标量的来源；本课默认实数域 $\mathbb R$，复数域 $\mathbb C$ 留给特征值章（第 2 章）。 |
| 标量 | Scalar | 域中的元素，用来给向量做数乘的数（第 2 章）。 |
| 向量 | Vector | 向量空间中的元素；从「箭头」升级为满足公理的抽象对象（第 2 章）。 |
| 子空间 | Subspace | 向量空间里「过原点的平的」子集，需含零、对加法与数乘封闭（第 2 章）。 |
| 仿射子空间 | Affine subspace | 子空间整体平移而成（如 $A\mathbf x=\mathbf b,\ \mathbf b\neq0$ 的解集），不含原点、不是子空间（第 2 章）。 |
| 张成 | Span | 一组向量全体线性组合构成的子空间，即含这些向量的最小子空间（第 2 章）。 |
| 线性组合 | Linear combination | 一组向量各乘标量再相加的结果，是向量空间里唯一的「合法操作」（第 2 章）。 |
| 线性无关 | Linear independence | 只有零系数才能把向量组合成零向量；等价于「没有冗余特征」（第 2 章）。 |
| 线性相关 | Linear dependence | 存在非全零系数使线性组合为零，即某向量是其余的线性组合（第 2 章）。 |
| 基 | Basis | 既线性无关又张成空间的一组向量，给出每个向量的唯一坐标（第 2 章）。 |
| 维数 | Dimension | 基的向量个数，不依赖基的选择，由 Steinitz 替换引理保证良定义（第 2 章）。 |
| 坐标 | Coordinates | 选定基后向量的唯一系数列向量，把抽象向量变成 $\mathbb R^n$ 数组；依赖于基（第 2 章）。 |
| 坐标映射 | Coordinate map | 由选定基诱导的「抽象向量 ↔ $\mathbb R^n$ 数组」同构（第 2 章）。 |
| 标准基 | Standard basis | $\mathbb R^n$ 中 $e_1\dots e_n$（第 $j$ 分量为 1 其余为 0）；氨基酸 one-hot 即 $\mathbb R^{20}$ 的标准基向量（第 2、3 章）。 |
| 独热编码 | One-hot encoding | 把 20 种氨基酸各编码为 $\mathbb R^{20}$ 的一个标准基向量，使类别得以进入线性代数（第 2 章）。 |
| 嵌入向量 | Embedding | 把氨基酸/残基/序列表示成高维向量空间里的点（第 1 章）。 |
| 维数定理 | Dimension theorem | 有限维实空间被维数完全分类，每个 $n$ 维实空间 $\cong\mathbb R^n$（坐标映射即同构）（第 2 章）。 |
| 同构 | Isomorphism | 保结构的双射；同构的空间在向量空间层面无法区分，但会「忘掉」额外结构（第 2 章）。 |
| 过渡矩阵 / 换基矩阵 | Change-of-basis matrix | 列为新基在旧基下的坐标，满足 $[\mathbf v]_B=P[\mathbf v]_C$（坐标变换与基变换方向相反）（第 2 章）。 |
| 相似 | Similar | $A'=P^{-1}AP$，同一线性变换在不同基下的两副面孔；特征值/迹/行列式/秩相同（第 2、6 章）。 |
| 切空间 | Tangent space | 非线性构象流形在每点的局部线性化，简正模式与 SE(3) 李代数都活在其中（第 2、9 章）。 |
| 哈梅尔基 | Hamel basis | 允许有限和展开的代数基；无限维下存在但无法显式写出，实用价值低（第 2 章）。 |
| 高维空间 | High-dimensional space | 维数很大的向量空间，蛋白构象/嵌入的栖息地，需警惕反直觉现象（第 1 章）。 |
| 维数灾难 | Curse of dimensionality | 高维空间反直觉现象：随机向量近正交、体积集中于壳、距离区分力下降（第 1 章）。 |
| 测度集中 | Concentration of measure | 高维下函数值与质量高度集中于「典型」区域的现象，是维数灾难的另一面（第 1、8 章）。 |

---

## 2. 矩阵与线性变换

| 中文名 | English | 解释 |
|---|---|---|
| 线性映射 / 线性变换 | Linear map / linear transformation | 保持加法与数乘的函数，必把原点映到原点；平移是仿射而非线性（第 2、3 章）。 |
| 矩阵 | Matrix | 线性映射在选定基下的数表表示；本质是「能算的变换」（第 2、3 章）。 |
| 列=基向量的像 | Columns are images of basis vectors | 矩阵表示的核心：第 $j$ 列是 $T$ 作用在第 $j$ 个基向量上的坐标（第 2、3 章）。 |
| 列视角 | Column view | 把 $A\mathbf x$ 读成 $A$ 各列以 $\mathbf x$ 分量为系数的线性组合，直通列空间/秩/注意力加权求和（第 3 章）。 |
| 矩阵–向量乘法 | Matrix–vector product | $A\mathbf x$：把变换作用在向量上，即各列的加权线性组合（第 3 章）。 |
| 矩阵乘法 = 变换复合 | Matrix product = composition | $AB$ 对应先 $B$ 后 $A$ 的复合，故不交换但结合（第 3 章）。 |
| 变换复合 | Composition | 先做 $B$ 再做 $A$，矩阵乘 $AB$ 即被定义为此，元素公式只是副产品（第 3 章）。 |
| 结合律 | Associativity | $(AB)C=A(BC)$，流水线分段不改结果，带来矩阵链省算（注意力 $Q(K^TV)$ 把 $O(L^2)$ 降到 $O(L)$）（第 3 章）。 |
| 转置 | Transpose | $A^T$ 行列互换，满足 $(AB)^T=B^TA^T$，连接内积与伴随（第 3、4 章）。 |
| 列空间 / 值域 | Column space / range | 矩阵各列张成的子空间，即线性映射所有可能输出；$A\mathbf x=\mathbf b$ 有解 ⟺ $\mathbf b$ 在其中（第 2 章）。 |
| 零空间 / 核 | Null space / kernel | 被映射压成零的输入方向集合；非平凡 ⟺ 不可逆 ⟺ 信息被压扁 ⟺ 不可辨识（第 2 章）。 |
| 行空间 | Row space | 各行张成的子空间，= $A^T$ 的列空间，最小范数解落于其中（第 2、5 章）。 |
| 左零空间 | Left null space | $A^T$ 的零空间，与列空间正交互补，是残差所在方向（第 2、8 章）。 |
| 四个基本子空间 | Four fundamental subspaces | 列空间/零空间/行空间/左零空间，刻画一个变换保留与丢失了什么（第 1、8 章）。 |
| 秩 | Rank | 列空间维数 = 行秩 = 线性无关方向数；衡量矩阵携带的独立信息/内在维度（第 2 章）。 |
| 秩–零化度定理 | Rank–nullity theorem | $\text{rank}+\text{零化度}=$ 定义域维数 $n$，一条「销毁的+保留的=原有的」守恒律（第 2 章）。 |
| 数值秩 | Numerical rank | 奇异值大于某阈值的个数；浮点世界判秩须看奇异值断崖，不能用行列式（第 2、8 章）。 |
| 低秩 | Low rank | 信息集中在少数独立方向，可压缩；PCA、MDS、距离几何的共同立足点（第 2 章）。 |
| 各向异性缩放 | Anisotropic scaling | 对角矩阵 $\text{diag}(s_1,s_2)$，沿各坐标轴独立伸缩，把单位圆变椭圆（第 3 章）。 |
| 对角矩阵 | Diagonal matrix | 只有对角元非零，作用即各轴独立缩放，最易计算的矩阵（第 3、6 章）。 |
| 旋转矩阵 | Rotation matrix | $\begin{bmatrix}\cos\theta&-\sin\theta\\\sin\theta&\cos\theta\end{bmatrix}$，保距保定向（$\det=1$）、正交，刚体旋转/SO(2)、SO(3)（第 3、9 章）。 |
| 剪切（错切） | Shear | 如 $\begin{bmatrix}1&k\\0&1\end{bmatrix}$，把图形推歪但保面积（$\det=1$），对应高斯消元的初等矩阵（第 3 章）。 |
| 反射 | Reflection | 镜像变换，保距但 $\det=-1$ 翻转定向，对应手性；结构叠合须排除（第 3 章）。 |
| 幂等矩阵 | Idempotent matrix | 满足 $P^2=P$ 的矩阵，正交投影是其核心实例（第 3 章）。 |
| 三角矩阵 | Triangular matrix | 上/下三角，几何上是单向剪切，解方程快，是 LU 分解的产物（第 3、5 章）。 |
| 置换矩阵 | Permutation matrix | 每行每列恰一个 1，重排坐标，正交（$P^{-1}=P^T$），连接置换不变/等变与 GNN（第 3、5 章）。 |
| 正交矩阵 | Orthogonal matrix | $Q^TQ=I$（逆=转置），保长度保角度的刚性变换，frame/SVD 因子/Kabsch 的核心（第 3、4、9 章）。 |
| 外积 | Outer product | $\mathbf u\mathbf v^T\to$ 秩 1 矩阵，一个「基本动作」，是 SVD/谱分解/协方差/秩 1 更新的构件（第 3、6、7 章）。 |
| 秩 1 矩阵 | Rank-1 matrix | 列全落在一条直线上的矩阵，恰能写成外积；矩阵 = 秩 1 外积之和（谱视角）（第 3 章）。 |
| Hadamard 积 | Hadamard product | $A\odot B$ 逐元素相乘，门控/掩码/dropout；numpy 里是 `*`（勿与矩阵乘 `@` 混）（第 3 章）。 |
| Kronecker 积 | Kronecker product | $A\otimes B$ 张量积，在乘积空间上同时施加两变换，对应批量/可分离/等变运算（第 3 章）。 |
| 分块矩阵 | Block matrix | 把大矩阵当「矩阵的矩阵」，分块乘法把块当元素，对应多头注意力与高性能 GEMM（第 3 章）。 |
| 多头注意力 | Multi-head attention | 按列把 $Q/K/V$ 分块成多个头各自做注意力再拼接，本质是分块运算（第 3 章）。 |
| 注意力 | Attention | $\text{softmax}(QK^T/\sqrt d)V$，由矩阵乘 + 按行归一化 + 矩阵乘构成，Transformer/Evoformer 核心（第 1、3、11 章）。 |
| 通用矩阵乘法 | GEMM (general matrix multiply) | GPU 因其算术强度高而偏爱，深度学习把一切（全连接/注意力/卷积）化成它（第 3、8 章）。 |
| 算术强度 | Arithmetic intensity | 计算量与访存量之比；矩阵乘 $O(n^3)$ 算/$O(n^2)$ 搬，比值高故硬件友好（第 3、8 章）。 |
| 恒等矩阵 | Identity matrix | $I$，什么也不做的变换，矩阵乘单位元，残差连接 $\mathbf h+\dots$ 的核心（第 3 章）。 |
| 逆矩阵 | Inverse matrix | $A^{-1}$ 撤销 $A$（$A^{-1}A=I$），是变换流水线的 undo，只对方阵满秩存在（第 3 章）。 |
| 可逆矩阵定理 | Invertible Matrix Theorem | 方阵可逆 ⟺ 满秩 ⟺ 列无关 ⟺ 零空间平凡 ⟺ $\det\neq0$ ⟺ 无零特征值 ⟺ 唯一解，本质 = 不丢信息 = 能撤销（第 3 章）。 |
| 奇异矩阵 | Singular matrix | 不可逆，零空间非平凡，把空间压扁到低维、多对一丢信息（第 2、3 章）。 |
| 伪逆 | Pseudoinverse | Moore–Penrose $A^+$，对任意矩阵做「最合理的近似撤销」（最小二乘/最小范数解），由 SVD 给出（第 3、5、8 章）。 |
| 简正模式分析 | Normal mode analysis (NMA) | 对 Hessian（或 ENM 代理）做特征分解，低频模式 = 蛋白集体运动；即「换特征基对角化」（第 1、2、7、11 章）。 |

---

## 3. 内积·范数·正交·投影

| 中文名 | English | 解释 |
|---|---|---|
| 内积 | Inner product | $\mathbf u^T\mathbf v\to$ 标量，度量两向量对齐/相似度，是几何（长度、角度、正交）的种子（第 1、3、4 章）。 |
| 点积 | Dot product | $\mathbb R^n$ 上的标准内积 $\sum u_iv_i$，是内积公理的具体范例（第 4 章）。 |
| 余弦相似度 | Cosine similarity | 归一化内积 $\mathbf u^T\mathbf v/(\|\mathbf u\|\|\mathbf v\|)$，度量两向量方向一致性（第 1 章）。 |
| 内积公理 | Inner product axioms | 对称、双线性、正定三条，抽象出「能量长度量角度」的最小要求（第 4 章）。 |
| 一般内积 | General inner product | 由正定对称矩阵诱导的内积 $\mathbf u^TM\mathbf v$，点积只是 $M=I$ 的特例（第 4 章）。 |
| 长度 / 模 | Length / magnitude | 由内积导出的 $\|\mathbf v\|=\sqrt{\langle\mathbf v,\mathbf v\rangle}$（第 4 章）。 |
| 夹角 | Angle | 由 $\cos\theta=\langle\mathbf u,\mathbf v\rangle/(\|\mathbf u\|\|\mathbf v\|)$ 定义，内积把几何角度代数化（第 4 章）。 |
| 范数 | Norm | 满足非负、齐次、三角不等式的「长度」函数；并非都来自内积（第 4 章）。 |
| $L^1$ / $L^2$ / $L^\infty$ 范数 | $L^1$ / $L^2$ / $L^\infty$ norm | 绝对值和 / 欧氏长度 / 最大分量，三种常用向量范数（第 4 章）。 |
| 范数等价 | Norm equivalence | 有限维下任意两范数互相被常数倍夹住，「殊途同归」（第 4 章）。 |
| Cauchy–Schwarz 不等式 | Cauchy–Schwarz inequality | $|\langle\mathbf u,\mathbf v\rangle|\le\|\mathbf u\|\|\mathbf v\|$，相关系数有界、夹角良定义的根源（第 4 章）。 |
| 三角不等式 | Triangle inequality | $\|\mathbf u+\mathbf v\|\le\|\mathbf u\|+\|\mathbf v\|$，范数的几何底线（第 4 章）。 |
| 极化恒等式 | Polarization identity | 由范数反推内积的公式，刻画「哪些范数来自内积」（第 4 章）。 |
| 正交 | Orthogonal | 内积为零 $\langle\mathbf u,\mathbf v\rangle=0$，几何上互相垂直（第 4 章）。 |
| 正交基 | Orthogonal basis | 两两正交的一组基，坐标可由内积直接读出（第 4 章）。 |
| 标准正交基 | Orthonormal basis | 两两正交且单位长度的一组基，正交对角化中 $Q$ 的列即一组标准正交特征向量（第 4、7 章）。 |
| 坐标 = 内积 | Coordinates as inner products | 标准正交基下，向量在某轴的坐标就是它与该轴的内积，无需解方程（第 4 章）。 |
| 正交补 | Orthogonal complement | 与给定子空间处处正交的全体向量，把全空间正交劈成两半（第 4 章）。 |
| 勾股定理 | Pythagorean theorem | 正交分量下 $\|\mathbf u+\mathbf v\|^2=\|\mathbf u\|^2+\|\mathbf v\|^2$，最小二乘误差分解的根（第 4 章）。 |
| 正交投影 | Orthogonal projection | 把空间垂直拍扁到子空间的最近点，幂等 $P^2=P$、对称 $P^T=P$，PCA/最小二乘内核（第 3、4 章）。 |
| 投影矩阵 | Projection matrix | $P=A(A^TA)^{-1}A^T$，把任意向量拍到 $A$ 列空间的算子（第 4 章）。 |
| Gram–Schmidt 正交化 | Gram–Schmidt process | 逐列减去在已有正交向量上的投影再归一化，产出正交基，即 QR 分解（第 4、5 章）。 |
| 内积空间 | Inner product space | 配备内积的向量空间，长度/角度/正交在此皆有定义（第 4 章）。 |
| 伴随 | Adjoint | 满足 $\langle A\mathbf x,\mathbf y\rangle=\langle\mathbf x,A^T\mathbf y\rangle$ 的算子，实空间即转置（第 4 章）。 |
| 最小二乘（几何） | Least squares (geometric) | $A\mathbf x=\mathbf b$ 无解时，把 $\mathbf b$ 投影到列空间求最近近似解（第 4、5 章）。 |

---

## 4. 线性方程组与矩阵分解

| 中文名 | English | 解释 |
|---|---|---|
| 线性方程组 | Linear system $A\mathbf x=\mathbf b$ | 用列空间（相容性）与零空间（唯一性）语言看穿解的结构，决定一切的是秩而非方程个数（第 5 章）。 |
| 相容性 | Consistency | $A\mathbf x=\mathbf b$ 有解当且仅当 $\mathbf b$ 落在 $A$ 的列空间中（第 5 章）。 |
| 超定系统 | Overdetermined system | 方程多于未知数，一般无精确解，转而求最小二乘解（第 5 章）。 |
| 欠定系统 | Underdetermined system | 未知数多于方程，有无穷多解，常取最小范数解（第 5 章）。 |
| 最小范数解 | Minimum-norm solution | 欠定系统无穷多解中范数最小的那个，落在行空间里，是正则化偏好的雏形（第 5 章）。 |
| 高斯消元 | Gaussian elimination | 用剪切型行操作把矩阵化为上三角，再回代求解（第 5 章）。 |
| 初等行变换 | Elementary row operation | 换行/数乘/倍加三种操作，对应左乘初等矩阵，是消元的原子（第 5 章）。 |
| 主元 | Pivot | 消元中每列用来消去下方元素的非零首元，其个数 = 秩（第 5 章）。 |
| LU 分解 | LU decomposition | $A=LU$，把消元乘数（$L$）与结果（$U$）打包，实现分解一次 $O(n^3)$、多次求解 $O(n^2)$（第 5 章）。 |
| 部分主元 | Partial pivoting | 每列消元前把绝对值最大元换到主元位置以保数值稳定，得 $PA=LU$（第 5 章）。 |
| PLU 分解 | PLU decomposition | 带行交换记录 $P$ 的 LU 分解，数值稳健的标准直接解法（第 5 章）。 |
| 前代 / 回代 | Forward / back substitution | 三角系统从一端逐个解出，各只需 $O(n^2)$，是 LU/Cholesky 求解的廉价后半步（第 5 章）。 |
| 对称正定矩阵 | Symmetric positive definite (SPD) | $A=A^T$ 且 $\mathbf x^TA\mathbf x>0$（处处开口向上的碗），是协方差/精度/Hessian/Gram 的共同身份（第 5、7 章）。 |
| Cholesky 分解 | Cholesky decomposition | SPD 矩阵 $A=LL^T$ 的三角平方根，比 LU 省一半且无需换行，是高斯采样引擎与正定性检验（第 5 章）。 |
| 多元高斯采样 | Multivariate Gaussian sampling | $\mathbf x=\boldsymbol\mu+L\mathbf z\ (\mathbf z\sim\mathcal N(0,I))$，用 Cholesky 因子 $L$ 把白噪声塑形成目标协方差（第 5 章）。 |
| QR 分解 | QR decomposition | $A=QR$（$Q$ 列正交、$R$ 上三角）= 对列做正交化，解最小二乘用 $R\mathbf x=Q^T\mathbf b$ 比正规方程稳（第 5 章）。 |
| Householder 反射 | Householder reflection | 用正交反射把矩阵逐列打成三角形，数值最稳健的 QR 算法，是库的默认（第 5、8 章）。 |
| 正规方程 | Normal equations | $A^TA\mathbf x=A^T\mathbf b$，最小二乘的经典解法但因 $\kappa(A^TA)=\kappa(A)^2$ 数值上不推荐（第 5、8 章）。 |
| 最小二乘 | Least squares | 超定无解时求最小残差（投影到列空间），欠定时求最小范数解（第 5 章）。 |
| 残差 | Residual | $\mathbf b-A\hat{\mathbf x}$，最小二乘最优时它正交于列空间（第 5 章）。 |
| Moore–Penrose 伪逆 | Moore–Penrose pseudoinverse | $A^+$ 用一个对象统一过定/欠定（$\hat{\mathbf x}=A^+\mathbf b$），可逆时退为逆，完整结构靠 SVD（第 5、8 章）。 |
| 岭回归 | Ridge regression | $\min\|A\mathbf x-\mathbf b\|^2+\lambda\|\mathbf x\|^2$，闭式解 $(A^TA+\lambda I)^{-1}A^T\mathbf b$，治病态的标准药（第 5、8 章）。 |
| Tikhonov 正则 | Tikhonov regularization | 岭回归别名，谱上把小奇异值方向放大因子从 $1/\sigma$ 改为 $\sigma/(\sigma^2+\lambda)$ 温柔压回（第 5、8 章）。 |
| 最大后验估计 | Maximum a posteriori (MAP) | 最大化后验 = 似然 × 先验，L2 正则正是零均值高斯先验下的 MAP，$\lambda=\sigma^2/\tau^2$（第 5 章）。 |
| 共轭梯度 | Conjugate gradient (CG) | 针对大型稀疏 SPD 系统的迭代法，只需矩阵 × 向量、不显式分解，收敛由条件数决定（第 5、8 章）。 |
| Gram 矩阵 | Gram matrix | $G=B^TB$（两两内积）天然半正定，列无关时正定，是协方差/核矩阵的共同来源（第 1、5、7 章）。 |
| 条件数 | Condition number | $\kappa(A)=\sigma_{\max}/\sigma_{\min}$，度量问题对扰动的固有敏感度，$\kappa\approx10^k$ 丢约 $k$ 位精度（第 1、3、5、8 章）。 |
| 病态 | Ill-conditioned | 问题本身对输入扰动敏感（列几乎相关、列空间被压扁），是问题属性而非算法缺陷（第 5、8 章）。 |
| 数值稳定性 | Numerical stability | 算法属性（不人为额外恶化误差），与问题良态（条件数）正交，两者须分开关心（第 5、8 章）。 |

---

## 5. 行列式·特征值·特征向量

| 中文名 | English | 解释 |
|---|---|---|
| 行列式 | Determinant | 线性变换对有向体积的缩放因子；符号 = 是否翻转定向，0 = 压扁降维（第 6 章）。 |
| 有向体积 | Signed volume | 带符号的体积，符号编码定向（右手系/镜像）是否保持（第 6 章）。 |
| 多线性与交替 | Multilinear and alternating | 行列式由多线性 + 交替 + 归一化三条几何公理唯一确定（第 6 章）。 |
| 余子式展开 | Cofactor / Laplace expansion | 沿一行/列把行列式拆成低阶行列式的递归公式，理论清晰但计算昂贵（第 6 章）。 |
| 雅可比矩阵 | Jacobian matrix | 非线性变换在一点的最佳线性近似，列 = 各方向的像（第 6 章）。 |
| 雅可比行列式 | Jacobian determinant | 非线性变换的局部体积缩放因子，多维变量替换的密度反缩放因子（第 6、11 章）。 |
| 特征值 / 特征向量 | Eigenvalue / eigenvector | 满足 $A\mathbf v=\lambda\mathbf v$ 的非零方向 $\mathbf v$（不转向只缩放）与缩放倍数 $\lambda$，谱视角的核心（第 1、6 章）。 |
| 特征多项式 | Characteristic polynomial | $\det(A-\lambda I)$，其根 = 全部特征值；仅作定义工具，不用于数值计算（第 6 章）。 |
| 特征子空间 | Eigenspace | 同一特征值对应的全部特征向量加零向量构成的子空间（第 6 章）。 |
| 谱 | Spectrum | 矩阵全部特征值（含重数、含复值）的集合，是变换的坐标无关指纹（第 1、6 章）。 |
| 代数重数 / 几何重数 | Algebraic / geometric multiplicity | 特征值作为根的重数 vs 其特征子空间维数；二者相等是可对角化条件（第 6 章）。 |
| 对角化 | Diagonalization | $A=PDP^{-1}$，换到特征基使变换变成沿各轴的纯缩放（第 6 章）。 |
| 谱分解 | Spectral decomposition | $A=\sum\lambda_i\mathbf v_i\mathbf w_i^T$，矩阵 = 各特征方向秩 1 投影的加权和（对称时 $\mathbf w_i=\mathbf v_i$）（第 1、6 章）。 |
| 相似不变量 | Similarity invariant | 相似矩阵共享的量（特征值、迹、行列式、秩），是变换的坐标无关属性（第 6 章）。 |
| 亏损矩阵 | Defective matrix | 几何重数小于代数重数、不可对角化的矩阵，只能化为 Jordan 形（第 6 章）。 |
| Jordan 标准形 | Jordan normal form | 亏损矩阵的最简形（对角放特征值、上对角放 1），对扰动不连续、数值上避开（第 6 章）。 |
| 迹 | Trace | 对角元之和 = 特征值之和；度量总缩放/总方差/有效维度（第 6 章）。 |
| 矩阵幂 | Matrix power | $A^k=PD^kP^{-1}$，特征基里各分量乘 $\lambda_i^k$，线性迭代 = 独立几何级数叠加（第 6 章）。 |
| 谱半径 | Spectral radius | $\rho(A)=\max|\lambda_i|$，决定线性动力系统长期行为（$<1$ 收敛、$>1$ 爆炸、$=1$ 稳态）（第 6 章）。 |
| 幂法 / 幂迭代 | Power iteration | 反复 $\mathbf x\leftarrow A\mathbf x/\|A\mathbf x\|$ 收敛到主特征向量；PageRank/谱聚类/essential dynamics 底层（第 6、7、8 章）。 |
| 随机矩阵 / 转移矩阵 | Stochastic / transition matrix | 每行非负且和为 1 的矩阵，描述马尔可夫链一步演化（第 6 章）。 |
| 稳态分布 | Stationary distribution | 满足 $\pi P=\pi$ 的分布，等于 $P^T$ 特征值 1 的特征向量（归一化为概率）（第 6 章）。 |
| 第二特征值 / 谱隙 | Second eigenvalue / spectral gap | $|\lambda_2|$ 控制马尔可夫链收敛速率；谱隙 $1-|\lambda_2|$ 越小混合越慢（MCMC 困局根源）（第 6、8 章）。 |
| 复特征值 | Complex eigenvalue | 实矩阵的复特征值成共轭对 $re^{\pm i\theta}$，几何上 = 旋转 $\theta$ + 缩放 $r$，是旋转/振荡成分的指纹（第 6 章）。 |
| 归一化流 | Normalizing flow | 用可逆可微变换把简单分布变形为复杂分布的生成模型，靠雅可比行列式精确算似然（第 6、11 章）。 |

---

## 6. 谱定理·对称矩阵·二次型

| 中文名 | English | 解释 |
|---|---|---|
| 谱定理 | Spectral theorem | 实对称矩阵必有实特征值、可取正交特征向量、可正交对角化 $A=Q\Lambda Q^T$，无旋转成分（第 6、7 章）。 |
| 实对称矩阵 | Real symmetric matrix | 满足 $A=A^T$ 的实矩阵，享有谱定理的三重恩赐（第 7 章）。 |
| 正交对角化 | Orthogonal diagonalization | $A=Q\Lambda Q^T$（$Q$ 正交），对称矩阵特有的「无畸变」对角化（第 7 章）。 |
| 埃尔米特矩阵 | Hermitian matrix | 满足 $A^*=A$ 的复矩阵，实对称是其特例，保证特征值为实（第 7 章）。 |
| 正规矩阵 | Normal matrix | 满足 $AA^*=A^*A$，谱定理在复情形的推广对象（可酉对角化）（第 7 章）。 |
| 秩 1 展开 | Rank-1 expansion | $A=\sum\lambda_i\mathbf q_i\mathbf q_i^T$，把对称矩阵拆成正交投影 $\mathbf q_i\mathbf q_i^T$ 加权（增益 $\lambda_i$）之和（第 7 章）。 |
| 低秩近似 | Low-rank approximation | 只保留前 $k$ 个大特征值项 $A_k=\sum_{i\le k}\lambda_i\mathbf q_i\mathbf q_i^T$，PCA/essential dynamics 的雏形（第 7、8 章）。 |
| 矩阵函数 | Matrix function | $f(A)=Qf(\Lambda)Q^T$，在每根主轴上把增益换成 $f(\lambda_i)$，统一 $e^{-tH}$、$\Sigma^{1/2}$、$\Sigma^{-1}$（第 7 章）。 |
| 矩阵平方根 | Matrix square root | $A^{1/2}=Q\Lambda^{1/2}Q^T$，对 SPD 矩阵良定义，白化与采样的工具（第 7 章）。 |
| 谱不变量 | Spectral invariants | $\text{tr}(A)=\sum\lambda_i$（总方差）、$\det(A)=\prod\lambda_i$、$\log\det\Sigma=\sum\log\lambda_i$，正交相似下不变（第 7 章）。 |
| 二次型 | Quadratic form | $\mathbf x^TA\mathbf x$，只看 $A$ 的对称部分；能量二阶展开、马氏距离皆是二次型（第 7 章）。 |
| 主轴定理 | Principal axis theorem | 经正交换基 $\mathbf y=Q^T\mathbf x$ 二次型化为纯平方和 $\sum\lambda_iy_i^2$，特征向量即主轴（第 7 章）。 |
| 合同变换 | Congruence transformation | $C^TAC$（≠ 相似 $C^{-1}AC$），二次型在换基下的变形规律，质量加权 NMA 即合同（第 7 章）。 |
| Sylvester 惯性定律 | Sylvester's law of inertia | 合同变换保持正/负/零特征值个数三元组 $(n_+,n_-,n_0)$ 不变（第 7 章）。 |
| 正定矩阵 | Positive definite matrix | $\mathbf x^TA\mathbf x>0$ 恒成立，等价于全部特征值 $>0$、可 Cholesky、可写满秩 $B^TB$（第 7 章）。 |
| 半正定矩阵 | Positive semidefinite matrix (PSD) | $\mathbf x^TA\mathbf x\ge0$，含零特征值 = 退化/平坦方向；协方差、Gram、$A^TA$ 天然半正定（第 7 章）。 |
| Sylvester 判据 | Sylvester's criterion | 全部顺序主子式 $>0$ ⟺ 严格正定；不能用顺序主子式 $\ge0$ 去判半正定（第 7 章）。 |
| 顺序主子式 | Leading principal minor | 左上角 $k\times k$ 子块的行列式，Sylvester 判据的检验对象（第 7 章）。 |
| 瑞利商 | Rayleigh quotient | $R_A(\mathbf x)=\mathbf x^TA\mathbf x/\|\mathbf x\|^2$，= 特征值按方向的加权平均，极值即 $\lambda_1$、$\lambda_n$（第 7 章）。 |
| Courant–Fischer 极小极大定理 | Courant–Fischer min-max theorem | 用子空间与瑞利商给特征值的坐标无关刻画，副产 Weyl 扰动不等式（第 7 章）。 |
| Weyl 不等式 | Weyl's inequality | 对称扰动 $A\to A+E$ 下 $|\lambda_k$ 变化$|\le\|E\|_2$，保证对称谱对扰动良态（第 7 章）。 |
| 马氏距离 | Mahalanobis distance | $(\mathbf x-\boldsymbol\mu)^T\Sigma^{-1}(\mathbf x-\boldsymbol\mu)$，以精度矩阵为核的二次型，等值面 = 协方差椭球（第 7 章）。 |
| 精度矩阵 | Precision matrix | $\Lambda=\Sigma^{-1}$，非对角零元 = 条件独立 = 无直接耦合，DCA 接触预测的核心（第 1、7、11 章）。 |
| 协方差矩阵 | Covariance matrix | 刻画各自由度两两协方差的对称半正定矩阵 $\Sigma$，其谱给出主要运动方向（第 1、7 章）。 |
| 简正模式 | Normal mode | 极小处 Hessian 的特征向量 = 解耦的集体振动方向，特征值 = 角频率平方（第 7 章）。 |
| 黑塞矩阵 / 海森矩阵 | Hessian | 能量面二阶导构成的对称矩阵，正定 = 稳定极小，特征分解给振动模式，逆正比涨落协方差（第 1、7、11 章）。 |
| 弹性网络模型 | Elastic network model (ENM) | 用接触图加权拉普拉斯当 Hessian 代理；GNM 给幅度、ANM 给方向（第 1、7、11 章）。 |
| 本质动力学 | Essential dynamics | 对 MD 轨迹协方差做谱分解，前几个主成分 = 蛋白主要集体运动（第 1、7、8、11 章）。 |
| 图拉普拉斯 | Graph Laplacian | $L=D-A$，半正定，$\mathbf x^TL\mathbf x=\tfrac12\sum A_{ij}(x_i-x_j)^2$，谱编码连通与社团结构（第 7、11 章）。 |
| Fiedler 向量 | Fiedler vector | 图拉普拉斯第二小特征值的特征向量，谱聚类/结构域划分的最小割松弛解（第 7 章）。 |
| 白化 | Whitening | 用 $\Sigma^{-1/2}$ 把相关、各向异性的特征拉直归一为各向同性白噪声（第 1、8 章）。 |

---

## 7. SVD 与 PCA

| 中文名 | English | 解释 |
|---|---|---|
| 奇异值分解 | Singular value decomposition (SVD) | 任意 $m\times n$ 实矩阵 $A=U\Sigma V^T$：正交旋转 $V^T\to$ 沿轴非负拉伸 $\Sigma\to$ 正交旋转 $U$，本课最有用的分解（第 1、8 章）。 |
| 奇异值 | Singular values | $\Sigma$ 的非负对角元 $\sigma_i$（降序），= $A^TA$（或 $AA^T$）特征值的非负平方根，= 椭球半轴长（第 8 章）。 |
| 左奇异向量 | Left singular vectors | $U$ 的列 $\mathbf u_i$，是 $AA^T$ 的特征向量，张成列空间/左零空间，= 椭球半轴方向（第 8 章）。 |
| 右奇异向量 | Right singular vectors | $V$ 的列 $\mathbf v_i$，是 $A^TA$ 的特征向量，张成行空间/零空间，被映成椭球半轴的原像方向（第 8 章）。 |
| 灵魂方程 | $A\mathbf v_i=\sigma_i\mathbf u_i$ | SVD 核心关系：输入端正交方向 $\mathbf v_i$ 被 $A$ 作用后仍是输出端正交方向 $\mathbf u_i$（拉伸 $\sigma_i$ 倍）（第 8 章）。 |
| 经济型 / 薄 SVD | Economy / thin SVD | 只保留 $k=\min(m,n)$ 个奇异值的紧凑形式，numpy 用 `full_matrices=False`，实务默认（第 8 章）。 |
| 截断 SVD | Truncated SVD | 只保留前 $k$ 个奇异值项 $A_k=\sum_{i\le k}\sigma_i\mathbf u_i\mathbf v_i^T$，秩 $\le k$，是抓大放小的有损压缩（第 8 章）。 |
| Eckart–Young 定理 | Eckart–Young–Mirsky theorem | 截断 SVD 是酉不变范数（Frobenius/谱范数）下最优秩 $k$ 近似，误差 = 丢掉的奇异值（第 8 章）。 |
| Frobenius 范数 | Frobenius norm | $\|A\|_F=\sqrt{\sum\sigma_i^2}$ = 矩阵「总能量」（奇异值平方和开根），正交变换不变（第 8 章）。 |
| 谱范数 / 算子 2-范数 | Spectral norm / operator 2-norm | $\|A\|_2=\sigma_1$ = 最大奇异值 = 把向量拉长的最大倍数 = 条件数分子（第 8 章）。 |
| 酉不变范数 | Unitarily invariant norm | 左右乘正交矩阵不改其值的范数族（Frobenius/谱/核范数），Eckart–Young 的适用范围（第 8 章）。 |
| 最小范数最小二乘解 | Minimum-norm least-squares solution | $\mathbf x^*=A^+\mathbf b$ 同时实现残差最小与解范数最小，统一过定/欠定问题（第 8 章）。 |
| 反问题 | Inverse problem | 从观测反推参数（如从距离约束重建结构）；小/零奇异值 = 不可辨识方向 = 病态的根（第 8 章）。 |
| 吉洪诺夫 / 岭回归滤波 | Tikhonov / ridge filtering | 把伪逆里 $1/\sigma_i$ 换成 $\sigma_i/(\sigma_i^2+\lambda^2)$，平滑压住小奇异值方向的噪声爆炸（第 8 章）。 |
| 主成分分析 | Principal component analysis (PCA) | = 中心化数据 $\tilde X$ 的 SVD = 协方差 $\Sigma$ 的特征分解；主方向 = 右奇异向量，解释方差 = $\sigma_i^2/(N-1)$（第 1、8 章）。 |
| 主成分 | Principal component | 数据方差最大的正交方向（右奇异向量/协方差特征向量），按解释方差降序排列（第 8 章）。 |
| 中心化 | Centering | PCA 第一步：每列减均值；PCA 是关于围绕均值的方差，不中心化几何会被均值偏移带歪（第 8 章）。 |
| 解释方差比例 | Explained variance ratio (EVR) | $\text{EVR}_i=\lambda_i/\sum\lambda_j=\sigma_i^2/\sum\sigma_j^2$，配合碎石图选主成分个数 $k$（第 8 章）。 |
| 碎石图 | Scree plot | 奇异值/特征值随序号的衰减曲线，找「肘部/悬崖」以决定保留几个主成分（第 8 章）。 |
| 主成分得分 / 载荷 | PC scores / loadings | 数据在主方向上的投影坐标 / 原始变量对主成分的权重（第 8 章）。 |
| Kabsch 算法 / 正交 Procrustes | Kabsch algorithm / orthogonal Procrustes | 用 SVD 求把两点云最优叠合的旋转：去质心 $\to H=P^TQ\to$ SVD $\to R=V\,\text{diag}(1,1,\det(VU^T))\,U^T$（第 1、8、9 章）。 |
| 反射修正 | Reflection correction | Kabsch 中当 $\det(VU^T)=-1$ 时翻转最小奇异方向，保 $\det R=+1$ 避免得到镜像（手性翻转）结构（第 8、9 章）。 |
| 均方根偏差 | Root-mean-square deviation (RMSD) | Kabsch 叠合后两结构对应原子距离的均方根，结构相似性的基本度量（第 1、8、9、11 章）。 |
| 低秩注意力 | Low-rank attention | 注意力 $QK^T$ 秩 $\le d\ll n$ 天生低秩，可截断加速（如 Linformer），把 $O(n^2)$ 降到 $O(nk)$（第 8 章）。 |
| 潜因子模型 | Latent factor model | 把评分/共现矩阵低秩分解成用户/物品/词的潜因子向量，内积预测；缺失场景属矩阵补全（第 8 章）。 |
| 矩阵补全 | Matrix completion | 由部分观测项在低秩假设下还原整张矩阵，推荐系统/缺失数据的核心问题（第 8 章）。 |

---

## 8. 数值线性代数

| 中文名 | English | 解释 |
|---|---|---|
| 浮点数 | Floating-point number | 用有限位尾数 × 基的指数表示实数，导致舍入误差，是数值线代的工作介质（第 9 章）。 |
| 机器精度 | Machine epsilon | 刻画浮点相对精度的常数，float64 约 $2.22\times10^{-16}$，决定有效数字位数（第 9 章）。 |
| 灾难性抵消 | Catastrophic cancellation | 两相近大数相减把早先舍入误差曝光放大成相对误差，是误差放大头号元凶（第 9 章）。 |
| 标准浮点模型 | Standard floating-point model | $\text{fl}(a\,\text{op}\,b)=(a\,\text{op}\,b)(1+\delta),\ |\delta|\le u$，把单次舍入形式化为微小相对扰动（第 9 章）。 |
| 向前误差 | Forward error | 算出值与真值之差，最终关心但难直接控制（第 9 章）。 |
| 向后误差 | Backward error | 算出的结果是哪个微扰输入的精确解，所需最小扰动大小；向后稳定 = 该扰动为 $O(u)$（第 9 章）。 |
| 向后稳定性 | Backward stability | 算法属性：结果总是某个微小扰动后输入的精确解；黄金法则 = 向前误差 ≲ 条件数 × 向后误差（第 9 章）。 |
| 改进 Gram–Schmidt | Modified Gram–Schmidt | 重排减投影顺序的正交化，代数上同经典 GS 但浮点下显著更稳（第 9 章）。 |
| 反幂法 / 移位反幂法 | Inverse / shift-and-invert iteration | 对 $A^{-1}$ 或 $(A-\mu I)^{-1}$ 做幂法，狙击最小或指定区域特征值（第 9 章）。 |
| Krylov 子空间 | Krylov subspace | $\text{span}\{\mathbf v,A\mathbf v,\dots,A^{k-1}\mathbf v\}$，只用矩阵乘向量能到达的最大信息空间，迭代法统一灵魂（第 9 章）。 |
| Lanczos 算法 | Lanczos algorithm | 对称矩阵在 Krylov 子空间上投影成小三对角阵，Ritz 值极快逼近极端特征对（`eigsh` 底座）（第 9 章）。 |
| Arnoldi 算法 | Arnoldi iteration | 非对称矩阵在 Krylov 子空间上的正交投影，Lanczos 的一般化（第 9 章）。 |
| Ritz 值 | Ritz value | Krylov 子空间投影所得小矩阵的特征值，逼近原大矩阵的极端特征值（第 9 章）。 |
| 预条件 | Preconditioning | 用 $M\approx A$ 使 $M^{-1}A$ 条件数大降，是迭代法实用化分水岭（Jacobi/不完全 Cholesky/多重网格）（第 9 章）。 |
| 稀疏矩阵 | Sparse matrix | 绝大多数元素为零，CSR/CSC 只存非零，矩阵乘成本正比非零元数，与迭代法天作之合（第 9 章）。 |
| 稀疏矩阵–向量乘 | Sparse matrix–vector product (SpMV) | 只对非零元做乘加，是迭代法每步主开销，蛋白图 GNN 的核心算子（第 9、11 章）。 |
| 随机化 SVD | Randomized SVD | 用随机高斯草图撞出主子空间，把巨型 SVD 化为几次矩阵乘加一个小 SVD，对数值低秩高效（第 8、9 章）。 |
| 随机投影 | Random projection | 用随机矩阵把高维数据降到低维而近似保距，随机化 SVD 与 JL 引理的工具（第 9 章）。 |
| Johnson–Lindenstrauss 引理 | Johnson–Lindenstrauss lemma | 保 $N$ 点两两距离所需随机投影维度 $O(\varepsilon^{-2}\log N)$，与原维数无关（高维测度集中）（第 9 章）。 |
| 混合精度 | Mixed precision | 主体用 fp16/bf16、关键处保 fp32（累加器/softmax/优化器状态）以省显存提速（第 9 章）。 |
| 脑浮点 | bfloat16 (bf16) | 指数位同 fp32（动态范围大、几乎不溢出）但尾数仅 7 位，大模型训练首选（第 9 章）。 |
| 损失缩放 | Loss scaling | fp16 训练把 loss 乘大常数撑起梯度量级、反传后除回，防小梯度下溢成零（第 9 章）。 |
| 归一化层 | Normalization layer | LayerNorm/BatchNorm 等本质压低损失曲面 Hessian 条件数、把病态峡谷揉圆，使优化快而稳（第 9 章）。 |
| 梯度爆炸 / 消失 | Exploding / vanishing gradient | 雅可比连乘的谱问题：奇异值普遍 $>1$ 爆炸、$<1$ 消失，靠控谱缓解（第 9 章）。 |
| 谱归一化 | Spectral normalization | 强制权重矩阵最大奇异值 $\le1$，控制雅可比谱防梯度爆炸（第 9 章）。 |
| 正交初始化 | Orthogonal initialization | 用正交矩阵初始化权重使初始雅可比谱接近 1，缓解梯度爆炸/消失（第 9 章）。 |
| 残差连接 | Residual connection | $\mathbf x+f(\mathbf x)$ 使每层雅可比变 $I+J_f$ 靠近单位阵，提供谱 ≈ 1 的恒等高速路缓解梯度消失（第 9 章）。 |
| FlashAttention | FlashAttention | 分块计算注意力以提升算术强度、绕开 $O(n^2)$ 显存的 IO 感知算法（第 9 章）。 |

---

## 9. 几何·旋转·SE(3)·等变性

| 中文名 | English | 解释 |
|---|---|---|
| 特殊正交群 | Special orthogonal group SO(3) | 三维旋转矩阵全体：正交（$R^TR=I$）且 $\det=+1$，保距保手性，运算为矩阵乘的非交换群（第 1、10 章）。 |
| 特殊欧氏群 | Special Euclidean group SE(3) | 刚体变换 $R\mathbf x+\mathbf t$ 全体，6 自由度，= 三维所有保距保定向运动（旋转 + 平移）（第 1、10 章）。 |
| 轴角表示 | Axis–angle representation | 绕单位轴 $\mathbf n$ 转角 $\theta$；由欧拉旋转定理任意旋转都等价于此，轴 = 特征值 1 的特征向量（第 10 章）。 |
| 欧拉旋转定理 | Euler's rotation theorem | 任意三维旋转都等价于绕某固定轴转某角度（第 10 章）。 |
| 欧拉角 | Euler angles | 绕三轴依次转三个角的紧凑表示，但有万向锁、不唯一、约定多，不宜用于计算（第 10 章）。 |
| 万向锁 | Gimbal lock | 欧拉角中间角到临界值时两轴重合、自由度塌缩；任何 3 参数全局参数化必有奇点的体现（第 10 章）。 |
| 四元数 | Quaternion | 单位四元数 $(w,x,y,z)$ 编码旋转，无奇点、复合 = 四元数乘、需归一化、双覆盖（$q$ 与 $-q$ 同旋转）、半角 $\theta/2$（第 10 章）。 |
| 球面线性插值 | Spherical linear interpolation (SLERP) | 沿 $S^3$ 大圆以恒定角速度在两旋转间插值，给出最短最匀的旋转过渡（第 10 章）。 |
| 群 | Group | 满足封闭、结合律、单位元、逆元四公理的集合配运算；旋转/刚体运动在复合下成群（第 10 章）。 |
| 李群 | Lie group | 既是群又是光滑流形的对象，SO(3) 是三维非交换李群（第 10 章）。 |
| 李代数 | Lie algebra so(3) | 李群在单位元处的切空间；so(3) = 反对称矩阵 = 角速度向量，平直、便于加减与求梯度（第 10 章）。 |
| 反对称矩阵 / hat 映射 | Skew-symmetric matrix / hat map $[\omega]_\times$ | $\Omega^T=-\Omega$，与角速度向量一一对应，作用在向量上等于与该向量叉乘（第 10 章）。 |
| 指数映射 | Exponential map | $R=\exp([\omega]_\times)$，把李代数（无穷小生成元）积分回李群（有限旋转）的桥（第 10 章）。 |
| 对数映射 | Logarithm map | 指数映射的逆，把旋转矩阵拉回李代数的轴角向量（第 10 章）。 |
| 罗德里格斯公式 | Rodrigues' rotation formula | 指数映射闭式：$R=I+\sin\theta\,K+(1-\cos\theta)K^2$，给出轴角 → 旋转矩阵（第 10 章）。 |
| 齐次坐标 | Homogeneous coordinates | 给三维点补一个 1 升到四维，把仿射变成 $4\times4$ 线性矩阵，使复合 = 矩阵乘、求逆有闭式（第 10 章）。 |
| 局部参考系 / 帧 | Frame / local frame | 一个 SE(3) 元素 = 原点 $\mathbf t$ + 轴朝向 $R$ 的局部坐标系；AlphaFold 给每残基一个 frame（第 10、11 章）。 |
| 相对帧 | Relative frame | $T_j^{-1}T_i$，残基 $i$ 在 $j$ 的 frame 下的位姿，对全局刚体变换天然不变，是 IPA/FAPE 的根（第 10 章）。 |
| 不变性 | Invariance | $f(g\cdot x)=f(x)$：群作用输入而输出不动；预测能量/距离/置信度等标量须对 SE(3) 不变（第 1、10 章）。 |
| 等变性 | Equivariance | $f(g\cdot x)=g\cdot f(x)$：群作用与 $f$ 可交换；预测坐标/力/结构须对 SE(3) 等变（第 1、10 章）。 |
| 表示 / 群表示 | Representation | 同一群作用在不同几何类型上的不同规则：标量 $s\mapsto s$、向量 $\mathbf v\mapsto R\mathbf v$、二阶张量 $M\mapsto RMR^T$（第 1、10 章）。 |
| 不可约表示 | Irreducible representation | 群表示不能再分解的最小积木，球谐函数给出 SO(3) 的各阶不可约表示（第 10 章）。 |
| 归纳偏置 | Inductive bias | 把已知对称性写进网络结构；等变性 = 正确归纳偏置 = 免费无穷数据增强 = 样本高效 + 泛化（第 10 章）。 |
| 不变点注意力 | Invariant point attention (IPA) | AlphaFold2 结构模块的几何注意力：在各残基 frame 下用刚体不变的点对距离调制权重（第 1、10、11 章）。 |
| 张量场网络 | Tensor field networks (TFN) | 用 SO(3) 不可约表示 + 球谐 + CG 张量积逐层严格等变的网络，SE(3)-Transformer/e3nn 同源（第 10 章）。 |
| 球谐函数 | Spherical harmonics | 球面上的傅里叶基，按阶 $\ell$ 给出 SO(3) 的 $2\ell+1$ 维不可约表示（$\ell=0$ 标量/1 向量/2 张量）（第 10 章）。 |
| Clebsch–Gordan 系数 | Clebsch–Gordan coefficients | 两个不可约表示张量积分解回不可约表示之和的规则，等变网络中保持特征类型的乘法表（第 10 章）。 |
| 正交 Procrustes 问题 | Orthogonal Procrustes problem | 求最优正交（旋转）矩阵把两组对应点对齐，$\min\|Q-PR^T\|_F^2$，SVD 给闭式解（第 10 章）。 |
| 帧对齐点误差 | Frame-aligned point error (FAPE) | AlphaFold 损失：在每个残基 frame 下分别算点误差，是 Kabsch 思想的局部化、天然 SE(3) 不变又局部敏感（第 10、11 章）。 |
| 刚体保距引理 | Rigid-motion isometry | $\|(R\mathbf a+\mathbf t)-(R\mathbf b+\mathbf t)\|=\|\mathbf a-\mathbf b\|$：刚体变换不改变两点距离，是 RMSD/距离矩阵/接触图/IPA 不变性的根（第 10 章）。 |
| 等变扩散 | Equivariant diffusion | RFdiffusion 的去噪网络预测随输入一同旋转的 SE(3) 等变向量场来生成蛋白（第 11 章）。 |
| 手性 | Chirality | 结构与其镜像的区分；距离矩阵对反射不变故丢失手性，需额外消歧（第 1、11 章）。 |

---

## 10. 蛋白质折叠 / 计算生物学应用

| 中文名 | English | 解释 |
|---|---|---|
| 坐标矩阵 | Coordinate matrix | 把蛋白 $N$ 个原子/残基坐标按行排成的 $N\times3$ 点云矩阵，紧凑但含 6 个 SE(3) 冗余自由度（第 1、11 章）。 |
| 点云 | Point cloud | 三维空间中一组点的集合，蛋白结构的几何读法（第 1 章）。 |
| 距离矩阵 | Distance matrix | $D_{ij}=\|\mathbf x_i-\mathbf x_j\|$ 的 $N\times N$ 对称矩阵，对刚体变换不变但有手性/可实现性问题（第 1、11 章）。 |
| 接触图 | Contact map | 距离矩阵按阈值二值化得到的 0/1 对称矩阵，等价于图的邻接矩阵；结构指纹、有损非完备（第 1、11 章）。 |
| 格拉姆矩阵（坐标） | Gram matrix (coordinates) | 去中心坐标两两内积 $G=XX^T$，桥接距离与坐标；真实三维结构 ⇒ 秩 $\le3$（第 1、11 章）。 |
| 双中心化 | Double centering | $G=-\tfrac12 JD^{(2)}J$ 从平方距离矩阵恢复 Gram 矩阵的公式，是经典 MDS 第一步（第 11 章）。 |
| 多维标度 / 经典 MDS | (Classical) multidimensional scaling (MDS) | 对 Gram 矩阵做特征分解取前三主方向，把距离矩阵还原为三维坐标（第 1、11 章）。 |
| 内坐标 | Internal coordinates | 用键长键角二面角 $(\phi,\psi,\omega)$ 描述结构，天然 SE(3) 不变但重建时误差沿链累积（第 11 章）。 |
| 二面角 | Dihedral / torsion angle | 主链扭转角 $\phi,\psi,\omega$，是落在环面上的角度变量（第 11 章）。 |
| 结构叠合 | Structure superposition | 在 SE(3) 上求最优刚体变换把两个构象贴合，即 Kabsch/正交 Procrustes 问题（第 11 章）。 |
| Kabsch 算法 | Kabsch algorithm | 中心化 $\to H=P^TQ\to$ SVD $\to R=V\,\text{diag}(1,1,\det(VU^T))\,U^T$ 求最优旋转，含反射修正保手性（第 1、8、10、11 章）。 |
| 简正模式分析 | Normal mode analysis (NMA) | 对能量极小点的 Hessian 特征分解，最低频模式 = 最软方向 = 大尺度功能运动（第 1、7、11 章）。 |
| 弹性网络模型 | Elastic network model (ENM/GNM) | 用统一弹簧 + 接触拓扑代替真实力场，Hessian = 接触图的图拉普拉斯，只凭拓扑预测低频运动（第 7、11 章）。 |
| 直接耦合分析 | Direct coupling analysis (DCA) | 从 MSA 协方差求精度矩阵分离直接 vs 间接耦合，反推残基直接接触（第 11 章）。 |
| 最大熵 Potts 模型 | Maximum-entropy Potts model | DCA 的离散（21 字母）推广，耦合参数 $J_{ij}$ 是精度矩阵非对角块的离散版（第 11 章）。 |
| 多序列比对 | Multiple sequence alignment (MSA) | 同源序列对齐成的矩阵，是带系统偏差、非 i.i.d. 的样本，共进化信号的来源（第 1、11 章）。 |
| 共进化 | Coevolution | 空间接触的残基协同变异，是接触预测的信号源（第 11 章）。 |
| 配对表示 | Pair representation | AlphaFold 中 $L\times L\times c$ 张量，是距离矩阵/接触图的连续高维推广，编码残基对几何关系（第 11 章）。 |
| 三角乘性更新 | Triangle multiplicative update | pair 表示沿中间残基 $k$ 收缩的矩阵乘，把距离三角不等式几何先验焊进网络（第 11 章）。 |
| 残基 frame | Residue frame | AlphaFold 给每个残基一个局部坐标系 = 一个 SE(3) 元素（旋转 + 平移）（第 10、11 章）。 |
| 消息传递 | Message passing | GNN 一层 $H'=\sigma(\hat A HW)$ = 一次稠密 + 一次稀疏矩阵乘，蛋白图稀疏故用 SpMM 高效（第 11 章）。 |
| 过平滑 | Over-smoothing | 深 GNN 反复图扩散使顶点特征趋同，本质是矩阵幂收敛到主特征向量（幂迭代副作用）（第 11 章）。 |
| 等变扩散模型 | Equivariant diffusion model | RFdiffusion 类生成模型，去噪网络对 SE(3) 等变以从学到分布采样蛋白（第 11 章）。 |
| 均方根偏差 | Root-mean-square deviation (RMSD) | 最优叠合后两结构对应原子距离的均方根，对同时施加刚体变换不变的标量（第 1、8、11 章）。 |
| 协方差矩阵（构象） | Covariance matrix (conformational) | MD 轨迹各自由度涨落的协方差，谱给出主要集体运动（第 1 章）。 |
| 群表示（几何类型） | Representation (geometric type) | 标量/向量/张量在 SE(3) 下的不同变换规则，等变网络设计的数学基础（第 1 章）。 |

---

## 11. 人物·文献·里程碑

| 中文名 | English | 解释 |
|---|---|---|
| Gauss | Carl Friedrich Gauss | 高斯消元、最小二乘、正态分布的奠基者，直接解法的源头（第 5 章）。 |
| Cauchy | Augustin-Louis Cauchy | Cauchy–Schwarz 不等式与行列式系统理论的奠基者（第 4、6 章）。 |
| Schwarz | Hermann Schwarz | Cauchy–Schwarz 不等式的共同名祖（第 4 章）。 |
| Gram–Schmidt | Jørgen Gram & Erhard Schmidt | 逐列减投影正交化过程的名祖，QR 分解的雏形（第 4、5 章）。 |
| Householder | Alston Householder | 用正交反射打三角的数值最稳 QR 算法发明者（第 5、9 章）。 |
| Cholesky | André-Louis Cholesky | SPD 矩阵 $LL^T$ 三角平方根分解的名祖（第 5 章）。 |
| Jacobi | Carl Gustav Jacob Jacobi | 雅可比矩阵/行列式与 Jacobi 预条件/迭代的名祖（第 6、9 章）。 |
| Sylvester | James Joseph Sylvester | 惯性定律与正定判据（顺序主子式）的提出者（第 7 章）。 |
| Rayleigh | Lord Rayleigh | 瑞利商与振动模式理论的名祖（第 7 章）。 |
| Courant–Fischer | Richard Courant & Ernst Fischer | 特征值极小极大刻画定理的提出者（第 7 章）。 |
| Weyl | Hermann Weyl | 对称矩阵特征值扰动不等式的提出者（第 7 章）。 |
| Perron–Frobenius | Oskar Perron & Georg Frobenius | 非负不可约矩阵主特征值唯一非负定理的提出者（第 6 章）。 |
| Schur | Issai Schur | Schur 补与 Schur 分解的名祖（分块求逆/三角化）（第 7 章）。 |
| Jordan | Camille Jordan | 亏损矩阵标准形（Jordan 形）的名祖（第 6 章）。 |
| Markov | Andrey Markov | 马尔可夫链与转移矩阵的名祖，幂法/稳态分布背景（第 6 章）。 |
| Eckart–Young | Carl Eckart & Gale Young (1936) | 截断 SVD 是最优低秩近似定理的提出者（Mirsky 1960 推广到酉不变范数）（第 8 章）。 |
| Moore–Penrose | E. H. Moore & Roger Penrose | 广义逆（伪逆）的两位独立提出者（第 3、5、8 章）。 |
| Beltrami / Jordan / Sylvester（SVD 史） | Beltrami, Jordan, Sylvester | 19 世纪独立发现奇异值分解的先驱（第 8 章）。 |
| Lanczos | Cornelius Lanczos | 对称矩阵 Krylov 子空间三对角化算法的发明者（第 9 章）。 |
| Arnoldi | Walter Arnoldi | 非对称矩阵 Krylov 正交化迭代的发明者（第 9 章）。 |
| Krylov | Alexei Krylov | Krylov 子空间（迭代法统一框架）的名祖（第 9 章）。 |
| Hestenes–Stiefel | Hestenes & Stiefel (1952) | 共轭梯度法的原始提出者（第 5、9 章）。 |
| Johnson–Lindenstrauss | William Johnson & Joram Lindenstrauss (1984) | 随机投影近似保距引理的提出者（第 9 章）。 |
| Halko–Martinsson–Tropp（2011） | Halko, Martinsson & Tropp (2011) | 随机化 SVD 的奠基综述，巨型矩阵近似分解的现代范式（第 8、9 章）。 |
| Euler | Leonhard Euler | 欧拉旋转定理与欧拉角的名祖（第 10 章）。 |
| Rodrigues | Olinde Rodrigues | 轴角 → 旋转矩阵闭式公式的名祖（指数映射）（第 10 章）。 |
| Hamilton | William Rowan Hamilton | 四元数的发明者，无奇点旋转表示的源头（第 10 章）。 |
| Lie | Sophus Lie | 李群/李代数理论的奠基者，连续对称性的语言（第 10 章）。 |
| Clebsch–Gordan | Alfred Clebsch & Paul Gordan | 不可约表示张量积分解系数的名祖，等变网络的乘法表（第 10 章）。 |
| Kabsch | Wolfgang Kabsch (1976) | 用 SVD 求最优叠合旋转的算法发明者，结构生物学日用工具（第 1、8、10、11 章）。 |
| Procrustes | Procrustes (orthogonal Procrustes problem) | 取自希腊神话，指「把一组点最优对齐到另一组」的正交对齐问题（第 8、10 章）。 |
| Morcos 等（2011，DCA） | Morcos et al., PNAS 2011 | 全局统计模型 DCA 从 MSA 捕获天然残基接触的里程碑论文（第 11 章）。 |
| AlphaFold2（Jumper 2021） | Jumper et al., Nature 2021 | 端到端结构预测（Evoformer + IPA + FAPE）的里程碑，线代工具的集大成（第 1、11 章）。 |
| RFdiffusion（Watson 2023） | Watson et al., Nature 2023 | 用 SE(3) 等变扩散从头设计蛋白的代表工作（第 11 章）。 |
| SE(3)-Transformer | Fuchs et al., NeurIPS 2020 | 对 SE(3) 严格等变的注意力网络，TFN/e3nn 同源谱系（第 10 章）。 |
| Strang《Linear Algebra and Its Applications》 | Gilbert Strang | 「四个基本子空间」几何直觉教学的经典，本课主参考之一（第 11 章）。 |
| Trefethen–Bau《Numerical Linear Algebra》 | Trefethen & Bau | 数值线代（稳定性/条件数/迭代法）的现代经典，呼应第 9 章（第 9、11 章）。 |
| Golub–Van Loan《Matrix Computations》 | Golub & Van Loan | 矩阵计算的权威工具书，分解/算法的全集（第 11 章）。 |
| Anfinsen 热力学假说 | Anfinsen's thermodynamic hypothesis | 天然态 = 生理条件下自由能最低的态，折叠研究的物理出发点（第 1 章）。 |

---

> **一句话收束**：这张表是全课术语的索引而非替代——遇到不熟的词先在此定位中英名与所在章，再回正文精读。把「中文直觉 + 英文术语 + 所在章」三者绑定，再加上「这个矩阵在做什么几何变换」的一句话直觉，你读折叠/计算生物学论文方法部分的速度与深度都会上一个台阶。
