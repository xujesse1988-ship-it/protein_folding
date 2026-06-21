# 蛋白质折叠深度入门课程 · 中英文术语对照表（GLOSSARY）

> 速查约定：每条格式为 **中文名（English term，常用缩写）—— 一句话精准解释**。
> 分组顺序大致对应课程模块；同一术语若跨章出现，归入其最核心的主题组。

---

## 目录

1. [生物化学基础](#1-生物化学基础)
2. [结构层级与几何](#2-结构层级与几何)
3. [结构表示与比较度量](#3-结构表示与比较度量)
4. [物理与热力学](#4-物理与热力学)
5. [经典计算方法（AlphaFold 之前）](#5-经典计算方法alphafold-之前)
6. [机器学习与深度学习](#6-机器学习与深度学习)
7. [AlphaFold 与现代结构预测方法](#7-alphafold-与现代结构预测方法)
8. [蛋白质设计（逆问题）](#8-蛋白质设计逆问题)
9. [工具、数据库与文件格式](#9-工具数据库与文件格式)
10. [评测、机构与里程碑](#10-评测机构与里程碑)

---

## 1. 生物化学基础

| 中文名（English，缩写） | 解释 |
|---|---|
| 蛋白质（Protein）| 由氨基酸经肽键连成的生物大分子，折叠成特定三维结构后行使功能。 |
| 中心法则（Central Dogma）| 遗传信息流向 DNA → RNA → 蛋白质，是序列得以编码结构与功能的源头。 |
| 氨基酸（Amino acid）| 蛋白质的基本单元，含 α 碳、氨基、羧基和决定其性质的侧链 R。 |
| α 碳（Alpha carbon，Cα）| 同时连接氨基、羧基、氢和侧链的中心碳原子，是主链几何的参考点。 |
| 侧链（Side chain，R group）| 区分 20 种氨基酸的可变基团，决定其疏水性、电荷与化学反应性。 |
| 主链 / 骨架（Backbone / Main chain）| 由重复的 N–Cα–C(=O) 单元构成的蛋白质共价主干。 |
| 残基（Residue）| 氨基酸在肽链中失去一分子水后的剩余部分，链中每个单元即一个残基。 |
| 疏水氨基酸（Hydrophobic residue）| 侧链非极性、倾向于埋藏在蛋白内部以避开水的氨基酸。 |
| 极性不带电氨基酸（Polar uncharged residue）| 侧链可形成氢键但不带净电荷的氨基酸。 |
| 带电氨基酸（Charged residue）| 侧链在生理 pH 下带正电（如 Lys、Arg）或负电（如 Asp、Glu）的氨基酸。 |
| 甘氨酸（Glycine，Gly/G）| 侧链仅为氢的最小氨基酸，赋予主链极大柔性。 |
| 脯氨酸（Proline，Pro/P）| 侧链环回主链 N 的"亚氨基酸"，限制 φ 角并常打断二级结构。 |
| 半胱氨酸（Cysteine，Cys/C）| 侧链含巯基（–SH），可氧化形成二硫键稳定结构。 |
| 肽键（Peptide bond）| 相邻氨基酸的羧基与氨基脱水缩合形成的酰胺共价键。 |
| 脱水缩合（Condensation / Dehydration）| 形成肽键时脱去一分子水的反应。 |
| 肽平面（Peptide plane）| 肽键因部分双键性质而呈刚性共平面的六原子单元。 |
| 部分双键性质（Partial double-bond character）| 肽键 C–N 因共振而带部分双键属性，使 ω 角锁定在约 180°。 |
| 二硫键（Disulfide bond / bridge）| 两个半胱氨酸巯基氧化形成的共价交联，是最强的稳定相互作用。 |
| 氢键（Hydrogen bond）| 供体–H 与受体间的弱静电吸引，是二级结构形成的关键。 |
| 疏水效应（Hydrophobic effect）| 非极性基团聚集以减少水的有序化、增加体系熵的驱动力。 |
| 范德华力（Van der Waals force）| 原子间短程的弱吸引/排斥作用，决定紧密堆积。 |
| 静电相互作用（Electrostatic interaction）| 带电基团间的库仑吸引或排斥。 |
| 盐桥（Salt bridge）| 异号带电侧链间兼具静电与氢键的相互作用。 |
| 溶剂 / 水（Solvent / Water）| 折叠的环境介质，其熵效应是疏水驱动力的物理来源。 |
| 变性（Denaturation）| 在加热、变性剂等作用下结构展开、丧失功能的过程。 |
| 复性 / 重折叠（Refolding）| 去除变性条件后蛋白自发恢复天然结构的过程。 |

---

## 2. 结构层级与几何

| 中文名（English，缩写） | 解释 |
|---|---|
| 一级结构（Primary structure）| 氨基酸的线性排列顺序，即序列本身。 |
| 二级结构（Secondary structure）| 由主链氢键形成的局部规则构象，如 α 螺旋、β 折叠。 |
| 三级结构（Tertiary structure）| 单条多肽链折叠成的完整三维空间排布。 |
| 四级结构（Quaternary structure）| 多条多肽链（亚基）组装成的复合体结构。 |
| α 螺旋（Alpha helix，α-helix）| 右手螺旋，每圈约 3.6 残基，靠 i→i+4 主链氢键稳定。 |
| β 折叠 / β 片层（Beta sheet，β-sheet）| 由相邻 β 链经主链氢键并排形成的片状结构。 |
| β 链（Beta strand，β-strand）| 构成 β 片层的单条伸展肽段。 |
| 平行 / 反平行（Parallel / Antiparallel）| β 链走向相同或相反的两种片层排列方式。 |
| β 转角（Beta turn）| 使主链反向的短回折，常连接反平行 β 链。 |
| 二面角 / 扭转角（Dihedral / Torsion angle）| 由连续四个原子定义、绕中间键旋转的角度，可用向量叉乘计算。 |
| φ 角（Phi，φ）| 绕 N–Cα 键的主链二面角。 |
| ψ 角（Psi，ψ）| 绕 Cα–C 键的主链二面角。 |
| ω 角（Omega，ω）| 绕肽键 C–N 的二面角，因部分双键性质近 180°。 |
| 侧链扭转角（Side-chain torsion / Chi angle，χ）| 描述侧链构象的二面角，结合主链 frame 可还原全原子坐标。 |
| 拉氏图 / 拉马钱德兰图（Ramachandran plot）| 以 φ–ψ 为坐标展示主链构象允许区的图，反映位阻约束。 |
| 允许区（Allowed region）| 拉氏图中无显著原子位阻、构象可行的区域。 |
| 位阻 / 立体位阻（Steric clash / hindrance）| 原子靠得过近产生的排斥，限定可行构象。 |
| 笛卡尔坐标（Cartesian coordinates）| 用 (x, y, z) 直接描述每个原子位置的表示。 |
| 内坐标（Internal coordinates）| 用键长、键角、二面角描述结构的表示，与笛卡尔坐标可互换。 |
| 残基刚体 / 局部坐标系（Residue frame）| 把每个残基视为带旋转+平移的刚体，是 AF2 结构模块的核心表示。 |
| 全原子表示（All-atom representation）| 显式记录所有原子坐标的结构表示。 |
| 仅 Cα 表示（Cα-only / Cα-trace）| 仅保留每残基 α 碳的简化骨架表示。 |
| 内在无序区 / 蛋白（Intrinsically disordered region / protein，IDR / IDP）| 天然状态下无稳定单一构象的柔性区段或蛋白。 |
| 结构域（Domain）| 可独立折叠、常具独立功能的结构单元。 |
| 模体（Motif）| 反复出现的小型结构或序列模式，常对应功能位点。 |

---

## 3. 结构表示与比较度量

| 中文名（English，缩写） | 解释 |
|---|---|
| 距离矩阵（Distance matrix）| 记录所有残基对间距离的矩阵，是旋转平移不变的结构指纹。 |
| 接触图（Contact map）| 将残基对距离阈值化得到的 0/1 图，标示空间上邻近的残基对。 |
| 接触预测（Contact prediction）| 从序列/MSA 预测哪些残基对在空间上接触的任务。 |
| 距离分布图（Distogram）| 把残基对距离离散为多个区间的概率分布，AF2 的关键中间输出。 |
| 叠合 / 对齐（Superposition / Alignment）| 将两个结构最优重叠以便逐原子比较的操作。 |
| 均方根偏差（Root-mean-square deviation，RMSD）| 叠合后对应原子距离的均方根，越小越相似但对局部错配敏感。 |
| 全局距离测试（Global Distance Test – Total Score，GDT-TS）| 多距离阈值下可叠合残基比例的综合得分，CASP 常用。 |
| TM-score（Template Modeling score）| 长度归一化的相似度（0–1），对整体拓扑稳健，>0.5 约为同折叠。 |
| 局部距离差异检验（local Distance Difference Test，lDDT）| 基于局部原子间距离一致性、无需叠合的逐残基相似度度量。 |

---

## 4. 物理与热力学

| 中文名（English，缩写） | 解释 |
|---|---|
| 吉布斯自由能（Gibbs free energy，G）| G = H − TS，其极小对应热力学最稳定状态。 |
| 焓（Enthalpy，H）| 体系的热含量，与相互作用能（氢键、范德华等）相关。 |
| 熵（Entropy，S）| 体系无序度/可及微观状态数的度量。 |
| 热力学假说（Thermodynamic hypothesis）| Anfinsen 提出：天然结构对应自由能全局最小，序列即决定结构。 |
| Anfinsen 实验（Anfinsen's experiment）| 核糖核酸酶变性后自发复性，证明序列编码结构的经典实验。 |
| Levinthal 佯谬（Levinthal's paradox）| 构象空间天文数字却能毫秒折叠，说明折叠非随机搜索。 |
| 折叠（Folding）| 多肽链自发达到天然三维结构的物理过程。 |
| 结构预测（Structure prediction）| 由序列计算预测三维结构的计算任务（区别于物理折叠过程）。 |
| 能量地形 / 自由能地形（Energy landscape）| 构象空间上自由能的高维曲面，折叠是在其上的下行过程。 |
| 折叠漏斗（Folding funnel）| 漏斗状能量地形，多条路径汇向天然态底部，调和 Levinthal 佯谬。 |
| 受挫（Frustration）| 局部相互作用无法同时最优化、造成地形粗糙的现象。 |
| 熔球态（Molten globule）| 已塌缩、具二级结构但侧链尚未锁定的折叠中间态。 |
| 折叠中间体（Folding intermediate）| 折叠路径上短暂存在的部分折叠构象。 |
| 过渡态（Transition state）| 折叠路径上自由能最高的瓶颈构象。 |
| 折叠速率（Folding rate）| 折叠反应的快慢，反映动力学难度。 |
| 两态 / 多态折叠（Two-state / Multi-state folding）| 折叠是否经过可检测中间体的两类动力学行为。 |
| 力场（Force field）| 用解析能量项近似分子势能的参数化模型，如 AMBER、CHARMM。 |
| 键长项 / 键角项（Bond / Angle term）| 力场中约束共价几何的简谐能量项。 |
| 二面角项（Torsion / Dihedral term）| 力场中描述绕键旋转能量势垒的项。 |
| 伦纳德-琼斯势（Lennard-Jones potential，LJ）| 描述范德华吸引与近程排斥的经典势函数。 |
| 库仑项（Coulomb term）| 力场中描述带电粒子静电相互作用的能量项。 |
| 隐式 / 显式溶剂（Implicit / Explicit solvent）| 将水近似为连续介质或显式建模每个水分子的两种处理。 |
| 分子动力学（Molecular Dynamics，MD）| 对牛顿运动方程数值积分模拟原子随时间运动的方法。 |
| 时间步（Time step）| MD 积分的离散步长，通常为飞秒级。 |
| 时间尺度鸿沟（Timescale gap）| 飞秒级步长与微秒至秒级真实折叠间的巨大差距。 |
| 增强采样（Enhanced sampling）| 加速跨越能垒、探索构象空间的一类方法。 |
| 副本交换分子动力学（Replica Exchange MD，REMD）| 多温度副本间交换以加速采样的增强采样方法。 |
| 元动力学（Metadynamics）| 沿集合变量填充历史势以逃逸局部极小的采样方法。 |
| 马尔可夫状态模型（Markov State Model，MSM）| 将 MD 轨迹离散为状态及其转移概率以刻画动力学的模型。 |
| 非凸优化（Non-convex optimization）| 存在多个局部极小的优化问题，折叠即在此类能量函数上求全局最优/采样。 |

---

## 5. 经典计算方法（AlphaFold 之前）

| 中文名（English，缩写） | 解释 |
|---|---|
| 同源建模 / 比较建模（Homology / Comparative modeling）| 借助已知同源模板结构建模目标蛋白的方法。 |
| 模板（Template）| 用作建模参照的已知实验结构。 |
| 穿线法 / 折叠识别（Threading / Fold recognition）| 将目标序列"穿"入已知折叠库以识别最匹配折叠的方法。 |
| 从头预测（Ab initio / De novo prediction）| 不依赖模板、仅从物理与统计原理预测结构的方法。 |
| 片段组装（Fragment assembly）| 用短肽段构象片段拼装并优化全长结构，Rosetta 为代表。 |
| Rosetta | 集片段组装、打分与设计于一体的经典结构建模软件套件。 |
| 打分函数 / 能量函数（Scoring / Energy function）| 评估候选结构优劣的函数，分物理基与知识基两类。 |
| 物理基势（Physics-based potential）| 基于物理相互作用项构建的能量函数。 |
| 知识基 / 统计势（Knowledge-based / Statistical potential）| 从已知结构统计规律推导的能量函数，如 DOPE。 |
| 多序列比对（Multiple Sequence Alignment，MSA）| 将同源序列对齐成的矩阵，蕴含进化与共进化信号。 |
| 共进化（Coevolution）| 空间接触残基对在进化中相关突变的现象，是接触预测的信号源。 |
| 直接耦合分析（Direct Coupling Analysis，DCA）| 从 MSA 中区分直接与间接相关、推断接触的统计方法。 |
| Potts 模型（Potts model）| 刻画 MSA 中残基对耦合的统计物理模型，DCA 的基础。 |
| 模拟退火（Simulated annealing）| 逐步降温以逃逸局部极小的随机优化方法。 |
| 蒙特卡洛（Monte Carlo，MC）| 用随机抽样探索构象/解空间的方法。 |
| 梯度法（Gradient-based optimization）| 沿能量梯度方向迭代优化的方法。 |
| 采样瓶颈（Sampling bottleneck）| 构象空间过大导致难以充分采样，经典方法两大瓶颈之一。 |
| 打分瓶颈（Scoring bottleneck）| 能量函数不够准、难以从候选中辨出天然态，另一大瓶颈。 |

---

## 6. 机器学习与深度学习

| 中文名（English，缩写） | 解释 |
|---|---|
| 线性代数（Linear algebra）| 矩阵运算、特征分解、SVD、旋转矩阵等，是几何与模型的数学基础。 |
| 奇异值分解（Singular Value Decomposition，SVD）| 将矩阵分解为正交因子，用于降维与结构叠合求最优旋转。 |
| 旋转矩阵（Rotation matrix）| 表示三维旋转的正交矩阵，描述残基 frame 朝向。 |
| 信息熵（Entropy / KL divergence，KL）| 信息论中不确定性度量与两分布差异度量，用于损失与序列保守性分析。 |
| 监督学习（Supervised learning）| 用带标注数据学习输入到输出映射的范式。 |
| 损失函数（Loss function）| 量化预测与真值差距、供优化最小化的目标函数。 |
| 过拟合 / 泛化（Overfitting / Generalization）| 模型过度拟合训练数据与其对新数据的表现能力。 |
| 正则化（Regularization）| 抑制过拟合、提升泛化的约束技术。 |
| 梯度下降（Gradient descent）| 沿损失负梯度迭代更新参数的优化算法。 |
| Adam 优化器（Adam optimizer）| 结合动量与自适应学习率的常用优化算法。 |
| 反向传播（Backpropagation）| 按链式法则反向计算各参数梯度的算法。 |
| 自动微分（Automatic differentiation，autodiff）| 框架自动计算复合函数精确梯度的机制。 |
| 多层感知机（Multilayer Perceptron，MLP）| 全连接前馈神经网络。 |
| 卷积神经网络（Convolutional Neural Network，CNN）| 用局部卷积核提取空间特征的网络，早期用于接触图预测。 |
| 循环神经网络 / 长短期记忆（RNN / Long Short-Term Memory，LSTM）| 处理序列、保留时序依赖的网络结构。 |
| Transformer | 完全基于注意力机制的网络架构，是 AF2 Evoformer 的核心范式。 |
| 注意力机制（Attention）| 按相关性对输入元素加权聚合信息的机制。 |
| 自注意力（Self-attention）| 序列内部元素相互计算注意力以建模长程依赖。 |
| 多头注意力（Multi-head attention）| 并行多组注意力以捕捉不同子空间关系。 |
| 图神经网络（Graph Neural Network，GNN）| 在图结构上通过消息传递学习节点/边表示的网络。 |
| 消息传递（Message passing）| 节点间沿边交换并聚合信息的 GNN 基本机制。 |
| 几何深度学习（Geometric deep learning，GDL）| 在非欧/几何结构数据上构建模型的范式。 |
| 等变性 / 不变性（Equivariance / Invariance）| 输出随输入对称变换相应变化 / 保持不变的性质。 |
| SE(3) / E(3) 群（SE(3) / E(3) group）| 三维旋转平移（含反射）的对称群，3D 结构预测须对其等变。 |
| 表示学习（Representation learning）| 自动学习数据有用特征表示的范式。 |
| 生成模型（Generative model）| 学习数据分布并采样生成新样本的模型。 |
| 变分自编码器（Variational Autoencoder，VAE）| 学习隐变量分布以生成数据的生成模型。 |
| 扩散模型（Diffusion model）| 通过逐步去噪从噪声生成样本的生成模型，用于 AF3 与 RFdiffusion。 |

---

## 7. AlphaFold 与现代结构预测方法

| 中文名（English，缩写） | 解释 |
|---|---|
| AlphaFold（AF）| DeepMind 的蛋白结构预测系统系列，开启端到端深度学习预测时代。 |
| AlphaFold2（AF2）| 2020 年在 CASP14 达到近实验精度的里程碑模型。 |
| Evoformer | AF2 核心模块，对 MSA 表示与残基对表示双轨交互处理。 |
| MSA 表示（MSA representation）| Evoformer 中编码多序列比对的特征张量。 |
| 残基对表示（Pair representation）| 编码残基对间几何/关系信息的特征张量。 |
| 行 / 列注意力（Row / Column-wise attention）| 在 MSA 表示上分别沿序列内与序列间方向施加的注意力。 |
| 三角乘法更新（Triangle multiplication）| 用第三残基约束更新残基对表示，体现三角一致性的几何动机。 |
| 三角注意力（Triangle attention）| 沿三角关系对残基对施加注意力，强化距离一致性约束。 |
| 三角不等式约束（Triangle inequality constraint）| 三残基两两距离须自洽，是三角操作的几何依据。 |
| 结构模块（Structure Module）| AF2 中由抽象表示生成全原子坐标的几何模块。 |
| 不变点注意力（Invariant Point Attention，IPA）| 在残基 frame 上施加、对全局刚体变换不变的注意力。 |
| 回收 / 循环精修（Recycling）| 将输出反馈为输入多轮迭代以逐步精化预测。 |
| FAPE 损失（Frame Aligned Point Error，FAPE）| 在各残基局部 frame 下度量原子误差、兼具等变/不变性的损失。 |
| 距离分布损失（Distogram loss）| 监督预测残基对距离分布的损失项。 |
| 辅助损失（Auxiliary loss）| 训练中辅助主任务、稳定/加速收敛的附加损失。 |
| 置信度预测头（Confidence head）| 预测模型自身可靠度（pLDDT、PAE）的输出分支。 |
| pLDDT（predicted lDDT）| 每残基置信度（0–100），越高表示局部结构越可信。 |
| 预测对齐误差（Predicted Aligned Error，PAE）| 残基对相对位置不确定性，用于判断结构域间相对摆放可信度。 |
| 自蒸馏（Self-distillation）| 用模型自身对未标注序列的预测扩充训练数据。 |
| AlphaFold3（AF3）| 2024 年转向扩散式生成、统一预测蛋白/核酸/配体/离子复合物。 |
| Pairformer | AF3 中取代部分 Evoformer、聚焦残基对表示的处理模块。 |
| 基于扩散的结构生成（Diffusion-based structure generation）| 以去噪扩散直接生成原子坐标的预测范式（AF3）。 |
| RoseTTAFold（RF）| Baker 实验室的深度学习预测模型，采用三轨架构。 |
| 三轨架构（Three-track architecture）| 同时处理序列、残基对与三维坐标三条信息轨道的设计。 |
| RoseTTAFold All-Atom（RFAA）| RoseTTAFold 的全原子版本，支持小分子/配体等非蛋白成分。 |
| 蛋白质语言模型（Protein Language Model，pLM）| 在海量序列上自监督训练、学习序列内在规律的模型。 |
| ESM（Evolutionary Scale Modeling）| Meta 推出的蛋白语言模型系列。 |
| ESMFold | 基于 ESM、无需 MSA 的快速结构预测模型，精度换速度。 |
| AlphaFold-Multimer | 面向蛋白复合物/界面预测的 AF2 扩展。 |
| 复合物 / 界面预测（Complex / Interface prediction）| 预测多链组装及其相互作用界面的任务。 |
| MSA-based vs 单序列（pLM-based）| 依赖多序列比对 vs 仅用单序列（语言模型）两条技术路线的权衡。 |

---

## 8. 蛋白质设计（逆问题）

| 中文名（English，缩写） | 解释 |
|---|---|
| 正问题 / 逆问题（Forward / Inverse problem）| 由序列求结构（预测）vs 由目标结构/功能反求序列（设计）。 |
| 蛋白质设计（Protein design）| 设计满足指定结构或功能的全新氨基酸序列/蛋白。 |
| 逆折叠（Inverse folding）| 给定骨架求可折叠成它的序列，即固定骨架序列设计。 |
| 固定骨架设计（Fixed-backbone design）| 在给定三维骨架上优化侧链/序列的设计范式。 |
| ProteinMPNN | 基于 GNN 的高效逆折叠/序列设计模型，影响广泛。 |
| 从头骨架生成（De novo backbone generation）| 不依赖现有骨架、生成全新三维骨架的任务。 |
| RFdiffusion | 用扩散模型生成全新蛋白骨架的方法（Baker 实验室）。 |
| Chroma | 基于扩散的可编程蛋白生成模型。 |
| 模体支架化（Motif scaffolding）| 围绕给定功能模体设计承载它的骨架。 |
| 结合蛋白设计（Binder design）| 设计能特异结合给定靶标的蛋白。 |
| 定向进化（Directed evolution）| 通过迭代突变与选择优化蛋白的实验方法，与计算设计互补。 |
| 湿实验验证（Wet-lab validation）| 通过实验表达与表征确认设计是否成功的闭环环节。 |
| 酶设计（Enzyme design）| 设计具催化活性的蛋白。 |
| 抗体 / 抗原设计（Antibody / Antigen design）| 设计抗体或免疫原（如疫苗抗原）的应用方向。 |

---

## 9. 工具、数据库与文件格式

| 中文名（English，缩写） | 解释 |
|---|---|
| 蛋白质数据库（Protein Data Bank，PDB）| 收录实验测定三维结构的主数据库，也指其经典坐标文件格式。 |
| mmCIF（macromolecular CIF）| PDB 推荐的现代结构文件格式，扩展性优于旧版 PDB 文本格式。 |
| FASTA | 以 `>` 标题行加序列的纯文本序列格式。 |
| UniProt | 主流蛋白质序列与功能注释数据库。 |
| AlphaFold 数据库（AlphaFold DB）| 收录约 2 亿条 AF2 预测结构的数据库。 |
| Pfam / InterPro | 蛋白质家族、结构域与功能特征的分类注释数据库。 |
| PyMOL | 广泛使用的分子结构可视化软件。 |
| ChimeraX | 现代分子可视化与分析软件。 |
| ColabFold | 在 Google Colab 上便捷运行 AlphaFold2/MSA 搜索的开源流程。 |
| Biopython | 解析序列/结构、做生信计算的 Python 库。 |
| Biotite | 用于结构与序列分析的现代 Python 库。 |
| 按 pLDDT 着色（Color by pLDDT）| 用置信度给预测结构着色以快速判读可靠区域的可视化惯例。 |
| GPU / 算力（GPU / Compute）| 运行深度学习预测所需的图形处理器算力资源。 |

---

## 10. 评测、机构与里程碑

| 中文名（English，缩写） | 解释 |
|---|---|
| CASP（Critical Assessment of Structure Prediction）| 始于 1994 的盲测竞赛，结构预测领域的"奥林匹克"与公认标尺。 |
| CASP14 | 2020 年 AlphaFold2 取得近实验精度突破的那届评测。 |
| 盲测（Blind assessment）| 预测时结构尚未公开、避免过拟合的客观评估机制。 |
| DeepMind | 开发 AlphaFold 系列的 AI 研究机构。 |
| Baker 实验室（Baker Lab）| David Baker 团队，Rosetta、RoseTTAFold、ProteinMPNN、RFdiffusion 的发源地。 |
| AlQuraishi 实验室（AlQuraishi Lab）| 端到端可微结构预测与蛋白机器学习的代表性研究组。 |
| 2024 诺贝尔化学奖（2024 Nobel Prize in Chemistry）| 一半授予计算蛋白设计（Baker），一半授予 AlphaFold（Hassabis & Jumper）。 |
| MLSB（Machine Learning for Structural Biology）| NeurIPS 关联的结构生物学机器学习研讨会。 |
| Rosetta Commons | 维护与开发 Rosetta 软件的学术社区联盟。 |
| 突变效应 / 稳定性变化（ΔΔG）| 点突变引起的折叠自由能变化，量化对稳定性的影响。 |
| bioRxiv / arXiv | 生物学与定量科学的预印本平台，前沿成果首发地。 |

---

> 备注：缩写在领域内广泛通用时随术语标注（如 AF2、MSA、RMSD、pLDDT）。阅读英文文献时，建议结合上下文确认缩写在具体语境中的指代。
