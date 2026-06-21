# 第 10 章 动手实践指南

> 前面九章里，我们把蛋白质折叠从生物化学（第 2 章）、几何与数学表示（第 3 章）、物理与热力学（第 4 章），一路讲到 AlphaFold2（第 7 章）与折叠的逆问题——蛋白质设计（第 9 章）。这一章的目标只有一个：**让你今天就能动手**。
>
> 我们会用尽量"开箱即用"的工具，带你走完一个完整的小研究闭环：找数据 → 读文件 → 跑预测 → 可视化 → 量化对比 → 用代码分析结构。读完本章，你应该能独立完成你的"第一个项目"，并且知道下一步该往哪查。

---

## 10.0 本章地图与心态

```
            ┌─────────────┐
   序列      │  数据库      │   结构
  UniProt ──▶│  (10.1)     │◀── PDB / AlphaFold DB
            └──────┬──────┘
                   │
            ┌──────▼──────┐
            │  文件格式    │   FASTA / PDB / mmCIF
            │  (10.2)     │
            └──────┬──────┘
          ┌────────┼────────┐
          ▼        ▼        ▼
   ┌──────────┐ ┌────────┐ ┌──────────┐
   │ 可视化    │ │ 跑预测  │ │ 编程分析  │
   │ PyMOL    │ │ColabFold│ │Biopython │
   │ ChimeraX │ │ESMFold │ │ numpy    │
   │ (10.3)   │ │ (10.4) │ │ (10.5)   │
   └──────────┘ └────────┘ └──────────┘
                   │
            ┌──────▼──────┐
            │  第一个项目  │  (10.6)
            └─────────────┘
```

**心态提示**：入门阶段，"跑通一个完整流程"比"完全理解每一行"更重要。先让管道流起来，再回头深挖。本章所有代码都是**最小可复制骨架**（minimal reproducible skeleton）——能跑、能改，但不是生产级。遇到报错是常态，把报错信息复制去搜索，是这个领域真实的工作方式。

---

## 10.1 关键数据库：你的原材料仓库

蛋白质研究的"原材料"是**序列**（sequence）和**结构**（structure）。下面四个数据库覆盖了入门所需的几乎全部数据。

| 数据库 | 全称 | 内容 | 你用它来做什么 |
|---|---|---|---|
| **PDB** | Protein Data Bank | 实验测定的 3D 结构 | 拿到"真值"结构，做对比基准 |
| **UniProt** | Universal Protein Resource | 蛋白质序列 + 功能注释 | 拿序列、查功能、找物种来源 |
| **AlphaFold DB** | AlphaFold Protein Structure Database | AI 预测的 3D 结构 | 海量预测结构，免跑模型直接下载 |
| **Pfam / InterPro** | Protein families / Integrated resource | 蛋白质家族、结构域 | 判断"这段序列属于哪个家族" |

### 10.1.1 PDB —— 实验结构的"金标准库"

- 网址：`https://www.rcsb.org`（RCSB 是 PDB 的美国镜像，功能最全）。
- 每个结构有一个 **4 位 PDB ID**，如 `1UBQ`（泛素 ubiquitin）、`1CRN`（crambin，一个常用的小蛋白）。
- 结构来自 **X 射线晶体学**（X-ray crystallography）、**核磁共振**（NMR）或**冷冻电镜**（cryo-EM，详见第 2 章对实验方法的介绍）。
- **关键质量指标**：**分辨率**（resolution，单位 Å，**越小越好**，<2.0 Å 算高质量）和 **R-free**（拟合质量，越小越好）。NMR 结构通常没有分辨率，而是给出一组（10–20 个）"模型"（model）。

> ⚠️ **诚实提醒**：实验结构是"真值"，但**不是完美真值**。晶体衍射给出的是晶体中大量晶胞、整个曝光时间内电子密度的**系综/时空平均**（spatio-temporal ensemble average）；建模时，这团电子密度**通常被拟合为一个（或少数几个 alternate conformation）代表构象**。结构可能缺失柔性区域（电子密度看不清的残基在文件里直接缺席），还可能受晶体堆积（crystal packing）影响。所以"实验 vs 预测不一致"未必是预测错了。

### 10.1.2 UniProt —— 序列与功能的百科

- 网址：`https://www.uniprot.org`。
- 核心标识符是 **UniProt accession**，如 `P0DTC2`（新冠病毒刺突蛋白 spike）。
- 分两库：**Swiss-Prot**（人工审编，高质量，带 `reviewed` 标记）和 **TrEMBL**（自动注释，量大但未审编）。
- 每个条目里你能查到：序列、物种、功能描述、结构域注释、已知的 PDB 结构交叉引用、以及**直接链接到 AlphaFold DB 的预测结构**。

### 10.1.3 AlphaFold DB —— 2 亿+ 预测结构的免费仓库

- 网址：`https://alphafold.ebi.ac.uk`。
- 由 DeepMind 与 EMBL-EBI 合作，覆盖 UniProt 中**几乎所有**蛋白质（2 亿+ 条），用 AlphaFold2 预测（第 7 章）。
- 用 UniProt accession 直接检索，下载得到 mmCIF/PDB 文件，**每个原子带 pLDDT 置信度**（写在 B-factor 列里，见 10.4.3）。
- **数据许可**：AlphaFold DB 的预测结构以 **CC-BY 4.0** 授权，可自由使用（包括商用），但**用于论文/报告时应引用** Jumper et al. 2021（AF2 原文）与 Varadi et al. 2022（数据库原文）。
- **省时关键**：如果你的蛋白已经在库里，**不必自己跑 ColabFold**，直接下载即可。自己跑预测只在以下情况需要：库里没有（如你新设计的序列）、你想要多构象/多序列变体、或你想自己控制 MSA 与模板。

### 10.1.4 Pfam / InterPro —— 把序列归类到"家族"

- **Pfam**：基于**隐马尔可夫模型**（HMM, Hidden Markov Model）的蛋白质家族库。给一条序列，它告诉你包含哪些已知**结构域**（domain）。注意：**Pfam 的独立门户（原 pfam.xfam.org）自 2022 年起已停止维护并整合进 InterPro**；Pfam 的家族定义与 HMM 模型仍然存在，但现在通过 InterPro 访问。
- **InterPro**：`https://www.ebi.ac.uk/interpro`，整合了 Pfam、PROSITE、CATH 等多个家族/结构域资源，是现在的统一入口。
- 为什么对你重要：**家族信息 ≈ 进化信息**。同一家族的蛋白共享祖先，序列上的"共进化"（coevolution）信号正是 AlphaFold 之类方法构建残基对表征的核心依据（详见第 5 章 MSA 与共进化、第 7 章 Evoformer）。

> **动手 10.A**：打开 RCSB，搜索 `1UBQ`。记录它的：实验方法、分辨率、链（chain）数量、残基（residue）数量。再去 UniProt 搜 `P0DTC2`（新冠刺突蛋白），仅作"如何检索一个条目、如何找到它在 AlphaFold DB 的预测结构链接"的演示——**这条序列长达 1273 残基、含大量低 pLDDT 区，不适合作为后续分析的对象**。真正下载来做分析的，请选一个小而干净的蛋白（如 `1UBQ` 或 `1CRN`，见动手 10.D）。

---

## 10.2 文件格式：怎么读懂一行原子坐标

### 10.2.1 FASTA —— 最简单的序列格式

纯文本，一行 `>` 开头的标题，后面跟序列（单字母氨基酸编码，见第 2 章）：

```
>sp|P0DTC2|SPIKE_SARS2 Spike glycoprotein OS=...
MFVFLVLLPLVSSQCVNLTTRTQLPPAYTNSFTRGVYYPDKVFRSSVLHSTQDLFLPFFSN
VTWFHAIHVSGTNGTKRFDNPVLPFNDGVYFASTEKSNIIRGWIFGTTLDSKTQSLLIVNN
```

- 标题行格式：`>db|accession|name description`。
- 序列用 20 种标准氨基酸字母（外加 `X` 未知、`U` 硒代半胱氨酸等）。
- **这是 ColabFold 等预测工具的标准输入**。

### 10.2.2 PDB 格式 —— 经典的固定列文本

`.pdb` 是一种**按列对齐**的古老纯文本格式。最重要的是 `ATOM` / `HETATM` 行。看一行真实的例子：

```
ATOM     12  CA  ASP A   2      26.266  25.413   2.842  1.00 16.74           C
^----^ ^--^ ^--^ ^-^ ^ ^--^      ^----^ ^----^ ^----^ ^--^ ^---^         ^--^
 记录   原子  原子 残基链 残基     x坐标   y坐标   z坐标  占据  B因子        元素
 类型   序号  名称 名称  序号                                率  (温度因子)
```

逐字段拆解（这张表值得你记住）：

| 列内容 | 例值 | 含义 |
|---|---|---|
| 记录类型 | `ATOM` | `ATOM`=标准残基原子；`HETATM`=杂原子（水、配体、金属离子） |
| 原子序号 | `12` | 该原子在文件里的编号 |
| 原子名 | `CA` | **C-alpha（α碳）**，蛋白主链骨架的关键原子；`N`,`C`,`O` 也是主链 |
| 残基名 | `ASP` | 三字母氨基酸代码（天冬氨酸） |
| 链标识 | `A` | 链 ID（多聚体会有 A/B/C...） |
| 残基序号 | `2` | 残基在链中的位置 |
| x, y, z | `26.266 ...` | 原子的 3D 笛卡尔坐标，单位 **埃（Å, ångström）= 10⁻¹⁰ m** |
| 占据率 | `1.00` | **occupancy**：该原子处于此建模位置的占比。存在多构象（altloc, alternate location）时，各构象的 occupancy 之和为 1（如两个构象各占 0.5） |
| **B-factor** | `16.74` | 温度因子，反映原子的**热运动/位置不确定性**；越大越"晃"。**注意：AlphaFold 把 pLDDT 写在这一列！** |
| 元素 | `C` | 化学元素符号 |

其他常见行：`HEADER`（标题信息）、`SEQRES`（完整序列）、`HELIX`/`SHEET`（二级结构注释）、`TER`（链结束）、`MODEL`/`ENDMDL`（NMR 的多模型分隔）。

> 关键直觉：**一个 PDB 文件本质上就是一张大表——每个原子一行，核心信息是 `(原子名, 残基, 链, x, y, z)`。** 第 3 章里"把结构写成 $N \times 3$ 的坐标矩阵"，在这里就是把所有 `CA` 原子的 `(x,y,z)` 抽出来堆成矩阵。

### 10.2.3 mmCIF —— 现代标准格式

- 全称 **macromolecular Crystallographic Information File**（`.cif`）。
- PDB 格式有个硬伤：**字段宽度固定**。链 ID 只有 1 个字符、原子序号上限 99999（5 列）、残基序号上限 9999（4 列）——一个巨型复合物（如冷冻电镜测的核糖体）会**同时撞上这几个上限**而装不下。mmCIF 用"键-值/表格"结构（类似数据库），**没有这些宽度限制**。
- **PDB 官方已把 mmCIF 作为主存档格式**。新工具（包括 AlphaFold DB 下载）默认给 mmCIF。
- 此外，mmCIF 区分两套残基编号：`label_seq_id`（连续、从 1 起的规范编号）与 `auth_seq_id`（作者编号，可能跳号、含负数，对应文献里引用的残基号）。这个区别在做结构对比时很关键（见 10.6）。
- 你不需要手写解析器——Biopython / Biotite（10.5）两种格式都能读。

> **思考 10.B**：为什么 AlphaFold 把 pLDDT（一个"置信度"）塞进 B-factor 列，而不是新开一列？提示：想想"复用现有格式"和"现有可视化工具能直接按 B-factor 着色"这两个工程上的好处。

---

## 10.3 可视化：用眼睛理解结构

读坐标数字没人能想象出 3D 形状。两款主流免费工具：

| 工具 | 特点 | 适合 |
|---|---|---|
| **PyMOL** | 经典、命令行强大、出版级图片 | 做图、脚本化批量渲染 |
| **ChimeraX** | UCSF 出品、现代、对 cryo-EM/大复合物友好、内置 AlphaFold 支持 | 大结构、教学、交互探索 |

> 安装提示：PyMOL 有开源版（`conda install -c conda-forge pymol-open-source`）；ChimeraX 在官网免费下载。两者都跨平台。

### 10.3.1 PyMOL 入门命令

在 PyMOL 命令行（界面下方输入框）逐条敲：

```python
fetch 1ubq              # 直接从 PDB 联网下载并载入泛素
hide everything         # 先清空显示
show cartoon            # 显示卡通（cartoon）表示——α螺旋画成螺旋带、β折叠画成箭头
spectrum count, rainbow # 从 N 端到 C 端彩虹着色（看清链的走向）
show sticks, resn ASP   # 把所有天冬氨酸用棍状（sticks）显示
color red, ss H         # 把α螺旋(helix)染红； ss S 是β折叠
bg_color white          # 白背景
ray 1200, 900           # 高质量渲染（光线追踪）
png ~/ubq.png           # 存图
```

**核心心智模型**：PyMOL 的命令几乎都是 `动作 选择表达式`。选择表达式（selection）是一种小型查询语言：`resi 10-20`（残基号）、`chain A`、`resn LYS`（残基名）、`name CA`（原子名）、`ss H`（二级结构）可以用 `and`/`or`/`not` 组合。这和 10.5 里我们用 Python 过滤原子是同一种思路。

### 10.3.2 ChimeraX 入门命令

```python
open 1ubq                      # 载入
cartoon                        # 卡通表示
rainbow                        # 彩虹着色
color bfactor                  # 按 B-factor 着色（看 AlphaFold 结构时就是按 pLDDT！）
hide atoms; show cartoons      # 只留卡通
save ~/ubq_cx.png width 1200   # 存图
```

> 提示：ChimeraX 的命令语法随版本演进，按 B-factor 着色推荐用 `color bfactor`；若要套用 AlphaFold 的官方四段配色，用 `color bfactor palette alphafold`。具体写法以你当前版本的 User Guide 为准。

ChimeraX 还能一行命令直接拉 AlphaFold 预测：`alphafold fetch P0DTC2`，并**自动按 pLDDT 官方配色**——这是最省心的查看方式。

> **动手 10.C**：用 PyMOL 载入 `1ubq`，找到它著名的 5 股 β 折叠片 + 1 个 α 螺旋的"β-grasp"折叠（fold）。用 `color red, ss H` 和 `color yellow, ss S` 把二级结构高亮，截一张图。对照第 2 章里二级结构的定义，确认你看到的螺旋和折叠片。

---

## 10.4 跑预测：用 ColabFold 在浏览器里运行 AlphaFold2

第 7 章讲了 AlphaFold2 的原理；这里讲**怎么实际跑一次**。最低门槛的方式是 **ColabFold**——它把 AlphaFold2 包装好，跑在免费的 **Google Colab**（云端 GPU 笔记本）上，**不需要你装任何东西**。

### 10.4.1 ColabFold 是什么、为什么快

- ColabFold（Mirdita et al., 2022）的关键创新：用 **MMseqs2** 做远程 MSA 搜索，替代了 AlphaFold2 原版又慢又吃硬盘的 `jackhmmer + 大数据库`流程。**把"找同源序列"从几十分钟/几百 GB，压缩到几分钟、零本地存储。**
- MSA（multiple sequence alignment，多序列比对）是 AlphaFold2 的核心输入。它提供进化/共进化信号，Evoformer 据此构建一套丰富的**残基对表征**（pair representation）——残基间的"接触"只是这套表征里**一种可解读的信号**，而非全部。最终这套表征指导网络生成 3D 坐标（详见第 5 章、第 7 章）。

### 10.4.2 操作步骤（标准流程）

1. 打开 ColabFold 官方笔记本：在浏览器搜 **"ColabFold AlphaFold2.ipynb"**，进入 GitHub `sokrypton/ColabFold` 仓库，点 `AlphaFold2.ipynb` 的 "Open in Colab"。
2. 顶部菜单 **代码执行程序 → 更改运行时类型 → 硬件加速器选 GPU**（务必确认，否则极慢）。
3. 在 `query_sequence` 框里粘贴你的氨基酸序列（FASTA 里 `>` 之后那部分）。多链复合物用 `:` 分隔不同链。
4. 设几个关键选项（保持默认也能跑）：
   - `num_recycles`（循环精修次数）：默认 3。**recycling 的本质是把当前这一轮的预测结果重新输入模型、迭代精修**；前几轮提升明显，之后收益递减，调大更准但更慢。
   - `use_templates`：是否用 PDB 已知结构当模板。
   - `msa_mode`：默认 `mmseqs2` 自动搜 MSA；也可传你自己的 MSA。
5. **代码执行程序 → 全部运行**。等待。
6. 跑完自动打包下载一个 `.zip`，里面有预测的 PDB/mmCIF、各项图表、置信度数据。

### 10.4.3 看懂输出：pLDDT、PAE 与排序

ColabFold 默认**最多输出 5 个模型**。AlphaFold2 有 5 套训练好的参数（model weights），ColabFold 在 monomer 默认设置下确实对应这 5 套各出一个；但具体输出几个、是否对每套参数再做多次随机种子采样，受 `num_models`、`num_seeds`、以及 `model_type`（monomer / multimer，权重套数与命名不同）控制。输出按总置信度排序，文件名形如：

```
<job>_unrelaxed_rank_001_alphafold2_ptm_model_3_seed_000.pdb
```

注意 **rank 从 `001` 起、零填充三位**（不是 `rank_1`）。`rank_001` 是最好的那个。关键产物：

**(1) pLDDT —— 每个残基的局部置信度**

- 全称 **predicted Local Distance Difference Test**，范围 **0–100，越高越可信**。
- 它是**逐残基**的（每个残基一个值），写进输出 PDB 的 **B-factor 列**。
- 解读分级（AlphaFold 官方约定）：

| pLDDT | 含义 | 官方配色 |
|---|---|---|
| **> 90** | 很高置信，主链+侧链都可信 | 深蓝 |
| **70–90** | 较高置信，主链可信 | 浅蓝 |
| **50–70** | 低置信，谨慎对待 | 黄 |
| **< 50** | 很可能无序/不可靠，常是柔性 loop 或内在无序区（IDR） | 橙 |

> 重要：**低 pLDDT 不一定是"预测错了"**，很可能是该区域**本身就没有固定结构**（内在无序区，intrinsically disordered region）。这是信息，不是 bug。

**按 pLDDT 着色（注意配色语义！）**：AlphaFold 官方配色是**离散四段：橙(<50) – 黄(50–70) – 浅蓝(70–90) – 深蓝(>90)**，即"蓝=高置信、橙=低置信"。如果你在 PyMOL 里用一个连续发散配色作临时查看，务必记住它**不带 AF 官方语义**——尤其别让"红色"被读成"高"：

```python
# 临时连续配色（非 AF 官方语义；这里仅按 B-factor 高低渐变，颜色含义需自己声明）
spectrum b, blue_white_red, minimum=50, maximum=90
```

要得到**官方语义**的配色，最稳妥的是用 ChimeraX：`color bfactor palette alphafold`，或直接 `alphafold fetch <accession>`（自动配色）。

**(2) PAE —— 残基对之间的相对位置置信度**

- 全称 **Predicted Aligned Error**，是一个 $L \times L$ 的矩阵（$L$=序列长度）。
- $\text{PAE}[i,j]$ 的含义：**当结构以残基 $i$ 为参考对齐时，残基 $j$ 的预期位置误差（埃）**。行索引 $i$ 是"对齐参考残基"，列索引 $j$ 是"被看误差的残基"。**越小越好**。
- **PAE 矩阵一般不对称**：$\text{PAE}[i,j] \neq \text{PAE}[j,i]$。这是它与距离矩阵（对称）的关键区别，读图时务必记住——"以 A 为基准看 B 的误差"和"以 B 为基准看 A 的误差"是两件事。
- **用途**：判断**结构域之间的相对取向**是否可信。单个结构域内部 pLDDT 高、但两个域之间 PAE 大 → 说明"每个域各自靠谱，但它们的相对摆放不确定"。这对多结构域蛋白和复合物尤其关键（第 7 章、第 8 章）。

> **动手 10.D**：用 ColabFold 预测一条**短肽或小蛋白**（建议 < 100 残基，跑得快；可取 `1UBQ` 的序列）。下载结果后：(1) 在 PyMOL 里按 B-factor（pLDDT）着色，记得自己声明配色语义；(2) 打开 PAE 图，找出是否有"块状"结构（块对应结构域）。把序列、pLDDT 平均值、你的观察记下来。

### 10.4.4 没有 MSA 怎么办？ESMFold 与单序列预测

AlphaFold2 高度依赖 MSA。如果你的蛋白是**孤儿蛋白**（orphan protein，找不到同源序列）或**你刚设计的全新序列**，MSA 会很浅甚至为空，AF2 置信度随之偏低——**这是预期行为，不是 bug**（第 8 章讨论了"单序列预测"的方向）。

这时可以试 **ESMFold**（Lin et al., 2023）：它基于蛋白质语言模型（protein language model，ESM-2），**只需单条序列、不做 MSA 搜索，几秒到几十秒出结果**。上手途径：

- 官方 **ESM Atlas / ESMFold API**：直接 POST 一条序列即可拿到 PDB（适合脚本化批量）。
- ESMFold 的 Colab 笔记本（在 `facebookresearch/esm` 或社区仓库里可找到）。

> 注意取舍：在**有深 MSA 的蛋白**上，ESMFold 整体精度**通常逊于 AF2**；但在 MSA 极浅/孤儿蛋白/超大规模快速筛选场景，它的"无需 MSA + 速度"优势非常实用。把它当作 AF2 的补充工具，而非替代。

### 10.4.5 算力与耗时心理预期

| 阶段 | 耗时（Colab 免费 T4 GPU 估计） | 主要瓶颈 |
|---|---|---|
| MSA 搜索（MMseqs2 远程） | 几十秒 ~ 几分钟 | 网络 + 服务器排队 |
| 结构预测（单序列 <300 aa） | 几分钟 | GPU 显存与算力 |
| 长序列（>1000 aa）/ 复合物 | 十几分钟 ~ 更久，**可能爆显存** | GPU **显存**（最常见的失败原因） |

- Colab 免费版会**断连**（闲置或超时），跑长任务有风险；Colab Pro 给更好 GPU（如 A100）和更长时限。
- 真正大规模（成百上千条序列）应在**本地/集群装 LocalColabFold 或 AlphaFold**，需要好 GPU（≥16 GB 显存更稳）和几十 GB ~ TB 级数据库（若用原版 jackhmmer 流程）。
- **省钱铁律**：先去 **AlphaFold DB（10.1.3）查一下**——天然蛋白多半已经预测好了，根本不用自己跑。

---

## 10.5 编程：用 Python 解析结构、计算几何量

可视化之外，真正的研究要**量化**。Python 生态里两大结构解析库：

| 库 | 风格 | 特点 |
|---|---|---|
| **Biopython** | 面向对象（Structure→Model→Chain→Residue→Atom 层级） | 经典、教程多、生信全家桶的一部分 |
| **Biotite** | 面向数组（原子数组 + numpy，向量化） | 现代、快、和 numpy 无缝、适合大规模计算 |

入门两者皆可。下面先用 Biopython 演示"读结构"，再用 numpy 演示两个最核心的计算：**距离矩阵/接触图**和**二面角**（概念见第 3 章）。

### 10.5.1 读取结构并抽取 Cα 坐标矩阵

```python
import numpy as np
from Bio.PDB import PDBParser, MMCIFParser

def load_ca_coords(path, chain_id="A"):
    """读 PDB/mmCIF，返回指定链所有标准残基的 CA 坐标，形状 (L, 3)。"""
    parser = MMCIFParser(QUIET=True) if path.endswith(".cif") else PDBParser(QUIET=True)
    structure = parser.get_structure("x", path)
    model = structure[0]                     # 取第一个 model（NMR 可能有多个）
    chain = model[chain_id]
    coords = []
    for residue in chain:
        # 只要标准氨基酸残基（过滤水、配体等 HETATM）
        if residue.id[0] == " " and "CA" in residue:
            coords.append(residue["CA"].get_coord())
    return np.array(coords)                   # (L, 3) 的 numpy 数组

ca = load_ca_coords("1ubq.pdb")
print(ca.shape)        # 例如 (76, 3)
```

到这里，第 3 章那句"结构就是一个 $N\times3$ 矩阵"已经具体化了：`ca` 就是它。

### 10.5.2 距离矩阵与接触图（contact map）

**距离矩阵**（distance matrix）$D$ 定义为 $D_{ij} = \lVert \mathbf{r}_i - \mathbf{r}_j \rVert_2$，即第 $i$ 和第 $j$ 个原子的欧氏距离。**接触图**（contact map）是把它二值化：距离小于阈值就算"接触"。

> ⚠️ **接触的定义不止一种，报告里必须写清楚**。文献中最经典的"8 Å 接触"（CASP 评测、DCA/RaptorX 等共进化接触预测）用的是 **Cβ–Cβ < 8 Å**（甘氨酸没有 Cβ，用 Cα 代替）。而 **Cα–Cα** 的常用阈值是 **6.5–8 Å**，因定义而异。下面的演示用 **Cα–Cα < 8 Å**（最简单，便于和上面的 `ca` 矩阵直接衔接），但你要清楚这只是诸多约定之一。

```python
def distance_matrix(coords):
    # 广播技巧：coords[:,None,:] 形状 (L,1,3)，coords[None,:,:] 形状 (1,L,3)
    diff = coords[:, None, :] - coords[None, :, :]   # (L, L, 3)
    return np.sqrt((diff ** 2).sum(-1))              # (L, L)

D = distance_matrix(ca)
contact = (D < 8.0).astype(int)        # 这里用 CA-CA < 8 Å（演示用，非唯一标准）

import matplotlib.pyplot as plt
plt.imshow(contact, cmap="Greys", origin="lower")
plt.xlabel("residue i"); plt.ylabel("residue j")
plt.title("Contact map (CA-CA < 8 A)")
plt.show()
```

**怎么读接触图**（三种典型花纹，对应不同二级结构）：

```
   反平行 β            平行 β              α 螺旋
   ┆                  ╲                  主对角线附近
   ┆ 垂直于主对角线     ╲ 平行于主对角线     一条"加宽的带"
 ──╋──> 反对角线方向     ╲ 的次对角线        (i 与 i±3/4 接触)
   ┆ 的细线             ╲ 细线
```

- **α 螺旋**：紧贴主对角线的**加宽带**——每个 $i$ 都与 $i\pm3$、$i\pm4$ 接触。
- **平行 β 折叠**：一条**平行于主对角线**的细线（次对角线）。
- **反平行 β 折叠**：一条**垂直于主对角线、沿反对角线方向**的细线。
- 远离对角线的孤立点 ≈ 长程接触（折叠把序列上远的部分拉近了）。

这正是 AlphaFold 之前一代方法（第 5 章）努力预测的对象（它们多用 Cβ 定义）。

### 10.5.3 主链二面角 φ/ψ 与 Ramachandran 图

第 3 章讲过，固定键长键角后，蛋白主链构象几乎完全由每个残基的两个**二面角**（dihedral / torsion angle）$\phi$（phi）和 $\psi$（psi）决定。

- $\phi$：绕 $N\text{–}C_\alpha$ 键，由原子 $(C_{i-1}, N_i, C\alpha_i, C_i)$ 定义。
- $\psi$：绕 $C_\alpha\text{–}C$ 键，由原子 $(N_i, C\alpha_i, C_i, N_{i+1})$ 定义。

**任意**四个点 $\mathbf{p}_1,\mathbf{p}_2,\mathbf{p}_3,\mathbf{p}_4$ 的二面角，用向量叉积求两个平面的法向量夹角（这是一个**通用函数**，记结果为 $\theta$，而非特指 φ）：

$$
\mathbf{b}_1=\mathbf{p}_2-\mathbf{p}_1,\quad \mathbf{b}_2=\mathbf{p}_3-\mathbf{p}_2,\quad \mathbf{b}_3=\mathbf{p}_4-\mathbf{p}_3
$$
$$
\mathbf{n}_1=\mathbf{b}_1\times\mathbf{b}_2,\quad \mathbf{n}_2=\mathbf{b}_2\times\mathbf{b}_3
$$
$$
\theta=\operatorname{atan2}\big((\mathbf{n}_1\times\mathbf{n}_2)\cdot\hat{\mathbf{b}}_2,\ \mathbf{n}_1\cdot\mathbf{n}_2\big)
$$

$\phi$、$\psi$ 都是把**不同的四个主链原子**代入这同一个 `dihedral(p1,p2,p3,p4)` 函数得到的。（用 `atan2` 而非 `arccos`，是为了得到带符号的 −180°~180° 全范围角度，数值也更稳。）

```python
def dihedral(p1, p2, p3, p4):
    """任意四点的二面角（通用函数），返回 -180~180 度。"""
    b1, b2, b3 = p2 - p1, p3 - p2, p4 - p3
    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)
    m1 = np.cross(n1, b2 / np.linalg.norm(b2))
    x = np.dot(n1, n2)
    y = np.dot(m1, n2)
    return np.degrees(np.arctan2(y, x))     # 角度，-180 ~ 180
```

> 实践中**不必自己实现**：Biopython 的 `Bio.PDB.internal_coords` 或 `calc_dihedral`、Biotite 的 `struc.dihedral_backbone` 都能直接给出全链 φ/ψ。上面的实现是为了让你理解"角度从坐标怎么来的"。

把所有残基的 $(\phi, \psi)$ 画成散点图，就是 **Ramachandran 图**（拉氏图，第 3、4 章）。由于侧链与主链的空间位阻（steric clash），大多数 $(\phi,\psi)$ 组合在物理上被禁止，允许区集中在几块。**以标准约定（φ 为横轴、ψ 为纵轴，范围 −180°~180°）**：

| 允许区 | 大致位置 | 典型 $(\phi,\psi)$ |
|---|---|---|
| **右手 α 螺旋**（right-handed α-helix） | **左下象限** | $(-60°, -45°)$ |
| **β 折叠 / 伸展构象**（β-sheet） | **左上象限** | $(-120°, +130°)$ |
| **左手 α 螺旋**（left-handed α-helix，少见） | **右上象限一小块** | $(+60°, +45°)$ |

**拉氏图是检查结构合理性的"体检表"**——落在禁区的残基往往意味着结构错误或特殊情况（如甘氨酸 glycine 没有侧链、更自由，脯氨酸 proline 因环状侧链 φ 受限）。

```python
phis, psis = [], []   # 用上面任一方法填充每个残基的 phi/psi（首尾残基缺一个，跳过）
plt.scatter(phis, psis, s=8)
plt.xlim(-180,180); plt.ylim(-180,180)
plt.axhline(0,c='gray',lw=.5); plt.axvline(0,c='gray',lw=.5)
plt.xlabel(r"$\phi$"); plt.ylabel(r"$\psi$"); plt.title("Ramachandran plot")
plt.show()
```

> **更专业的结构质检**：拉氏图只是肉眼层面的"体检"。要系统评估一个结构（无论实验还是预测）的化学合理性——键长键角偏差、原子碰撞（clash）、拉氏离群点、侧链旋转异构体——业界标准工具是 **MolProbity**（molprobity.biochem.duke.edu）；提交到 PDB 的结构还会自动生成 **wwPDB Validation Report**。把这一步纳入你的分析习惯。

---

## 10.6 你的"第一个项目"：预测 → 可视化 → 量化对比

把前面所有零件拼成一个完整闭环。这是一个能写进学习笔记、甚至作为简历作品的小项目。

### 10.6.1 项目流程图

```
 选一个小蛋白（已有PDB实验结构, <100残基）
            │
            ▼
 从 PDB 拿"真值"结构  ───────┐
            │                │
            ▼                │
 取其序列 → ColabFold 预测     │
            │                │
            ▼                ▼
   按序列比对找对应残基对 ◀────┘
            │
            ▼
   PyMOL 可视化 ─── 叠合(align) ──► 算 RMSD / TM-score
            │
            ▼
   Biopython+numpy：
   画 Ramachandran 图 + 接触图
            │
            ▼
   写一段对比分析（哪里准、哪里不准、为什么）
```

**选蛋白建议**：`1CRN`（crambin，46 残基，X 射线 **1.5 Å**，干净的经典小蛋白；若想要"超高分辨率 crambin"的极端例子，那是另一条目 **3NIR，0.48 Å**，不要和 1CRN 混淆）、`1UBQ`（泛素，76 残基）、或任一你感兴趣的小蛋白。**避免**首个项目就挑大复合物或含金属/配体的酶——变量太多。

### 10.6.2 第一个真实坑：残基对应（别跳过）

把"预测结构"和"实验结构"逐残基比对，**前提是两条链的残基要一一对应**。但实际上它们常常对不齐：

- **缺失残基**：实验结构里看不清的柔性残基直接缺席，编号会**跳号**。
- **编号体系**：PDB/mmCIF 的 `auth_seq_id`（作者编号）可能跳号、含负数，与你的连续序列编号不同。
- **额外片段**：实验构建体常带 **His-tag**（纯化标签）、信号肽被切除等，导致两端长度不一。

**最省心的做法**：直接把两个 PDB 丢给 **TM-align**（见 10.6.3），它**内部自动做最优序列/结构对齐**，你不必手动配对。若要在 Python 里自己配对，最小做法是先做序列比对：

```python
from Bio import Align
aligner = Align.PairwiseAligner()
aligner.mode = "global"
aln = aligner.align(seq_exp, seq_pred)[0]   # seq_* 是从两个结构各自抽出的一字母序列
# 从 aln 里读出对齐列，只保留两边都非 gap 的位置，得到一一对应的残基索引对
```

记住：**这一步是初学者第一个项目最容易翻车的地方**。如果 RMSD 大得离谱，先怀疑"是不是配错了对"，而不是"预测真的很烂"。

### 10.6.3 两个核心指标：RMSD 与 TM-score

预测结构和实验结构"像不像"，需要量化。先要**叠合**（superpose）：把两个结构旋转平移到最佳重合。这是一个有解析解的最优化问题——**Kabsch 算法**，**经 SVD（奇异值分解）求出最优旋转矩阵**。

**RMSD**（Root-Mean-Square Deviation，均方根偏差），叠合后对应原子的均方根距离：

$$
\text{RMSD} = \sqrt{\frac{1}{N}\sum_{i=1}^{N}\lVert \mathbf{r}_i^{\text{pred}} - \mathbf{r}_i^{\text{exp}} \rVert^2}
$$

- 单位 Å，**越小越好**。整体折叠对的小蛋白通常 RMSD 1–3 Å。
- **缺点**：对**局部大偏差极其敏感**——一条乱飘的尾巴就能把整体 RMSD 拉爆，即使核心折叠完美。
- **关于长度**：RMSD 公式本身**没有长度归一化**（它是对 $N$ 个偏差平方取平均再开方，并不显式随 $N$ 单调增大）。但实践中**越长的蛋白越可能含有更多易偏区段**，从而 RMSD 偏大；而且**同一个 RMSD 值在不同长度的蛋白之间不可直接比较**。

**TM-score**（Template Modeling score），为弥补 RMSD 的上述缺点而设计：

$$
\text{TM-score} = \frac{1}{L_{\text{target}}}\sum_{i=1}^{L_{\text{aligned}}} \frac{1}{1 + (d_i/d_0)^2}
$$

- **分母** $L_{\text{target}}$ 是**目标蛋白的总长度**；**求和只对比对上的 $L_{\text{aligned}}$ 对残基**（$d_i$ 是第 $i$ 对残基叠合后的距离）。
- **关键的归一化尺度** $d_0$ 随目标长度变化：
$$
d_0(L_{\text{target}}) = 1.24\,\sqrt[3]{\,L_{\text{target}}-15\,}\;-\;1.8
$$
正是这个**随 $L$ 增长的 $d_0$** 抵消了打分对长度的依赖——这才是 TM-score "长度无关"的真正来源，不是一句空话。
- 范围 **0–1，越大越好**。经验阈值：**TM-score > 0.5 通常意味着两者属于同一种折叠（same fold）**；> 0.9 几乎一致。
- **它对局部噪声更鲁棒**：近的残基贡献大，远的（如乱尾）经 $1/(1+(d_i/d_0)^2)$ 压低——比 RMSD 更能反映"整体折叠对不对"。

> 工具：算 TM-score 用官方 **TM-align** 程序（输入两个 PDB，自动叠合并输出 TM-score 与 RMSD），或 Python 包 `tmtools`。RMSD 可用 PyMOL 的 `align`/`super` 命令，或 Biopython 的 `Superimposer`（其内部即用 Kabsch/SVD）。

### 10.6.4 最小可复制代码骨架

```python
"""第一个项目骨架：对比预测结构 vs 实验结构。
依赖: biopython, numpy, matplotlib, (可选) tmtools
注意: 本骨架为简洁起见假设两链等长同序；真实项目务必先做 10.6.2 的残基配对！
"""
import numpy as np
from Bio.PDB import PDBParser, Superimposer

parser = PDBParser(QUIET=True)

def get_ca_atoms(path, chain_id="A"):
    """返回 (CA Atom 对象列表, CA 坐标 ndarray)。"""
    s = parser.get_structure("s", path)
    atoms = [res["CA"] for res in s[0][chain_id]
             if res.id[0] == " " and "CA" in res]
    coords = np.array([a.get_coord() for a in atoms])
    return atoms, coords

# 1) 载入实验结构与预测结构
#    ColabFold 输出名形如 *_rank_001_alphafold2_*.pdb
exp_atoms,  exp_xyz  = get_ca_atoms("1ubq_experimental.pdb")
pred_atoms, pred_xyz = get_ca_atoms("1ubq_predicted_rank_001.pdb")

# 2) 长度对齐检查（真实项目里必须按 10.6.2 做序列比对找对应残基对）
assert len(exp_atoms) == len(pred_atoms), "残基数不一致，需先做序列比对（见 10.6.2）"

# 3) 用 Kabsch/SVD 叠合并算 RMSD（Biopython Superimposer 内部即此）
sup = Superimposer()
sup.set_atoms(exp_atoms, pred_atoms)     # (参考, 待叠)
print(f"RMSD (CA) = {sup.rms:.3f} A")
sup.apply(pred_atoms)                     # 把预测结构原子对象就地旋平移到叠合位置

# 4) TM-score（可选，用 tmtools；它内部自己叠合，与上面的 apply 无关）
try:
    from tmtools import tm_align
    seq = "".join("A" for _ in exp_atoms)        # 占位序列；真实应由三字母转一字母
    res = tm_align(pred_xyz, exp_xyz, seq, seq)
    print(f"TM-score = {res.tm_norm_chain2:.3f}")
except ImportError:
    print("装 tmtools 或用 TM-align 程序来算 TM-score")

# 5) 距离矩阵差异：哪里偏得最多？
#    注意：距离矩阵是刚体变换(旋转+平移)的不变量，所以这里用不用 apply 都一样，
#    可直接用最初抽出的 exp_xyz / pred_xyz。
#    （但若要画"叠合后的坐标图"，则必须用 apply 之后的坐标。）
def dmat(c):
    d = c[:,None,:] - c[None,:,:]
    return np.sqrt((d**2).sum(-1))
delta = np.abs(dmat(exp_xyz) - dmat(pred_xyz))
print("距离矩阵最大偏差残基对:", np.unravel_index(delta.argmax(), delta.shape))
# → 接着可以 plt.imshow(delta) 看偏差热图，定位预测最不准的区域
```

> **诚实提醒**：上面"假设等长同序"是为了骨架简洁。**真实情况**两条链常因实验缺失残基而长度不等、编号跳号，必须先按 10.6.2 做**序列比对**找到对应残基对再叠合；或直接交给 TM-align（它内部自动做最优对齐）。

### 10.6.5 项目产出清单（做完你应该有）

- [ ] 一张 PyMOL 叠合图（实验=灰，预测=按 pLDDT 着色，并标注配色语义），肉眼看哪里贴合、哪里偏。
- [ ] 一个 RMSD 数值和一个 TM-score 数值。
- [ ] 预测结构的 Ramachandran 图（检查是否大部分落在允许区）。
- [ ] 接触图 / 距离矩阵差异热图（定位预测最不准的残基区段）。
- [ ] 一段 200 字分析：**结合 pLDDT 解释偏差**——通常你会发现"低 pLDDT 区"恰好就是"偏差大区"，且常是柔性 loop 或端部。这正是把第 7 章理论和第 10 章实践连起来的关键洞察。

---

## 10.7 环境与算力速查

| 需求 | 推荐方案 | 备注 |
|---|---|---|
| 只想跑几条预测（要 MSA） | **ColabFold on Colab** | 零安装；免费 GPU 但会断连 |
| 无 MSA / 孤儿蛋白 / 极速筛选 | **ESMFold（API 或 Colab）** | 单序列、秒级；深 MSA 蛋白上整体逊于 AF2 |
| 频繁跑、要稳定 | **LocalColabFold**（本地装） | 需 NVIDIA GPU（≥12–16 GB 显存更稳） |
| 只看天然蛋白结构 | **AlphaFold DB 直接下载** | 不耗算力，首选 |
| 解析/分析结构 | **CPU 即可**，`conda` 装 biopython/biotite | 不需要 GPU |
| 可视化 | PyMOL / ChimeraX，本地装 | 普通笔记本足够 |
| 结构质检 | **MolProbity / wwPDB validation** | 评估化学合理性 |

**通用环境配置（推荐用 conda/mamba 隔离）**：

```bash
conda create -n fold python=3.10
conda activate fold
conda install -c conda-forge biopython biotite numpy matplotlib pymol-open-source
pip install tmtools          # 算 TM-score
# ColabFold / ESMFold 走浏览器或 API，无需本地安装
```

**几个常见坑（提前知道少踩）**：
- **显存爆炸（CUDA out of memory）**：跑长序列/复合物时最常见。对策：减小序列、关模板、用更大显存的 GPU、或分段预测。
- **MSA 搜不到同源序列**：孤儿蛋白或你设计的全新序列，MSA 很浅 → AF2 预测置信度低，这是**预期行为**而非 bug。此时可改用 **ESMFold（10.4.4）单序列预测**（第 8 章讨论了这一新方向）。
- **链/残基对不齐**：实验结构常缺失柔性残基、编号跳号（`auth_seq_id` vs `label_seq_id`），还可能带 His-tag、切除信号肽；对比前务必按 10.6.2 核对编号与序列。

---

## 本章要点回顾

1. **数据**：PDB（实验真值）、UniProt（序列+注释）、AlphaFold DB（预测结构，先查它别急着自己跑；CC-BY 4.0，引用 Jumper 2021 / Varadi 2022）、Pfam/InterPro（家族/进化信号，Pfam 现经 InterPro 访问）。
2. **格式**：FASTA（序列）、PDB/mmCIF（结构）。一个结构文件本质是"每原子一行的大表"，核心是 `(原子名, 残基, 链, x, y, z)`；**AlphaFold 把 pLDDT 写在 B-factor 列**；occupancy 是"原子处于此建模位置的占比"。
3. **可视化**：PyMOL/ChimeraX，命令模式都是"动作 + 选择表达式"；按 pLDDT（B-factor）着色看可信度，但**注意官方配色是橙(低)→蓝(高)**，别让连续发散配色反向误读。
4. **跑预测**：ColabFold = 浏览器里的 AlphaFold2，靠 MMseqs2 快速搜 MSA；输出文件名 `rank_001` 起；看 **pLDDT（逐残基置信）** 与 **PAE（域间相对位置置信，矩阵不对称）**。无 MSA 时用 **ESMFold** 单序列预测。
5. **编程**：Biopython/Biotite 读结构 → numpy 抽 $N\times3$ 坐标 → 算**距离矩阵/接触图**（注明 Cα/Cβ 与阈值）与 **φ/ψ 二面角**（用通用 `dihedral` 函数；拉氏图：右手 α 在左下、β 在左上）。
6. **第一个项目**：小蛋白 → ColabFold 预测 → **先做残基配对** → PyMOL 叠合（Kabsch/SVD）→ **RMSD / TM-score（含 $d_0$ 公式）** 量化 → 拉氏图 + 接触图分析，并用 pLDDT 解释偏差。
7. **诚实边界**：低 pLDDT/高偏差常意味着"本就无序"而非"预测错"；实验结构也非完美真值（是电子密度的时空平均拟合出的代表构象）；预测结构不替代实验验证。

---

## 深入阅读

**论文 / 方法**
- Mirdita M., et al. *"ColabFold: making protein structure prediction accessible to all."* **Nature Methods, 2022.** —— ColabFold 原文，理解 MMseqs2 加速 MSA 的关键。
- Jumper J., et al. *"Highly accurate protein structure prediction with AlphaFold."* **Nature, 2021.** —— AF2 原文（pLDDT、PAE 的定义在此，配合第 7 章读）。
- Varadi M., et al. *"AlphaFold Protein Structure Database."* **Nucleic Acids Research, 2022.** —— AlphaFold DB 介绍（引用它以合规使用 AF DB 数据）。
- Lin Z., et al. *"Evolutionary-scale prediction of atomic-level protein structure with a language model (ESMFold)."* **Science, 2023.** —— 单序列、基于蛋白语言模型的结构预测。
- Zhang Y. & Skolnick J. *"Scoring function for automated assessment of protein structure template quality (TM-score)."* **Proteins, 2004.** —— TM-score 与 $d_0$、"0.5 阈值"的由来。
- Kabsch W. *"A solution for the best rotation to relate two sets of vectors."* **Acta Cryst., 1976.** —— RMSD 叠合背后的最优旋转算法（经 SVD 求解）。
- Williams C. J., et al. *"MolProbity: More and better reference data for improved all-atom structure validation."* **Protein Science, 2018.** —— 结构质检的业界标准工具。

**工具文档 / 教程**
- **Biopython Structural Bioinformatics FAQ / Tutorial**（`biopython.org`）—— `Bio.PDB` 官方教程。
- **Biotite documentation**（`www.biotite-python.org`）—— 数组式结构处理，向量化计算范例丰富。
- **PyMOL Wiki**（`pymolwiki.org`）—— 命令与选择语法大全。
- **UCSF ChimeraX User Guide / Tutorials**（`www.cgl.ucsf.edu/chimerax`）—— 含 AlphaFold 可视化教程（以你当前版本语法为准）。
- **RCSB PDB "101" 教育页**（`pdb101.rcsb.org`）—— 极佳的结构生物学入门图文。
- **MolProbity**（`molprobity.biochem.duke.edu`）—— 在线结构质检。

**数据库门户**
- RCSB PDB `rcsb.org` ｜ UniProt `uniprot.org` ｜ AlphaFold DB `alphafold.ebi.ac.uk` ｜ InterPro `ebi.ac.uk/interpro`

**承上启下**：本章的实操技能将贯穿后续学习——用它去复现第 7 章 AlphaFold2 的结果、检验第 8 章新方法（如 ESMFold、复合物预测）的输出、以及验证第 9 章你自己设计的蛋白序列能否折叠成目标结构。**真正的理解，从你亲手跑通第一个项目开始。**
