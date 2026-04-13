# 指标补充说明（Novelty 与 FCD）

## 1. 文档目的

本文档对主文稿中使用的部分分布与新颖性指标进行补充说明，重点回答两个问题：

1. 新颖度（novelty）在本文中是如何定义和计算的；
2. Fréchet ChemNet Distance（FCD）在当前 anchored linker 任务中如何使用、如何解释，以及本实验中不同生成版本的 FCD 结果如何。

本文档对应的原始分析输出位于：

- [/Users/lintianjian/diffusion/outputs/fingerprint_analysis/summary_metrics.csv](/Users/lintianjian/diffusion/outputs/fingerprint_analysis/summary_metrics.csv)
- [/Users/lintianjian/diffusion/outputs/fingerprint_analysis/nn_similarity_summary.csv](/Users/lintianjian/diffusion/outputs/fingerprint_analysis/nn_similarity_summary.csv)
- [/Users/lintianjian/diffusion/outputs/fingerprint_analysis/fcd_summary.csv](/Users/lintianjian/diffusion/outputs/fingerprint_analysis/fcd_summary.csv)

## 2. Novelty 的定义

本文中的 novelty 采用基于 canonical SMILES 的集合比较定义，而不是基于指纹距离阈值的近似判定。具体步骤如下：

1. 对每个数据集中的分子进行 RDKit 解析与 sanitize；
2. 仅保留可成功解析的有效分子；
3. 将有效分子转换为 canonical SMILES；
4. 对生成集按 canonical SMILES 去重，得到 unique valid molecules；
5. 以训练集（train）中的 canonical SMILES 集合为参考，将生成集中不属于训练集 canonical 集合的样本记为 novel；
6. 计算：

   `novelty = novel_unique_count / unique_valid_count`

因此，novelty 衡量的是“生成集中有多少唯一有效结构没有在训练集里原样出现过”，而不是“生成结构距离训练集有多远”。这也是为什么 novelty 需要与 nearest-neighbor similarity、internal diversity 等指标配合解读。

以当前实验为例：

- `gen_16d` 的 novelty 为 `0.953`，表示其唯一有效生成 linker 中约 95.3% 未在训练集 canonical anchored linker 集合中原样出现；
- `gen_32d` 的 novelty 为 `1.000`，表示其唯一有效生成 linker 没有与训练集中任一 canonical anchored linker 完全重合；
- `raw_linker` 的 novelty 为 `0.474`，这是因为原始 anchored linker 库与 weak-anchor 训练集本身存在较高重合。

## 3. FCD 的定义与当前任务中的使用方式

### 3.1 FCD 的基本含义

Fréchet ChemNet Distance（FCD）是一种分布级指标，用于比较两个分子集合在 ChemNet 特征空间中的统计距离。其思想与图像生成中的 Fréchet Inception Distance（FID）类似：先将分子映射到预训练网络的表征空间，再比较两个集合的均值和协方差，从而衡量两个集合在“高层语义特征分布”上的接近程度。

在本文中，FCD 通过 `fcd_torch` 实现计算，默认以训练集（train）作为参考分布。数值越小，表示待比较集合与训练集分布越接近；数值越大，表示分布差异越明显。

### 3.2 本文中的实现口径

本文额外计算了两种 FCD：

1. `fcd_vs_train_all`
   - 直接使用每个数据集中的全部有效分子；
   - 保留重复样本，因此更敏感于样本频率分布。

2. `fcd_vs_train_unique`
   - 对每个数据集的有效 canonical SMILES 去重后再计算；
   - 更强调“结构集合差异”，而弱化重复样本频次影响。

对应脚本为：

- [/Users/lintianjian/diffusion/compute_fcd_metrics.py](/Users/lintianjian/diffusion/compute_fcd_metrics.py)

### 3.3 anchored linker 场景下的解释边界

需要注意的是，标准 FCD 更常用于普通小分子集合比较，而本文比较对象是带有 dummy anchor 的 anchored linker。尽管 `fcd_torch` 可以正常处理这类 SMILES，并给出稳定数值，但该结果仍应被视为一种“补充性的分布距离指标”，而不是与标准小分子 benchmark 完全等价的绝对分数。

因此，在 anchored linker 场景下，FCD 更适合回答：

- 哪个生成版本在整体分布上更接近训练 anchored linker 集合；
- 不同控制变量（如 16d / 32d / pad suffix）之间，谁偏离训练分布更明显。

而不宜单独将其解释为“化学质量优劣”的唯一标准。

## 4. 当前实验的 FCD 结果

当前实验中，基于训练集作为参考分布得到的 FCD 结果如下表所示。

| 数据集 | valid(all) | valid(unique) | FCD vs train (all) | FCD vs train (unique) |
| --- | ---: | ---: | ---: | ---: |
| train | 5020 | 2263 | 0.000 | 0.000 |
| raw_linker | 2749 | 2749 | 0.599 | 0.293 |
| gen_16d | 128 | 128 | 3.150 | 3.258 |
| gen_pad_suffix_ce015 | 128 | 128 | 5.993 | 5.616 |
| gen_pad_suffix_ce005 | 128 | 124 | 4.226 | 4.183 |
| gen_32d | 128 | 128 | 4.298 | 4.368 |
| Link-INVENT | 123 | 115 | 25.747 | 25.410 |

从结果上看，当前各版本相对训练集的 FCD 排序与前面 Morgan fingerprint 最近邻相似度分析基本一致。若以 `all-valid` 口径为主，则在生成模型中：

- `gen_16d` 的 FCD 最低（`3.150`），说明其整体分布最接近训练集；
- `gen_pad_suffix_ce005` 次之（`4.226`），说明温和停止约束虽然改变了部分性质分布，但仍保持了相对可接受的整体接近度；
- `gen_32d` 与 `gen_pad_suffix_ce015` 的 FCD 更高（分别为 `4.298` 与 `5.993`），与前文“更长、更重或更偏离训练分布”的观察一致；
- `Link-INVENT` 的 FCD 显著更高（`25.747`），这说明其虽然在 QED、SA 上更接近传统小分子药化风格，但在当前 anchored linker 任务分布上与训练集存在明显差异。

这一点也可以与 `raw_linker` 对照理解：原始 anchored linker 库相对训练集的 FCD 仅为 `0.599`，说明 weak-anchor 训练集与原始 anchored linker 库在整体结构空间上仍高度相关；而生成模型和外部 baseline 的 FCD 则主要反映了各自引入的生成偏移。

## 5. 初步解释框架

FCD 与当前已有指标可以形成如下互补关系：

- novelty 高，但 FCD 也高：
  - 说明模型生成了许多训练集中未出现过的结构，但这些结构在整体分布上也明显偏离训练集；
- novelty 高，但 FCD 中等：
  - 说明模型能够在保持一定分布贴合度的同时探索新结构；
- nearest-neighbor similarity 高，但 FCD 仍不低：
  - 说明局部最近邻接近训练样本，但整体集合分布可能仍存在系统性偏移；
- FCD 很低但 uniqueness 下降：
  - 说明模型可能更接近训练分布，但也可能伴随一定程度的模式收缩或重复。

因此，在本文的 anchored linker 任务中，更合理的做法不是单独依赖某一个指标，而是联合观察：

- novelty
- mean nearest-neighbor Tanimoto
- internal diversity
- FCD
- QED / SA / MW / rotatable bonds 等性质分布

## 6. 图表输出

FCD 的计算输出位于：

- [/Users/lintianjian/diffusion/outputs/fingerprint_analysis/fcd_summary.csv](/Users/lintianjian/diffusion/outputs/fingerprint_analysis/fcd_summary.csv)
- [/Users/lintianjian/diffusion/outputs/fingerprint_analysis/fcd_summary.json](/Users/lintianjian/diffusion/outputs/fingerprint_analysis/fcd_summary.json)
- [/Users/lintianjian/diffusion/outputs/fingerprint_analysis/fcd_vs_raw_linker.csv](/Users/lintianjian/diffusion/outputs/fingerprint_analysis/fcd_vs_raw_linker.csv)
- [/Users/lintianjian/diffusion/outputs/fingerprint_analysis/fcd_vs_raw_linker.json](/Users/lintianjian/diffusion/outputs/fingerprint_analysis/fcd_vs_raw_linker.json)

如果图已经生成，则可进一步引用：

- [/Users/lintianjian/diffusion/outputs/fingerprint_analysis/figure7_fcd_vs_train_all.png](/Users/lintianjian/diffusion/outputs/fingerprint_analysis/figure7_fcd_vs_train_all.png)
- [/Users/lintianjian/diffusion/outputs/fingerprint_analysis/figure7_fcd_vs_train_unique.png](/Users/lintianjian/diffusion/outputs/fingerprint_analysis/figure7_fcd_vs_train_unique.png)
- [/Users/lintianjian/diffusion/outputs/fingerprint_analysis/figure7_fcd_vs_raw_linker_fcd_vs_reference_all.png](/Users/lintianjian/diffusion/outputs/fingerprint_analysis/figure7_fcd_vs_raw_linker_fcd_vs_reference_all.png)
- [/Users/lintianjian/diffusion/outputs/fingerprint_analysis/figure7_fcd_vs_raw_linker_fcd_vs_reference_unique.png](/Users/lintianjian/diffusion/outputs/fingerprint_analysis/figure7_fcd_vs_raw_linker_fcd_vs_reference_unique.png)

后续可将本补充文档中的部分内容合并到主文稿第 5.2 节和第 6 章实验分析中。

## 7. 以真实 anchored linker 库作为参考的 FCD

除以 weak-anchor 训练集作为参考外，本文也进一步采用原始 anchored linker 库（`raw_linker`）作为参考分布，对训练集、各生成版本和 baseline 进行比较。该视角更适合回答：

- 训练集相对原始真实库发生了多大分布偏移；
- 不同生成模型是更接近真实 anchored linker 库，还是更接近 weak-anchor 训练分布；
- 外部 baseline 在真实 linker 空间中与原始库的偏离程度如何。

对应结果如下表：

| 数据集 | valid(all) | valid(unique) | FCD vs raw (all) | FCD vs raw (unique) |
| --- | ---: | ---: | ---: | ---: |
| raw_linker | 2749 | 2749 | 0.000 | 0.000 |
| train | 5020 | 2263 | 0.599 | 0.293 |
| gen_16d | 128 | 128 | 2.828 | 2.828 |
| gen_pad_suffix_ce005 | 128 | 124 | 3.999 | 4.001 |
| gen_32d | 128 | 128 | 4.062 | 4.062 |
| gen_pad_suffix_ce015 | 128 | 128 | 5.646 | 5.646 |
| Link-INVENT | 123 | 115 | 25.828 | 25.483 |

从这一视角看，结论仍然比较稳定：

- 训练集与原始 anchored linker 库非常接近，说明 weak-anchor 过程引入的是“任务化偏移”，但并未彻底脱离原始 linker 化学空间；
- 在生成模型中，`gen_16d` 仍然是最接近真实集的版本；
- `gen_pad_suffix_ce005` 和 `gen_32d` 居中，说明它们相比 16 维主模型引入了额外偏移；
- `gen_pad_suffix_ce015` 偏移最明显；
- `Link-INVENT` 相对真实集的 FCD 仍然极高，说明它的生成风格与当前 anchored linker 任务空间存在较大差异。

这组结果与之前基于训练集参考得到的排序基本一致，因此也进一步支持：当前 16 维主模型仍然是最均衡、最贴近任务分布的默认版本。
