# 当代人工智能实验二：搜索算法

本项目实现普通迷宫 A*、加权地形 A*、Minimax、Alpha-Beta，以及结合寻路与对抗搜索的宝藏分配任务。拓展实验比较启发函数、平局策略、地图分布、博弈树规模和子节点访问顺序。本文说明运行与复现方法，实验设计和结果分析见report。

## 1. 项目简介

四道题依次研究单位代价最短路径、加权地形最小代价、博弈树最优值及竞争宝藏分配。优化与拓展包括启发函数对照、障碍密度、同f平局处理、多地图验证、随机树规模、Oracle排序成本和Greedy排序。

## 2. 环境与依赖

已记录并完成基础验收的环境：Python **3.14.7**（Anaconda，64位）、Windows 11、AMD Ryzen 9 8945H。绘图依赖为 NumPy **2.5.4**、pandas **3.1.0**、Matplotlib **3.11.2**，与 `requirements.txt` 一致。

在 PowerShell 中进入项目根目录后安装依赖：

```powershell
cd C:\当代人工智能\lab2
python --version
python -m pip install -r requirements.txt
```

核心算法、测试和搜索实验使用 `heapq`、`itertools`、`time`、`random`、`statistics`、`csv` 等标准库，无须额外安装；第三方库用于绘图。Windows 11 已验收，其他支持这些依赖的平台尚未验证；中文图表字体可能需要调整。以下命令统一在项目根目录执行，使用 `-m` 运行模块；`-B` 避免生成缓存。不要使用 `-O`，否则测试中的 `assert` 会被禁用。

## 3. 项目目录结构与输入

```text
lab2/
├── algorithms/           # astar、weighted_astar、alpha_beta、treasure_game
├── tests/                # 五个基础测试
├── experiments/
│   ├── q1/               # 基准、障碍密度实验及绘图
│   ├── q2/               # 启发性质、地形、平局、多地图实验及绘图
│   ├── q3/               # 随机树、排序辅助实现及绘图
│   └── q4/               # 搜索深度、排序实验及绘图
├── results/q1—q4/        # 四个子目录，保存15份当前实验CSV
├── figures/              # 当前16张PNG
├── report.md             # 当前实验报告Markdown源文件
├── README.md
└── requirements.txt
```


|题目|输入及来源|接口与输出|
|---|---|---|
|一：普通迷宫|测试地图定义于 `tests/test_q1.py`；随机地图由密度脚本生成。二维网格，`X` 障碍，四邻接每步1|`algorithms.astar.find_path(start, goal, grid)`：含起终点的最短坐标序列，无解 `[]`|
|二：加权地形|测试地图定义于 `tests/test_q2.py`；实验脚本生成固定或随机地形，数字为进入格代价，`E` 进入代价0|`algorithms.weighted_astar.find_minimum_cost(start, goal, grid)`：最小总代价，无解 `-1`|
|三：博弈树|标准树在 `tests/test_alpha_beta.py`；随机树由实验脚本生成，无外部数据集。非叶有 `children`，叶有 `value`|`algorithms.alpha_beta.minimax(root)`、`alpha_beta(root)`：根节点博弈值|
|四：宝藏博弈|`algorithms/treasure_game.py` 中的 `GRID`、`POSITIONS`、`TREASURE_VALUES` 定义7×7地图、双方及四宝藏|先算距离矩阵并创建状态，调用 `minimax_game` 或 `alpha_beta_game`，返回值、行动、统计|


## 4. 实验配置与运行方法

完整流程为 **基础测试 → 性质验证 → 性能实验 → 绘图**。已有CSV可直接用于绘图，但绘图不会自动执行实验。

### 4.1 基础功能测试与性质验证

```powershell
python -B -m tests.test_q1
python -B -m tests.test_q2
python -B -m tests.test_q2_tiebreak
python -B -m tests.test_alpha_beta
python -B -m tests.test_treasure_game
python -B -m experiments.q2.experiment_q2_heuristic_validity
python -B -m experiments.q2.experiment_q2_consistency
```

前五个命令分别验证：题一路径与特殊输入、题二路径累计代价、题二平局策略保持成本、题三两算法值一致且发生剪枝、题四H=2/H=4标准值与行动。后两个命令分别验证标准图上的可采纳性和一致性。

这些脚本输出到终端，不覆盖CSV或PNG。打印题四6×6距离矩阵可运行：

```powershell
python -B -m algorithms.treasure_game
```

### 4.2 性能与拓展实验

```powershell
# 题一：基准计时（终端输出）、障碍密度
python -B -m experiments.q1.benchmark_q1
python -B -m experiments.q1.experiment_q1_density
# 题二：地形、平局策略、多地图
python -B -m experiments.q2.experiment_q2_terrain
python -B -m experiments.q2.experiment_q2_tiebreak
python -B -m experiments.q2.experiment_q2_multimap
# 题三：随机树与排序
python -B -m experiments.q3.experiment_q3_ordering
# 题四：深度与排序
python -B -m experiments.q4.experiment_q4
python -B -m experiments.q4.experiment_q4_ordering
```

另有 `experiments.q2.experiment_q2_counterexample` 搜索普通Manhattan实际返回非最优解的反例，上限500000次，不属于快速验收。

### 4.3 图表生成

先确保对应CSV存在，再在副本运行：

```powershell
python -B -m experiments.q1.plot_q1
python -B -m experiments.q2.plot_q2
python -B -m experiments.q2.plot_q2_multimap
python -B -m experiments.q3.plot_q3_ordering
python -B -m experiments.q4.plot_q4
python -B -m experiments.q4.plot_q4_ordering
```

## 5. 实验配置与结果说明

|实验|输入与参数|方法与重复设置|
|---|---|---|
|题一基准|标准迷宫|h=0、Manhattan；30批，每批1000次|
|题一密度|20×20，障碍概率10%/20%/30%，seed=2026；每组保留10张可达地图|h=0、Manhattan；同图每法5次|
|题二地形|10×10：全1、中央1/8高代价区域、随机1/3/8；随机seed=20261101|Dijkstra、修正启发A*；每场景每法30次|
|题二平局|同三种地形|Dijkstra、AStar_FIFO、AStar_LowerH；各30次|
|题二多地图|30张10×10无障碍图，权值1/3/8；seed从20261101至20261130|同图三方法各5次|
|题三随机树|深度3/4/5/6，分支2/3/4，每条件30树；seed=20261101；叶值整数[-100,100]|Minimax、AB FIFO/Reverse/Oracle；同树每法5次，总360树|
|题四深度|代码内固定棋局，H=1/2/3/4|Minimax、AB；每组预热20次，正式1000次|
|题四排序|相同棋局，H=2/3/4|AB FIFO/Reverse/Greedy；每组预热20次，正式1000次|

随机种子用于重复生成地图或树，但计时不会逐位一致；题三使用连续随机数流，复现时需要保留生成顺序。预热结果不计入正式统计。重复次数用于测量波动；各实验保存的均值、中位数等指标以CSV字段为准。

### 5.1 结果文件与提交

#### CSV与字段

文件名以下省略 `.csv` 后缀；当前数据位于对应 `results/q*/`

|位置|文件|用途与数据行数（不含表头）|
|---|---|---|
|q1|`q1_density_detail`、`q1_density_summary`|60行地图×方法记录、6行汇总；detail已是5次均值|
|q2|`q2_terrain_detail`、`q2_terrain_summary`|180行运行明细、6行场景汇总|
|q2|`q2_tiebreak_detail`、`q2_tiebreak_summary`|270行运行明细、9行汇总|
|q2|`q2_multimap_detail`、`q2_multimap_by_map`、`q2_multimap_summary`|450行运行明细、90行地图级统计、3行总体统计|
|q3|`q3_ordering_detail`、`q3_ordering_summary`|7200行明细、48行汇总|
|q4|`q4_detail`、`q4_summary`|8000行运行明细、8行汇总|
|q4|`q4_ordering_detail`、`q4_ordering_summary`|9000行运行明细、9行汇总|

主要字段：`density/scenario/depth/branching_factor` 为条件；`map_id/seed/tree_id/repeat/method` 标识样本与方法；`cost/path_length` 为路径代价与步数；`search_nodes/expanded_nodes/generated_nodes` 为搜索指标。博弈记录含 `best_value/best_action/visited_nodes/leaf_evaluations/pruned_branches/cutoffs`。时间单位均为ms；`_mean/_median/_std` 分别表示均值、中位数、样本标准差。题二 `runtime_ms_mean_mean` 是先按地图求均值再对地图求均值。题三配对键必须包含深度、分支和tree_id。

#### 图表对应关系

|绘图模块|输入CSV|`figures/`内输出PNG（省略后缀）|
|---|---|---|
|q1.plot_q1|q1_density_summary|q1_search_nodes、q1_expanded_nodes、q1_runtime|
|q2.plot_q2|q2_tiebreak_summary|q2_expanded_nodes、q2_runtime、q2_reduction_ratio|
|q2.plot_q2_multimap|q2_multimap_by_map|q2_multimap_boxplot、q2_multimap_reduction|
|q3.plot_q3_ordering|q3_ordering_summary|q3_node_comparison、q3_runtime_comparison、q3_oracle_time_breakdown|
|q4.plot_q4|q4_summary|q4_visited_nodes、q4_runtime、q4_node_reduction|
|q4.plot_q4_ordering|q4_ordering_summary|q4_ordering_nodes、q4_ordering_runtime|


#### 核心结果与提交内容

- 标准输出：题一最短路径7步；题二最小代价7；题三博弈值5；题四H=2为T2/-1、H=4为T3/3。
- 题三实际排序只有FIFO、Reverse、Oracle（Greedy仅用于题四）。题三D6-B4 Oracle正式访问268节点，搜索约0.6177ms、预计算9.6842ms、总10.3019ms。
- 题四H=4 Greedy节点47降至37，但均值时间约0.1726升至0.1959ms；不能用节点数替代实际耗时。


## 6. 结果验证与复现说明

|核验内容|预期|
|---|---|
|题一|标准路径含8个坐标、7步；邻接合法；无解、障碍起终点、起终点相同测试通过|
|题二|标准成本7；路径累计等于返回成本；平局测试两法成本17、扩展99/18|
|题二性质|标准5×5图普通Manhattan有20处高估；修正启发高估0，一致性违反边0|
|题三|Minimax与AB均返回5，AB确有剪枝，基础版本保持左到右|
|题四|两算法H2均-1/T2，H4均3/T3；访问数分别17/13与65/47|

题四输出应包含：

```text
最优博弈值一致性：PASS
H=2 标准结果验证：PASS
H=4 标准结果验证：PASS
```

## 7. 已知限制

- **题一输入边界：** `astar_search` 默认使用 h=0，正式接口 `find_path` 显式使用 Manhattan。当前实现对空地图和非矩形输入的检查不足，基础测试尚未覆盖全部异常输入。
- **题二启发函数适用范围：** 加权实现将 S、E 的进入代价均设为0。修正启发函数 `max(0, dM-1)` 的性质验证仅适用于当前 S/E 角点布局及数值地形权值至少为1的场景，不能直接推广至任意布局、零权或负权地形，以及小数权值。题四关键位置所在格子的进入权值均为1。
- **启发函数性质与实际表现：** 普通 Manhattan 在标准地图上已有20处高估真实剩余代价的证据，但当前未保存其实际返回非最优路径的搜索反例，因此不能将启发函数不可采纳直接等同于已观察到错误搜索结果。
- **题一数据统计：** 实验仅统计可达地图，不同障碍密度下生成尝试次数分别为10、15、25次，每组保留10张可达地图。60行 detail 数据表示地图与方法的组合记录，每条记录为5次运行的均值，并非60次独立搜索；原始单次计时未保存。
- **Oracle 排序成本：** Oracle 依赖预先计算真实 Minimax 值，正式搜索节点数和搜索时间不包含预计算过程。比较实际效率时，应同时考虑预计算成本。
- **拓展实验与计时局限：** 随机树和随机地图实验的结论受数据分布及随机种子限制，题四仅验证了一个固定棋局。Greedy 减少访问节点不代表运行时间必然下降，也不能据此证明其在所有场景下最优。固定运行顺序、系统负载及计时精度也可能影响较小的时间差异。