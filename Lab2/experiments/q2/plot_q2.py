
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


# =========================
# 1. 文件路径
# =========================

RESULT_DIR = PROJECT_ROOT / "results" / "q2"
FIGURE_DIR = PROJECT_ROOT / "figures"

FIGURE_DIR.mkdir(parents=True, exist_ok=True)

data = pd.read_csv(
    RESULT_DIR / "q2_tiebreak_summary.csv"
)

# 检查数据
required_columns = {
    "scenario",
    "method",
    "expanded_nodes_mean",
    "runtime_ms_mean"
}

assert required_columns.issubset(data.columns)


# 中文字体设置
plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "Arial Unicode MS"
]

plt.rcParams["axes.unicode_minus"] = False


scenario_names = {
    "uniform": "均匀地形",
    "high_cost_region": "高代价区域",
    "random_mixed": "随机混合地形"
}

method_names = {
    "Dijkstra": "Dijkstra",
    "AStar_FIFO": "原始 A*",
    "AStar_LowerH": "优化 A*"
}

scenarios = list(scenario_names.keys())
methods = list(method_names.keys())


# =========================
# 2. 图1：扩展节点数对照
# =========================

fig, ax = plt.subplots(figsize=(9, 5))

x = np.arange(len(scenarios))
width = 0.24

for i, method in enumerate(methods):

    values = []

    for scenario in scenarios:

        row = data[
            (data["scenario"] == scenario)
            & (data["method"] == method)
        ]

        values.append(
            float(row["expanded_nodes_mean"].iloc[0])
        )

    bars = ax.bar(
        x + (i - 1) * width,
        values,
        width,
        label=method_names[method]
    )

    ax.bar_label(bars, fmt="%.0f", padding=3)

ax.set_title("不同地形下三种算法的扩展节点数")
ax.set_ylabel("扩展节点数")
ax.set_xticks(x)
ax.set_xticklabels(
    [scenario_names[s] for s in scenarios]
)
ax.legend()
ax.grid(axis="y", alpha=0.3)

fig.tight_layout()

fig.savefig(
    FIGURE_DIR / "q2_expanded_nodes.png",
    dpi=300
)

plt.close(fig)


# =========================
# 3. 图2：平均运行时间对照
# =========================

fig, ax = plt.subplots(figsize=(9, 5))

for i, method in enumerate(methods):

    values = []

    for scenario in scenarios:

        row = data[
            (data["scenario"] == scenario)
            & (data["method"] == method)
        ]

        values.append(
            float(row["runtime_ms_mean"].iloc[0])
        )

    bars = ax.bar(
        x + (i - 1) * width,
        values,
        width,
        label=method_names[method]
    )

    ax.bar_label(
        bars,
        fmt="%.3f",
        padding=3
    )

ax.set_title("不同地形下三种算法的平均运行时间")
ax.set_ylabel("运行时间（ms）")
ax.set_xticks(x)
ax.set_xticklabels(
    [scenario_names[s] for s in scenarios]
)
ax.legend()
ax.grid(axis="y", alpha=0.3)

fig.tight_layout()

fig.savefig(
    FIGURE_DIR / "q2_runtime.png",
    dpi=300
)

plt.close(fig)


# =========================
# 4. 图3：排序优化节点减少比例
# =========================

reductions = []

for scenario in scenarios:

    original = data[
        (data["scenario"] == scenario)
        & (data["method"] == "AStar_FIFO")
    ]["expanded_nodes_mean"].iloc[0]

    optimized = data[
        (data["scenario"] == scenario)
        & (data["method"] == "AStar_LowerH")
    ]["expanded_nodes_mean"].iloc[0]

    reduction = (
        (original - optimized) / original * 100
    )

    reductions.append(reduction)


fig, ax = plt.subplots(figsize=(8, 5))

bars = ax.bar(
    [scenario_names[s] for s in scenarios],
    reductions
)

ax.bar_label(
    bars,
    fmt="%.2f%%",
    padding=4
)

ax.axhline(0, linewidth=0.8)

ax.set_title("排序优化后的扩展节点减少比例")
ax.set_ylabel("节点减少比例（%）")
ax.grid(axis="y", alpha=0.3)

fig.tight_layout()

fig.savefig(
    FIGURE_DIR / "q2_reduction_ratio.png",
    dpi=300
)

plt.close(fig)


print("===== A* 实验图表生成完成 =====")

for file in FIGURE_DIR.glob("q2_*.png"):
    print(file)
