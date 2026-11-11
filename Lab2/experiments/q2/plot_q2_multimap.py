
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
from matplotlib.lines import Line2D

# =========================
# 1. 文件路径与数据读取
# =========================

INPUT_FILE = (PROJECT_ROOT / "results" / "q2" / "q2_multimap_by_map.csv")
OUTPUT_DIR = PROJECT_ROOT / "figures"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(INPUT_FILE)

# 检查数据完整性
assert len(df) == 90, "地图级数据应有90条记录"

methods = [
    "Dijkstra",
    "AStar_FIFO",
    "AStar_LowerH"
]

labels = [
    "Dijkstra",
    "Original A*",
    "Optimized A*"
]

for method in methods:
    assert (
        df["method"] == method
    ).sum() == 30, f"{method} 数据数量不正确"


# 中文字体
plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei"
]

plt.rcParams["axes.unicode_minus"] = False


# =========================
# 2. 图1：扩展节点数箱线图
# =========================

data = []

for method in methods:

    selected = df[df["method"] == method]

    values = selected[
        "expanded_nodes_mean"
    ].tolist()

    data.append(values)


fig, ax = plt.subplots(figsize=(9, 5))

ax.boxplot(
    data,
    tick_labels=labels,
    showmeans=True,
    meanprops={
        "marker": "^",
        "markerfacecolor": "green",
        "markeredgecolor": "green"
    },
    medianprops={
        "color": "tab:orange",
        "linewidth": 1.5
    }
)

# 添加均值和中位数图例
legend_items = [
    Line2D(
        [0], [0],
        color="tab:orange",
        linewidth=1.5,
        label="中位数"
    ),
    Line2D(
        [0], [0],
        marker="^",
        color="green",
        linestyle="None",
        label="均值"
    )
]

ax.legend(handles=legend_items, loc="upper right")

ax.set_ylim(0, 105)

ax.set_title(
    "30张随机地图上的平均扩展节点数分布"
)

ax.set_ylabel("平均扩展节点数")
ax.set_xlabel("搜索算法")

ax.grid(
    axis="y",
    alpha=0.3
)

fig.tight_layout()

fig.savefig(
    OUTPUT_DIR / "q2_multimap_boxplot.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)


# =========================
# 3. 图2：逐地图节点减少比例
# =========================

fifo = df[
    df["method"] == "AStar_FIFO"
][
    ["map_id", "expanded_nodes_mean"]
].rename(
    columns={
        "expanded_nodes_mean": "original"
    }
)

optimized = df[
    df["method"] == "AStar_LowerH"
][
    ["map_id", "expanded_nodes_mean"]
].rename(
    columns={
        "expanded_nodes_mean": "optimized"
    }
)

merged = fifo.merge(
    optimized,
    on="map_id"
)

merged = merged.sort_values("map_id")

merged["reduction"] = (
    (merged["original"] - merged["optimized"])
    / merged["original"] * 100
)


fig, ax = plt.subplots(figsize=(11, 5))

bars = ax.bar(
    merged["map_id"],
    merged["reduction"]
)

ax.axhline(
    0,
    linewidth=1
)

ax.set_title(
    "30张随机地图上的排序优化效果"
)

ax.set_xlabel("地图编号")
ax.set_ylabel("扩展节点减少比例（%）")

ax.set_xticks(range(1, 31))

ax.grid(
    axis="y",
    alpha=0.3
)

fig.tight_layout()

fig.savefig(
    OUTPUT_DIR / "q2_multimap_reduction.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)


# =========================
# 4. 输出统计信息
# =========================

print("===== 多随机地图图表生成完成 =====")

print("地图数量：", len(merged))

print(
    "平均节点减少比例：",
    round(merged["reduction"].mean(), 2),
    "%"
)

print(
    "中位数节点减少比例：",
    round(merged["reduction"].median(), 2),
    "%"
)

print(
    "最大节点减少比例：",
    round(merged["reduction"].max(), 2),
    "%"
)

print(
    "最小节点减少比例：",
    round(merged["reduction"].min(), 2),
    "%"
)

print(
    "节点减少的地图数量：",
    (merged["reduction"] > 0).sum()
)

print(
    "节点数量不变的地图数量：",
    (merged["reduction"] == 0).sum()
)

print(
    "节点增加的地图数量：",
    (merged["reduction"] < 0).sum()
)

print("\n生成的图片：")

print(OUTPUT_DIR / "q2_multimap_boxplot.png")
print(OUTPUT_DIR / "q2_multimap_reduction.png")
