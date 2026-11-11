
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (PROJECT_ROOT / "results" / "q4" / "q4_ordering_summary.csv")
OUTPUT_DIR = PROJECT_ROOT / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(INPUT_FILE)

methods = ["fifo", "reverse", "greedy"]
labels = ["FIFO", "Reverse", "Greedy"]
depths = [2, 3, 4]

plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei", "SimHei"
]
plt.rcParams["axes.unicode_minus"] = False

x = np.arange(len(depths))
width = 0.24

# =====================================
# 图1：访问节点数
# =====================================

fig, ax = plt.subplots(figsize=(9, 5))

for i, (method, label) in enumerate(zip(methods, labels)):

    values = []

    for depth in depths:
        row = df[
            (df["depth"] == depth) &
            (df["method"] == method)
        ]

        values.append(
            float(row["visited_nodes_mean"].iloc[0])
        )

    ax.bar(
        x + (i - 1) * width,
        values,
        width,
        label=label
    )

ax.set_title("不同排序策略的搜索节点数对比")
ax.set_xlabel("搜索深度")
ax.set_ylabel("平均访问节点数")
ax.set_xticks(x)
ax.set_xticklabels([f"H={d}" for d in depths])
ax.legend()
ax.grid(axis="y", alpha=0.3)

fig.tight_layout()
fig.savefig(
    OUTPUT_DIR / "q4_ordering_nodes.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close(fig)

# =====================================
# 图2：运行时间
# =====================================

fig, ax = plt.subplots(figsize=(9, 5))

for i, (method, label) in enumerate(zip(methods, labels)):

    values = []

    for depth in depths:
        row = df[
            (df["depth"] == depth) &
            (df["method"] == method)
        ]

        values.append(
            float(row["runtime_ms_median"].iloc[0])
        )

    ax.bar(
        x + (i - 1) * width,
        values,
        width,
        label=label
    )

ax.set_title("不同排序策略的运行时间对比")
ax.set_xlabel("搜索深度")
ax.set_ylabel("运行时间中位数（ms）")
ax.set_xticks(x)
ax.set_xticklabels([f"H={d}" for d in depths])
ax.legend()
ax.grid(axis="y", alpha=0.3)

fig.tight_layout()
fig.savefig(
    OUTPUT_DIR / "q4_ordering_runtime.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close(fig)

print("===== 题目四图表生成完成 =====")
print("figures/q4_ordering_nodes.png")
print("figures/q4_ordering_runtime.png")
