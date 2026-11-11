
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


# =====================================
# 1. 读取实验数据
# =====================================

INPUT_FILE = (PROJECT_ROOT / "results" / "q4" / "q4_summary.csv")
OUTPUT_DIR = PROJECT_ROOT / "figures"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(INPUT_FILE)

depths = [1, 2, 3, 4]

minimax = (
    df[df["method"] == "Minimax"]
    .sort_values("depth")
)

alpha_beta = (
    df[df["method"] == "AlphaBeta"]
    .sort_values("depth")
)

assert len(minimax) == 4
assert len(alpha_beta) == 4

assert minimax["best_value"].tolist() == (
    alpha_beta["best_value"].tolist()
)

plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei"
]

plt.rcParams["axes.unicode_minus"] = False


# =====================================
# 2. 图1：访问节点数对照
# =====================================

fig, ax = plt.subplots(figsize=(8, 5))

x = range(len(depths))
width = 0.35

ax.bar(
    [i - width / 2 for i in x],
    minimax["visited_nodes_mean"],
    width,
    label="Minimax"
)

ax.bar(
    [i + width / 2 for i in x],
    alpha_beta["visited_nodes_mean"],
    width,
    label="Alpha-Beta"
)

ax.set_xticks(list(x))
ax.set_xticklabels([f"H={d}" for d in depths])

ax.set_xlabel("搜索深度")
ax.set_ylabel("平均访问节点数")
ax.set_title("不同搜索深度下的访问节点数对比")

ax.legend()
ax.grid(axis="y", alpha=0.3)

fig.tight_layout()

fig.savefig(
    OUTPUT_DIR / "q4_visited_nodes.png",
    dpi=300
)

plt.close(fig)


# =====================================
# 3. 图2：平均运行时间对照
# =====================================

fig, ax = plt.subplots(figsize=(8, 5))

ax.plot(
    depths,
    minimax["runtime_ms_mean"],
    marker="o",
    label="Minimax"
)

ax.plot(
    depths,
    alpha_beta["runtime_ms_mean"],
    marker="s",
    label="Alpha-Beta"
)

ax.set_xticks(depths)
ax.set_xticklabels([f"H={d}" for d in depths])

ax.set_xlabel("搜索深度")
ax.set_ylabel("平均运行时间（ms）")
ax.set_title("不同搜索深度下的运行时间对比")

ax.legend()
ax.grid(alpha=0.3)

fig.tight_layout()

fig.savefig(
    OUTPUT_DIR / "q4_runtime.png",
    dpi=300
)

plt.close(fig)


# =====================================
# 4. 图3：节点减少比例
# =====================================

merged = minimax[
    ["depth", "visited_nodes_mean"]
].merge(
    alpha_beta[
        ["depth", "visited_nodes_mean"]
    ],
    on="depth",
    suffixes=("_minimax", "_alpha_beta")
)

merged["reduction"] = (
    (
        merged["visited_nodes_mean_minimax"]
        - merged["visited_nodes_mean_alpha_beta"]
    )
    / merged["visited_nodes_mean_minimax"]
    * 100
)

fig, ax = plt.subplots(figsize=(8, 5))

bars = ax.bar(
    [f"H={d}" for d in merged["depth"]],
    merged["reduction"]
)

ax.set_xlabel("搜索深度")
ax.set_ylabel("访问节点减少比例（%）")
ax.set_title("Alpha-Beta 剪枝优化效果")

ax.grid(axis="y", alpha=0.3)

for bar, value in zip(
    bars,
    merged["reduction"]
):

    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        f"{value:.2f}%",
        ha="center",
        va="bottom"
    )

fig.tight_layout()

fig.savefig(
    OUTPUT_DIR / "q4_node_reduction.png",
    dpi=300
)

plt.close(fig)


# =====================================
# 5. 输出统计信息
# =====================================

print("===== 题目四图表生成完成 =====")

print("\n节点减少比例：")

print(
    merged[
        ["depth", "reduction"]
    ].to_string(index=False)
)

print("\n生成图片：")

print(OUTPUT_DIR / "q4_visited_nodes.png")
print(OUTPUT_DIR / "q4_runtime.png")
print(OUTPUT_DIR / "q4_node_reduction.png")
