
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


# =====================================
# 1. 文件路径与数据读取
# =====================================

INPUT_FILE = (PROJECT_ROOT / "results" / "q3" / "q3_ordering_summary.csv")
OUTPUT_DIR = PROJECT_ROOT / "figures"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(INPUT_FILE)

methods = [
    "Minimax",
    "AlphaBeta_FIFO",
    "AlphaBeta_Reverse",
    "AlphaBeta_Oracle"
]

labels = [
    "Minimax",
    "Alpha-Beta FIFO",
    "Alpha-Beta Reverse",
    "Alpha-Beta Oracle"
]

required_columns = [
    "depth",
    "branching_factor",
    "method",
    "visited_nodes_mean",
    "search_runtime_ms_mean",
    "oracle_precompute_ms_mean",
    "total_runtime_ms_mean"
]

for col in required_columns:
    assert col in df.columns, f"缺少字段：{col}"

assert len(df) == 48, "汇总数据应为48条"

for method in methods:
    assert (df["method"] == method).sum() == 12

# 按树深度、分支因子排序
df = df.sort_values(
    ["depth", "branching_factor"]
)

conditions = df[
    ["depth", "branching_factor"]
].drop_duplicates()

x_labels = [
    f"D{row.depth}-B{row.branching_factor}"
    for row in conditions.itertuples()
]

plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei"
]
plt.rcParams["axes.unicode_minus"] = False


# =====================================
# 2. 图1：四种算法访问节点数对比
# =====================================

fig, ax = plt.subplots(figsize=(12, 6))

for method, label in zip(methods, labels):

    selected = df[df["method"] == method]

    values = selected[
        "visited_nodes_mean"
    ].tolist()

    ax.plot(
        x_labels,
        values,
        marker="o",
        linewidth=2,
        label=label
    )

ax.set_title(
    "不同树规模下四种算法的节点访问数量对比"
)

ax.set_xlabel("树深度与分支因子")
ax.set_ylabel("平均访问节点数（对数坐标）")

ax.set_yscale("log")
ax.legend()
ax.grid(axis="y", alpha=0.3, which="both")

plt.xticks(rotation=45)
fig.tight_layout()

fig.savefig(
    OUTPUT_DIR / "q3_node_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)


# =====================================
# 3. 图2：四种算法总运行时间对比
# =====================================

fig, ax = plt.subplots(figsize=(12, 6))

for method, label in zip(methods, labels):

    selected = df[df["method"] == method]

    values = selected[
        "total_runtime_ms_mean"
    ].tolist()

    ax.plot(
        x_labels,
        values,
        marker="o",
        linewidth=2,
        label=label
    )

ax.set_title(
    "不同树规模下四种算法的总运行时间对比"
)

ax.set_xlabel("树深度与分支因子")
ax.set_ylabel("平均总运行时间（ms）")

ax.legend()
ax.grid(axis="y", alpha=0.3)

plt.xticks(rotation=45)
fig.tight_layout()

fig.savefig(
    OUTPUT_DIR / "q3_runtime_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)


# =====================================
# 4. 图3：Oracle 时间成本分解
# =====================================

oracle = df[
    df["method"] == "AlphaBeta_Oracle"
].copy()

oracle = oracle.sort_values(
    ["depth", "branching_factor"]
)

oracle_labels = [
    f"D{row.depth}-B{row.branching_factor}"
    for row in oracle.itertuples()
]

search_time = oracle[
    "search_runtime_ms_mean"
].tolist()

precompute_time = oracle[
    "oracle_precompute_ms_mean"
].tolist()

fig, ax = plt.subplots(figsize=(12, 6))

ax.bar(
    oracle_labels,
    search_time,
    label="Alpha-Beta Search"
)

ax.bar(
    oracle_labels,
    precompute_time,
    bottom=search_time,
    label="Oracle Precomputation"
)

ax.set_title(
    "Oracle 排序策略的运行时间成本分解"
)

ax.set_xlabel("树深度与分支因子")
ax.set_ylabel("平均运行时间（ms）")

ax.legend()
ax.grid(axis="y", alpha=0.3)

plt.xticks(rotation=45)
fig.tight_layout()

fig.savefig(
    OUTPUT_DIR / "q3_oracle_time_breakdown.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)


# =====================================
# 5. 输出数据检查与统计结果
# =====================================

print("===== 题目三图表生成完成 =====")

print("实验条件数量：", len(conditions))
print("算法数量：", df["method"].nunique())
print("汇总记录数量：", len(df))

print("\nOracle 时间统计：")

print(
    "平均搜索耗时：",
    round(oracle["search_runtime_ms_mean"].mean(), 4),
    "ms"
)

print(
    "平均预计算耗时：",
    round(oracle["oracle_precompute_ms_mean"].mean(), 4),
    "ms"
)

print(
    "平均总耗时：",
    round(oracle["total_runtime_ms_mean"].mean(), 4),
    "ms"
)

print("\n生成的图片：")

print(OUTPUT_DIR / "q3_node_comparison.png")
print(OUTPUT_DIR / "q3_runtime_comparison.png")
print(OUTPUT_DIR / "q3_oracle_time_breakdown.png")
