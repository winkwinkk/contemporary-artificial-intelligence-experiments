
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei"
]

plt.rcParams["axes.unicode_minus"] = False
# =====================================
# 1. 文件路径
# =====================================

# 项目根目录：lab2
ROOT = Path(__file__).resolve().parents[2]

CSV_FILE = ROOT / "results" / "q1" / "q1_density_summary.csv"

FIGURE_DIR = ROOT / "figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

if not CSV_FILE.exists():
    raise FileNotFoundError(f"找不到实验数据：{CSV_FILE}")

# =====================================
# 2. 读取实验数据
# =====================================

df = pd.read_csv(CSV_FILE)

required_columns = [
    "density",
    "method",
    "search_nodes",
    "expanded_nodes",
    "runtime_ms",
]

for column in required_columns:
    if column not in df.columns:
        raise ValueError(f"CSV 缺少字段：{column}")

df = df.sort_values(["density", "method"])

densities = sorted(df["density"].unique())

x_labels = [
    f"{density:.0%}"
    for density in densities
]

methods = {
    "h=0": "A* (h=0)",
    "Manhattan": "A* (Manhattan)"
}

styles = {
    "h=0": {"marker": "o"},
    "Manhattan": {"marker": "s"}
}

print("===== 题目一绘图 =====")
print(f"数据文件：{CSV_FILE}")
print(f"障碍密度：{x_labels}")
print(f"实验记录：{len(df)}")


# =====================================
# 3. 通用绘图函数
# =====================================

def plot_metric(
    column,
    title,
    ylabel,
    filename,
    show_values=False
):

    plt.figure(figsize=(9, 5.5))

    for method, label in methods.items():

        subset = (
            df[df["method"] == method]
            .sort_values("density")
        )

        values = subset[column].tolist()

        plt.plot(
            x_labels,
            values,
            marker=styles[method]["marker"],
            linewidth=2,
            markersize=7,
            label=label
        )

        if show_values:
            for x, y in zip(x_labels, values):
                plt.annotate(
                    f"{y:.3f}",
                    (x, y),
                    textcoords="offset points",
                    xytext=(0, 8),
                    ha="center",
                    fontsize=8
                )

    plt.title(title, fontsize=15)
    plt.xlabel("障碍密度", fontsize=12)
    plt.ylabel(ylabel, fontsize=12)

    plt.legend(fontsize=10)
    plt.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    output_path = FIGURE_DIR / filename

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"已保存：{output_path}")


# =====================================
# 4. 图一：平均访问节点数
# =====================================

plot_metric(
    column="search_nodes",
    title="不同障碍密度下的平均访问节点数对比",
    ylabel="平均访问节点数",
    filename="q1_search_nodes.png"
)


# =====================================
# 5. 图二：平均扩展节点数
# =====================================

plot_metric(
    column="expanded_nodes",
    title="不同障碍密度下的平均扩展节点数对比",
    ylabel="平均扩展节点数",
    filename="q1_expanded_nodes.png"
)


# =====================================
# 6. 图三：平均运行时间
# =====================================

plot_metric(
    column="runtime_ms",
    title="不同障碍密度下的平均运行时间对比",
    ylabel="平均运行时间（ms）",
    filename="q1_runtime.png",
    show_values=True
)


# =====================================
# 7. 完成
# =====================================

print("\n===== 题目一绘图全部完成 =====")
