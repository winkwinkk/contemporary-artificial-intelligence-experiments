
import csv
import statistics
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

from algorithms.treasure_game import (
    calculate_distance_matrix,
    create_initial_state,
    minimax_game,
    alpha_beta_game
)


# =====================================
# 1. 实验参数
# =====================================

DEPTHS = [1, 2, 3, 4]

WARMUP = 20
REPEATS = 1000

OUTPUT_DIR = PROJECT_ROOT / "results" / "q4"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DETAIL_FILE = OUTPUT_DIR / "q4_detail.csv"
SUMMARY_FILE = OUTPUT_DIR / "q4_summary.csv"


# =====================================
# 2. 实验准备
# =====================================

distance_matrix = calculate_distance_matrix()
initial_state = create_initial_state()

methods = {
    "Minimax": minimax_game,
    "AlphaBeta": alpha_beta_game
}

detail_records = []
summary_records = []


# =====================================
# 3. 重复运行实验
# =====================================

print("===== 题目四：Minimax 与 Alpha-Beta 性能实验 =====")
print("搜索深度：", DEPTHS)
print("预热次数：", WARMUP)
print("重复次数：", REPEATS)

for depth in DEPTHS:

    print(f"\n===== 搜索深度 H={depth} =====")

    for method_name, algorithm in methods.items():

        # 预热
        for _ in range(WARMUP):

            algorithm(
                initial_state,
                distance_matrix,
                depth
            )

        times = []

        node_counts = []
        expanded_counts = []
        leaf_counts = []
        pruned_counts = []
        cutoff_counts = []

        expected_value = None
        expected_action = None

        for repeat in range(1, REPEATS + 1):

            value, action, stats = algorithm(
                initial_state,
                distance_matrix,
                depth
            )

            # 正确性检查
            if expected_value is None:
                expected_value = value
                expected_action = action

            assert value == expected_value
            assert action == expected_action

            times.append(stats["runtime_ms"])

            node_counts.append(stats["visited_nodes"])
            expanded_counts.append(stats["expanded_nodes"])
            leaf_counts.append(stats["leaf_evaluations"])

            pruned_counts.append(
                stats.get("pruned_branches", 0)
            )

            cutoff_counts.append(
                stats.get("cutoffs", 0)
            )

            detail_records.append({
                "depth": depth,
                "method": method_name,
                "repeat": repeat,
                "best_value": value,
                "best_action": action,
                "visited_nodes": stats["visited_nodes"],
                "expanded_nodes": stats["expanded_nodes"],
                "leaf_evaluations": stats["leaf_evaluations"],
                "pruned_branches": stats.get("pruned_branches", 0),
                "cutoffs": stats.get("cutoffs", 0),
                "runtime_ms": stats["runtime_ms"]
            })

        summary_records.append({
            "depth": depth,
            "method": method_name,
            "best_value": expected_value,
            "best_action": expected_action,
            "visited_nodes_mean": statistics.mean(node_counts),
            "expanded_nodes_mean": statistics.mean(expanded_counts),
            "leaf_evaluations_mean": statistics.mean(leaf_counts),
            "pruned_branches_mean": statistics.mean(pruned_counts),
            "cutoffs_mean": statistics.mean(cutoff_counts),
            "runtime_ms_mean": statistics.mean(times),
            "runtime_ms_median": statistics.median(times)
        })

        print(f"\n{method_name}")
        print("最优博弈值：", expected_value)
        print("最优行动：", expected_action)
        print("平均访问节点：", round(statistics.mean(node_counts), 2))
        print("平均扩展节点：", round(statistics.mean(expanded_counts), 2))
        print("平均叶节点：", round(statistics.mean(leaf_counts), 2))
        print("平均运行时间(ms)：", round(statistics.mean(times), 6))
        print("运行时间中位数(ms)：", round(statistics.median(times), 6))


# =====================================
# 4. 保存 CSV
# =====================================

def save_csv(path, records):

    if not records:
        return

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=records[0].keys()
        )

        writer.writeheader()
        writer.writerows(records)


save_csv(DETAIL_FILE, detail_records)
save_csv(SUMMARY_FILE, summary_records)


# =====================================
# 5. 实验完成
# =====================================

print("\n================================")
print("===== 题目四实验全部完成 =====")

print("详细记录数：", len(detail_records))
print("汇总记录数：", len(summary_records))

print("详细结果：", DETAIL_FILE)
print("汇总结果：", SUMMARY_FILE)
