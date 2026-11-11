
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

DEPTHS = [2, 3, 4]

STRATEGIES = [
    "fifo",
    "reverse",
    "greedy"
]

WARMUP_TIMES = 20
REPEAT_TIMES = 1000

RESULT_DIR = PROJECT_ROOT / "results" / "q4"

DETAIL_FILE = RESULT_DIR / "q4_ordering_detail.csv"
SUMMARY_FILE = RESULT_DIR / "q4_ordering_summary.csv"


# =====================================
# 2. 实验方法
# =====================================

def run_algorithm(state, distance_matrix, depth, method):

    return alpha_beta_game(
        state,
        distance_matrix,
        depth,
        ordering=method
    )


# =====================================
# 3. 统计辅助函数
# =====================================

def mean_value(rows, key):

    return statistics.mean(
        float(row[key]) for row in rows
    )


def median_value(rows, key):

    return statistics.median(
        float(row[key]) for row in rows
    )


# =====================================
# 4. 主实验
# =====================================

def run_experiment():

    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    distance_matrix = calculate_distance_matrix()

    # 每次实验使用相同的初始状态
    initial_state = create_initial_state()

    detail_rows = []
    summary_rows = []

    total_runs = len(DEPTHS) * len(STRATEGIES)

    print("===== 题目四：Alpha-Beta 节点排序实验 =====")
    print("搜索深度：", DEPTHS)
    print("排序策略：", STRATEGIES)
    print("预热次数：", WARMUP_TIMES)
    print("重复次数：", REPEAT_TIMES)
    print("-------------------------------------")

    for depth_index, depth in enumerate(DEPTHS, start=1):

        print(f"\n===== 搜索深度 H={depth} =====")

        depth_results = {}

        # -----------------------------
        # 逐一测试三种排序策略
        # -----------------------------

        for method in STRATEGIES:

            # 预热，不记录结果
            for _ in range(WARMUP_TIMES):

                run_algorithm(
                    initial_state,
                    distance_matrix,
                    depth,
                    method
                )

            current_rows = []

            # 正式重复实验
            for repeat in range(1, REPEAT_TIMES + 1):

                best_value, best_action, stats = run_algorithm(
                    initial_state,
                    distance_matrix,
                    depth,
                    method
                )

                row = {
                    "depth": depth,
                    "method": method,
                    "repeat": repeat,
                    "best_value": best_value,
                    "best_action": best_action,
                    "visited_nodes": stats["visited_nodes"],
                    "expanded_nodes": stats["expanded_nodes"],
                    "leaf_evaluations": stats["leaf_evaluations"],
                    "pruned_branches": stats["pruned_branches"],
                    "cutoffs": stats["cutoffs"],
                    "runtime_ms": stats["runtime_ms"]
                }

                current_rows.append(row)
                detail_rows.append(row)

            depth_results[method] = current_rows

            # -------------------------
            # 输出单种策略统计
            # -------------------------

            avg_nodes = mean_value(
                current_rows, "visited_nodes"
            )

            avg_runtime = mean_value(
                current_rows, "runtime_ms"
            )

            median_runtime = median_value(
                current_rows, "runtime_ms"
            )

            print(f"\n{method.upper()}")
            print(f"平均访问节点：{avg_nodes:.2f}")
            print(f"平均扩展节点：{mean_value(current_rows, 'expanded_nodes'):.2f}")
            print(f"平均叶节点：{mean_value(current_rows, 'leaf_evaluations'):.2f}")
            print(f"平均剪枝分支：{mean_value(current_rows, 'pruned_branches'):.2f}")
            print(f"平均运行时间：{avg_runtime:.6f} ms")
            print(f"运行时间中位数：{median_runtime:.6f} ms")

            summary_rows.append({
                "depth": depth,
                "method": method,
                "best_value": current_rows[0]["best_value"],
                "best_action": current_rows[0]["best_action"],
                "visited_nodes_mean": avg_nodes,
                "expanded_nodes_mean": mean_value(
                    current_rows, "expanded_nodes"
                ),
                "leaf_evaluations_mean": mean_value(
                    current_rows, "leaf_evaluations"
                ),
                "pruned_branches_mean": mean_value(
                    current_rows, "pruned_branches"
                ),
                "cutoffs_mean": mean_value(
                    current_rows, "cutoffs"
                ),
                "runtime_ms_mean": avg_runtime,
                "runtime_ms_median": median_runtime
            })

        # -----------------------------
        # 正确性检查
        # -----------------------------

        values = {
            method: {
                row["best_value"]
                for row in depth_results[method]
            }
            for method in STRATEGIES
        }

        all_values = set()

        for method_values in values.values():
            all_values.update(method_values)

        if len(all_values) != 1:
            raise AssertionError(
                f"H={depth} 不同排序策略的博弈值不一致：{values}"
            )

        print("\n最优博弈值一致性：PASS")

        # -----------------------------
        # 与 FIFO 比较节点减少比例
        # -----------------------------

        fifo_nodes = mean_value(
            depth_results["fifo"],
            "visited_nodes"
        )

        for method in ["reverse", "greedy"]:

            method_nodes = mean_value(
                depth_results[method],
                "visited_nodes"
            )

            reduction = (
                (fifo_nodes - method_nodes)
                / fifo_nodes * 100
            )

            print(
                f"{method.upper()} 相比 FIFO 节点减少比例："
                f"{reduction:.2f}%"
            )

    # =================================
    # 5. 保存详细 CSV
    # =================================

    detail_fields = [
        "depth",
        "method",
        "repeat",
        "best_value",
        "best_action",
        "visited_nodes",
        "expanded_nodes",
        "leaf_evaluations",
        "pruned_branches",
        "cutoffs",
        "runtime_ms"
    ]

    with open(
        DETAIL_FILE,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=detail_fields
        )

        writer.writeheader()
        writer.writerows(detail_rows)

    # =================================
    # 6. 保存汇总 CSV
    # =================================

    summary_fields = [
        "depth",
        "method",
        "best_value",
        "best_action",
        "visited_nodes_mean",
        "expanded_nodes_mean",
        "leaf_evaluations_mean",
        "pruned_branches_mean",
        "cutoffs_mean",
        "runtime_ms_mean",
        "runtime_ms_median"
    ]

    with open(
        SUMMARY_FILE,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=summary_fields
        )

        writer.writeheader()
        writer.writerows(summary_rows)

    # =================================
    # 7. 完成信息
    # =================================

    print("\n=====================================")
    print("===== 题目四节点排序实验全部完成 =====")
    print("实验深度数量：", len(DEPTHS))
    print("排序策略数量：", len(STRATEGIES))
    print("每种条件重复次数：", REPEAT_TIMES)
    print("详细记录数量：", len(detail_rows))
    print("汇总记录数量：", len(summary_rows))
    print("详细结果：", DETAIL_FILE)
    print("汇总结果：", SUMMARY_FILE)
    print("=====================================")


if __name__ == "__main__":
    run_experiment()
