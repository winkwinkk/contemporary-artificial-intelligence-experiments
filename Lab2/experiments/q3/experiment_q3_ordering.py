
import csv
import random
import statistics

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

from algorithms.alpha_beta import (
    TreeNode,
    minimax_search
)

from experiments.q3.alpha_beta_experiments import (
    alpha_beta_ordered_search
)


# ==========================================
# 1. 实验参数
# ==========================================

SEED = 20261101

TREE_DEPTHS = [3, 4, 5, 6]

BRANCHING_FACTORS = [2, 3, 4]

MAPS_PER_SETTING = 30

REPEATS = 5

VALUE_MIN = -100
VALUE_MAX = 100

RESULT_DIR = PROJECT_ROOT / "results" / "q3"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

DETAIL_FILE = RESULT_DIR / "q3_ordering_detail.csv"
SUMMARY_FILE = RESULT_DIR / "q3_ordering_summary.csv"


METHODS = [
    "Minimax",
    "AlphaBeta_FIFO",
    "AlphaBeta_Reverse",
    "AlphaBeta_Oracle"
]


STAT_KEYS = [
    "visited_nodes",
    "expanded_nodes",
    "leaf_evaluations",
    "pruned_branches",
    "cutoffs",
    "search_runtime_ms",
    "oracle_precompute_ms",
    "total_runtime_ms"
]


# ==========================================
# 2. 随机博弈树生成
# ==========================================

def generate_random_tree(depth, branching_factor, rng):
    """
    生成完整随机博弈树。

    depth:
        根节点到叶节点的边数。

    branching_factor:
        每个非叶节点的子节点数量。

    根节点为 MAX，之后交替 MAX/MIN。
    """

    if depth == 0:
        return TreeNode(
            value=rng.randint(VALUE_MIN, VALUE_MAX)
        )

    children = []

    for _ in range(branching_factor):

        child = generate_random_tree(
            depth - 1,
            branching_factor,
            rng
        )

        children.append(child)

    return TreeNode(children=children)


# ==========================================
# 3. 统一统计字段
# ==========================================

def normalize_minimax_stats(stats):
    """
    为 Minimax 补充统一的时间统计字段。

    Minimax 不存在 Oracle 预计算。
    """

    runtime = stats["runtime_ms"]

    result = dict(stats)

    result["search_runtime_ms"] = runtime
    result["oracle_precompute_ms"] = 0.0
    result["total_runtime_ms"] = runtime

    return result


# ==========================================
# 4. 运行单棵树的四种算法
# ==========================================

def run_one_tree(root):

    results = {}

    # Minimax
    value, stats = minimax_search(root)

    results["Minimax"] = (
        value,
        normalize_minimax_stats(stats)
    )

    # 原始顺序 Alpha-Beta
    value, stats = alpha_beta_ordered_search(
        root,
        strategy="fifo"
    )

    results["AlphaBeta_FIFO"] = (value, stats)

    # 反向排序 Alpha-Beta
    value, stats = alpha_beta_ordered_search(
        root,
        strategy="reverse"
    )

    results["AlphaBeta_Reverse"] = (value, stats)

    # Oracle 理想排序 Alpha-Beta
    value, stats = alpha_beta_ordered_search(
        root,
        strategy="oracle"
    )

    results["AlphaBeta_Oracle"] = (value, stats)

    # 正确性检查
    reference_value = results["Minimax"][0]

    for method, (value, _) in results.items():

        assert value == reference_value, (
            f"{method} 结果不一致："
            f"{value} != {reference_value}"
        )

    return results


# ==========================================
# 5. 主实验
# ==========================================

def main():

    rng = random.Random(SEED)

    detail_records = []
    summary_records = []

    total_trees = (
        len(TREE_DEPTHS)
        * len(BRANCHING_FACTORS)
        * MAPS_PER_SETTING
    )

    completed = 0

    print("===== Alpha-Beta 节点排序实验 =====")

    print("随机种子：", SEED)
    print("树深度：", TREE_DEPTHS)
    print("分支因子：", BRANCHING_FACTORS)
    print("每种条件随机树数量：", MAPS_PER_SETTING)
    print("每棵树重复次数：", REPEATS)
    print("算法数量：", len(METHODS))

    print("-------------------------------------")

    for depth in TREE_DEPTHS:

        for branching in BRANCHING_FACTORS:

            print(
                f"\n===== 深度 {depth}，"
                f"分支因子 {branching} ====="
            )

            setting_records = {
                method: {
                    key: []
                    for key in STAT_KEYS
                }
                for method in METHODS
            }

            for tree_id in range(1, MAPS_PER_SETTING + 1):

                root = generate_random_tree(
                    depth,
                    branching,
                    rng
                )

                # 同一棵树重复测量
                for repeat in range(1, REPEATS + 1):

                    results = run_one_tree(root)

                    for method, (value, stats) in results.items():

                        record = {
                            "depth": depth,
                            "branching_factor": branching,
                            "tree_id": tree_id,
                            "repeat": repeat,
                            "method": method,
                            "best_value": value
                        }

                        # 保存统一统计字段
                        for key in STAT_KEYS:

                            record[key] = stats[key]

                            setting_records[method][key].append(
                                stats[key]
                            )

                        detail_records.append(record)

                completed += 1

                if tree_id % 10 == 0:

                    print(
                        f"已完成：{completed}/{total_trees} "
                        f"棵随机树"
                    )

            # ==================================
            # 当前实验条件下的统计汇总
            # ==================================

            for method in METHODS:

                values = setting_records[method]

                summary = {
                    "depth": depth,
                    "branching_factor": branching,
                    "method": method
                }

                for key in STAT_KEYS:

                    summary[f"{key}_mean"] = (
                        statistics.mean(values[key])
                    )

                    summary[f"{key}_median"] = (
                        statistics.median(values[key])
                    )

                summary_records.append(summary)

                print(
                    f"{method}: "
                    f"访问节点 "
                    f"{statistics.mean(values['visited_nodes']):.2f} | "
                    f"搜索时间 "
                    f"{statistics.mean(values['search_runtime_ms']):.4f} ms | "
                    f"预计算时间 "
                    f"{statistics.mean(values['oracle_precompute_ms']):.4f} ms | "
                    f"总时间 "
                    f"{statistics.mean(values['total_runtime_ms']):.4f} ms"
                )

    # ==========================================
    # 6. 保存详细 CSV
    # ==========================================

    detail_fields = [
        "depth",
        "branching_factor",
        "tree_id",
        "repeat",
        "method",
        "best_value"
    ] + STAT_KEYS

    with open(
        DETAIL_FILE,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=detail_fields
        )

        writer.writeheader()
        writer.writerows(detail_records)

    # ==========================================
    # 7. 保存汇总 CSV
    # ==========================================

    summary_fields = list(
        summary_records[0].keys()
    )

    with open(
        SUMMARY_FILE,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=summary_fields
        )

        writer.writeheader()
        writer.writerows(summary_records)

    # ==========================================
    # 8. 实验完成信息
    # ==========================================

    print("\n=====================================")
    print("===== 实验全部完成 =====")

    print("随机树数量：", total_trees)
    print("详细记录数量：", len(detail_records))
    print("汇总记录数量：", len(summary_records))

    print("\n详细数据：", DETAIL_FILE)
    print("汇总数据：", SUMMARY_FILE)

    print("\n所有算法的结果一致性检查：PASS")


if __name__ == "__main__":
    main()
