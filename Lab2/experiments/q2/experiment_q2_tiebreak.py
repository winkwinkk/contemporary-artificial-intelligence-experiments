
import csv
import random
import statistics
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

from algorithms.weighted_astar import (
    weighted_astar_search,
    weighted_manhattan_distance
)


# =========================
# 1. 实验配置
# =========================

ROWS = 10
COLS = 10

START = (0, 0)
GOAL = (9, 9)

REPEATS = 30
SEED = 20261101

RESULT_DIR = PROJECT_ROOT / "results" / "q2"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

DETAIL_FILE = RESULT_DIR / "q2_tiebreak_detail.csv"
SUMMARY_FILE = RESULT_DIR / "q2_tiebreak_summary.csv"


# =========================
# 2. 生成三种地形
# =========================

def generate_uniform_terrain():

    grid = [
        [1 for _ in range(COLS)]
        for _ in range(ROWS)
    ]

    grid[0][0] = 'S'
    grid[9][9] = 'E'

    return grid


def generate_high_cost_terrain():

    grid = [
        [1 for _ in range(COLS)]
        for _ in range(ROWS)
    ]

    for r in range(1, 9):
        for c in range(4, 6):
            grid[r][c] = 8

    grid[0][0] = 'S'
    grid[9][9] = 'E'

    return grid


def generate_random_terrain(seed):

    rng = random.Random(seed)

    grid = []

    for r in range(ROWS):

        row = []

        for c in range(COLS):
            row.append(rng.choice([1, 3, 8]))

        grid.append(row)

    grid[0][0] = 'S'
    grid[9][9] = 'E'

    return grid


# =========================
# 3. 定义三种算法
# =========================

METHODS = [
    ("Dijkstra", lambda p, g: 0, "fifo"),

    (
        "AStar_FIFO",
        weighted_manhattan_distance,
        "fifo"
    ),

    (
        "AStar_LowerH",
        weighted_manhattan_distance,
        "lower_h"
    )
]


# =========================
# 4. 独立计算路径代价
# =========================

def calculate_path_cost(path, grid):

    if not path:
        return -1

    total = 0

    for r, c in path[1:]:

        cell = grid[r][c]

        if cell == 'E':
            continue

        total += int(cell)

    return total


# =========================
# 5. 单个算法重复测试
# =========================

def evaluate(grid, method_name, heuristic, tie_break):

    results = []

    for repeat in range(1, REPEATS + 1):

        path, cost, stats = weighted_astar_search(
            START,
            GOAL,
            grid,
            heuristic=heuristic,
            tie_break=tie_break
        )

        assert path, f"{method_name} 未找到路径"

        actual_cost = calculate_path_cost(path, grid)

        assert actual_cost == cost

        results.append({
            "cost": cost,
            "path_length": len(path) - 1,
            "search_nodes": stats["search_nodes"],
            "expanded_nodes": stats["expanded_nodes"],
            "generated_nodes": stats["generated_nodes"],
            "runtime_ms": stats["runtime_ms"]
        })

    return results


# =========================
# 6. 主实验
# =========================

def main():

    scenarios = {
        "uniform": generate_uniform_terrain(),
        "high_cost_region": generate_high_cost_terrain(),
        "random_mixed": generate_random_terrain(SEED)
    }

    detail_rows = []
    summary_rows = []

    print("===== A* 节点排序优化对照实验 =====")
    print("地图尺寸：10x10")
    print("每种场景重复次数：30")
    print("随机种子：", SEED)

    for scenario_name, grid in scenarios.items():

        print(f"\n===== 场景：{scenario_name} =====")

        scenario_costs = {}

        for method_name, heuristic, tie_break in METHODS:

            results = evaluate(
                grid,
                method_name,
                heuristic,
                tie_break
            )

            scenario_costs[method_name] = results[0]["cost"]

            metrics = [
                "cost",
                "path_length",
                "search_nodes",
                "expanded_nodes",
                "generated_nodes",
                "runtime_ms"
            ]

            values = {}

            for metric in metrics:

                data = [
                    result[metric]
                    for result in results
                ]

                values[metric + "_mean"] = statistics.mean(data)
                values[metric + "_median"] = statistics.median(data)

            # 保存30次详细记录
            for repeat_id, result in enumerate(results, 1):

                detail_rows.append({
                    "scenario": scenario_name,
                    "method": method_name,
                    "repeat": repeat_id,
                    **result
                })

            # 保存汇总数据
            summary_rows.append({
                "scenario": scenario_name,
                "method": method_name,
                **values
            })

            print(f"\n方法：{method_name}")

            for metric in metrics:

                print(
                    f"{metric}: "
                    f"{values[metric + '_mean']:.4f} "
                    f"(中位数："
                    f"{values[metric + '_median']:.4f})"
                )

        # 三种方法必须得到相同的最优总代价
        assert len(set(scenario_costs.values())) == 1

        print("\n三种算法最优总代价一致：PASS")

    # =========================
    # 7. 保存详细实验数据
    # =========================

    with open(
        DETAIL_FILE,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=detail_rows[0].keys()
        )

        writer.writeheader()
        writer.writerows(detail_rows)

    # =========================
    # 8. 保存汇总数据
    # =========================

    with open(
        SUMMARY_FILE,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=summary_rows[0].keys()
        )

        writer.writeheader()
        writer.writerows(summary_rows)

    print("\n===== 实验全部完成 =====")
    print("详细记录数：", len(detail_rows))
    print("汇总记录数：", len(summary_rows))
    print("详细结果：", DETAIL_FILE)
    print("汇总结果：", SUMMARY_FILE)


if __name__ == "__main__":
    main()
