
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

DETAIL_FILE = RESULT_DIR / "q2_terrain_detail.csv"
SUMMARY_FILE = RESULT_DIR / "q2_terrain_summary.csv"


# =========================
# 2. 三种自设地形
# =========================

def generate_uniform_terrain():
    """场景A：均匀低代价地形"""

    grid = [
        [1 for _ in range(COLS)]
        for _ in range(ROWS)
    ]

    grid[0][0] = 'S'
    grid[9][9] = 'E'

    return grid


def generate_high_cost_terrain():
    """场景B：中央集中高代价区域"""

    grid = [
        [1 for _ in range(COLS)]
        for _ in range(ROWS)
    ]

    # 中央设置连续高代价区域
    for r in range(1, 9):
        for c in range(4, 6):
            grid[r][c] = 8

    grid[0][0] = 'S'
    grid[9][9] = 'E'

    return grid


def generate_random_terrain(seed):
    """场景C：随机混合地形"""

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
# 3. 路径验证
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
# 4. 算法配置
# =========================

METHODS = {
    "Dijkstra": lambda p, g: 0,
    "Weighted_AStar": weighted_manhattan_distance
}


# =========================
# 5. 执行单个实验
# =========================

def evaluate(grid, method_name, heuristic):

    results = []

    for _ in range(REPEATS):

        path, cost, stats = weighted_astar_search(
            START,
            GOAL,
            grid,
            heuristic=heuristic
        )

        assert path, "当前地图应该存在可达路径"

        actual_cost = calculate_path_cost(path, grid)

        assert actual_cost == cost, "路径代价验证失败"

        results.append({
            "cost": cost,
            "path_length": len(path) - 1,
            "search_nodes": stats["search_nodes"],
            "expanded_nodes": stats["expanded_nodes"],
            "generated_nodes": stats["generated_nodes"],
            "runtime_ms": stats["runtime_ms"]
        })

    # 同一地图重复运行，最优代价必须保持一致
    assert len(set(r["cost"] for r in results)) == 1

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

    print("===== 题目二：不同地形权值分布实验 =====")
    print(f"地图尺寸：{ROWS}x{COLS}")
    print(f"重复次数：{REPEATS}")
    print(f"随机种子：{SEED}")

    for scenario_name, grid in scenarios.items():

        print(f"\n===== 场景：{scenario_name} =====")

        scenario_results = {}

        for method_name, heuristic in METHODS.items():

            results = evaluate(
                grid,
                method_name,
                heuristic
            )

            scenario_results[method_name] = results

            metrics = [
                "cost",
                "path_length",
                "search_nodes",
                "expanded_nodes",
                "generated_nodes",
                "runtime_ms"
            ]

            averages = {}

            for metric in metrics:

                values = [
                    result[metric]
                    for result in results
                ]

                averages[metric] = statistics.mean(values)

            # 保存详细数据
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
                **averages
            })

            print(f"\n方法：{method_name}")

            for metric, value in averages.items():
                print(f"{metric}: {value:.4f}")

        # 两种算法必须得到相同的最优代价
        dijkstra_cost = scenario_results["Dijkstra"][0]["cost"]

        astar_cost = scenario_results["Weighted_AStar"][0]["cost"]

        assert dijkstra_cost == astar_cost

        print("\n最优代价一致性：PASS")

    # =========================
    # 7. 保存详细数据
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
