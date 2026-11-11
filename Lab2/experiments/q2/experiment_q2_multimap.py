
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

MAP_COUNT = 30
REPEATS = 5
BASE_SEED = 20261101

RESULT_DIR = PROJECT_ROOT / "results" / "q2"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

DETAIL_FILE = RESULT_DIR / "q2_multimap_detail.csv"
MAP_FILE = RESULT_DIR / "q2_multimap_by_map.csv"
SUMMARY_FILE = RESULT_DIR / "q2_multimap_summary.csv"


# =========================
# 2. 算法配置
# =========================

METHODS = [
    ("Dijkstra", lambda p, g: 0, "fifo"),
    ("AStar_FIFO", weighted_manhattan_distance, "fifo"),
    ("AStar_LowerH", weighted_manhattan_distance, "lower_h")
]


# =========================
# 3. 随机地图生成
# =========================

def generate_terrain(seed):

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

        if cell == 'X':
            raise ValueError("路径经过障碍物")

        total += int(cell)

    return total


# =========================
# 5. 单次算法运行
# =========================

def run_once(grid, heuristic, tie_break):

    path, cost, stats = weighted_astar_search(
        START,
        GOAL,
        grid,
        heuristic=heuristic,
        tie_break=tie_break
    )

    assert path, "未找到可行路径"

    actual_cost = calculate_path_cost(path, grid)

    assert actual_cost == cost, "路径代价验证失败"

    return {
        "cost": cost,
        "path_length": len(path) - 1,
        "search_nodes": stats["search_nodes"],
        "expanded_nodes": stats["expanded_nodes"],
        "generated_nodes": stats["generated_nodes"],
        "runtime_ms": stats["runtime_ms"]
    }


# =========================
# 6. 主实验
# =========================

def main():

    detail_rows = []
    map_rows = []

    print("===== A* 多随机地图泛化实验 =====")
    print("地图尺寸：10x10")
    print("随机地图数量：", MAP_COUNT)
    print("每张地图重复次数：", REPEATS)
    print("随机种子：", BASE_SEED)
    print("--------------------------------")

    for map_id in range(1, MAP_COUNT + 1):

        seed = BASE_SEED + map_id - 1

        grid = generate_terrain(seed)

        print(f"\n地图 {map_id}/{MAP_COUNT}，seed={seed}")

        map_results = {}

        # 三种算法使用完全相同的地图
        for method_name, heuristic, tie_break in METHODS:

            results = []

            for repeat in range(1, REPEATS + 1):

                result = run_once(
                    grid,
                    heuristic,
                    tie_break
                )

                results.append(result)

                detail_rows.append({
                    "map_id": map_id,
                    "seed": seed,
                    "method": method_name,
                    "repeat": repeat,
                    **result
                })

            map_results[method_name] = results

        # 每张地图三种算法必须获得相同最优代价
        costs = {
            name: results[0]["cost"]
            for name, results in map_results.items()
        }

        assert len(set(costs.values())) == 1

        # 每张地图按算法计算平均指标
        for method_name, results in map_results.items():

            row = {
                "map_id": map_id,
                "seed": seed,
                "method": method_name,
                "optimal_cost": results[0]["cost"]
            }

            for metric in [
                "path_length",
                "search_nodes",
                "expanded_nodes",
                "generated_nodes",
                "runtime_ms"
            ]:

                values = [
                    item[metric]
                    for item in results
                ]

                row[metric + "_mean"] = statistics.mean(values)

                row[metric + "_median"] = statistics.median(values)

            map_rows.append(row)

        print(
            "最优代价：",
            costs["Dijkstra"],
            "| 三种算法一致：PASS"
        )

    # =========================
    # 7. 按地图汇总
    # =========================

    metrics = [
        "optimal_cost",
        "path_length_mean",
        "search_nodes_mean",
        "expanded_nodes_mean",
        "generated_nodes_mean",
        "runtime_ms_mean"
    ]

    summary_rows = []

    for method_name, _, _ in METHODS:

        selected = [
            row for row in map_rows
            if row["method"] == method_name
        ]

        summary = {
            "method": method_name,
            "map_count": MAP_COUNT
        }

        for metric in metrics:

            values = [
                row[metric]
                for row in selected
            ]

            summary[metric + "_mean"] = statistics.mean(values)

            summary[metric + "_median"] = statistics.median(values)

            summary[metric + "_std"] = (
                statistics.stdev(values)
                if len(values) > 1 else 0
            )

        summary_rows.append(summary)

    # =========================
    # 8. 计算每张地图的排序优化收益
    # =========================

    fifo_rows = {
        row["map_id"]: row
        for row in map_rows
        if row["method"] == "AStar_FIFO"
    }

    optimized_rows = {
        row["map_id"]: row
        for row in map_rows
        if row["method"] == "AStar_LowerH"
    }

    reductions = []

    for map_id in range(1, MAP_COUNT + 1):

        original = fifo_rows[map_id]["expanded_nodes_mean"]

        optimized = optimized_rows[map_id]["expanded_nodes_mean"]

        reduction = (
            (original - optimized) / original * 100
            if original > 0 else 0
        )

        reductions.append(reduction)

    # =========================
    # 9. 保存CSV
    # =========================

    def save_csv(filename, rows):

        with open(
            filename,
            "w",
            newline="",
            encoding="utf-8-sig"
        ) as f:

            writer = csv.DictWriter(
                f,
                fieldnames=rows[0].keys()
            )

            writer.writeheader()
            writer.writerows(rows)

    save_csv(DETAIL_FILE, detail_rows)
    save_csv(MAP_FILE, map_rows)
    save_csv(SUMMARY_FILE, summary_rows)

    # =========================
    # 10. 输出最终统计
    # =========================

    print("\n================================")
    print("===== 多随机地图实验完成 =====")

    print("详细运行记录：", len(detail_rows))
    print("地图级汇总记录：", len(map_rows))
    print("算法汇总记录：", len(summary_rows))

    print("\n===== 三种算法总体结果 =====")

    for row in summary_rows:

        print("\n方法：", row["method"])

        print(
            "平均扩展节点数：",
            round(row["expanded_nodes_mean_mean"], 4)
        )

        print(
            "平均运行时间(ms)：",
            round(row["runtime_ms_mean_mean"], 4)
        )

    print("\n===== 排序优化效果 =====")

    print(
        "平均节点减少比例：",
        round(statistics.mean(reductions), 2),
        "%"
    )

    print(
        "中位数节点减少比例：",
        round(statistics.median(reductions), 2),
        "%"
    )

    print(
        "优化有效的地图数量：",
        sum(x > 0 for x in reductions)
    )

    print(
        "优化无变化的地图数量：",
        sum(x == 0 for x in reductions)
    )

    print(
        "优化扩展节点更多的地图数量：",
        sum(x < 0 for x in reductions)
    )

    print("\n结果文件：")
    print(DETAIL_FILE)
    print(MAP_FILE)
    print(SUMMARY_FILE)


if __name__ == "__main__":
    main()
