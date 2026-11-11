
import csv
import json
import random
import statistics
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

from algorithms.astar import astar_search, manhattan_distance


# =========================
# 1. 实验配置
# =========================

ROWS = 20
COLS = 20

START = (0, 0)
GOAL = (19, 19)

DENSITIES = [0.1, 0.2, 0.3]

MAPS_PER_DENSITY = 10
REPEATS = 5
SEED = 2026

RESULT_DIR = PROJECT_ROOT / "results" / "q1"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

DETAIL_FILE = RESULT_DIR / "q1_density_detail.csv"
SUMMARY_FILE = RESULT_DIR / "q1_density_summary.csv"


# =========================
# 2. 地图生成
# =========================

def generate_maze(density, rng):
    grid = []

    for r in range(ROWS):
        row = []

        for c in range(COLS):
            if rng.random() < density:
                row.append("X")
            else:
                row.append(".")

        grid.append(row)

    grid[0][0] = "S"
    grid[19][19] = "E"

    return grid


# =========================
# 3. 单张地图实验
# =========================

def evaluate_map(grid, heuristic):

    stats_list = []
    path_length = None

    for _ in range(REPEATS):

        path, stats = astar_search(
            START,
            GOAL,
            grid,
            heuristic=heuristic
        )

        if not path:
            raise RuntimeError("可达地图实验中出现无解结果")

        stats_list.append(stats)
        path_length = len(path) - 1

    return {
        "search_nodes": statistics.mean(
            s["search_nodes"] for s in stats_list
        ),
        "expanded_nodes": statistics.mean(
            s["expanded_nodes"] for s in stats_list
        ),
        "generated_nodes": statistics.mean(
            s["generated_nodes"] for s in stats_list
        ),
        "runtime_ms": statistics.mean(
            s["runtime_ms"] for s in stats_list
        ),
        "path_length": path_length
    }


# =========================
# 4. 执行完整实验
# =========================

def run_experiment():

    rng = random.Random(SEED)

    detail_rows = []
    summary_rows = []

    print("===== A* 不同障碍密度拓展实验 =====")
    print(f"随机种子：{SEED}")
    print(f"地图尺寸：{ROWS}x{COLS}")
    print(f"每种密度目标地图数：{MAPS_PER_DENSITY}")
    print(f"每张地图重复次数：{REPEATS}")

    methods = {
        "h=0": lambda p, g: 0,
        "Manhattan": manhattan_distance
    }

    for density in DENSITIES:

        records = []
        unreachable = 0
        attempts = 0

        while len(records) < MAPS_PER_DENSITY:

            attempts += 1

            if attempts > 10000:
                raise RuntimeError("生成可达地图数量不足")

            grid = generate_maze(density, rng)

            # 使用 h=0 判断地图是否有解
            baseline_path, _ = astar_search(
                START,
                GOAL,
                grid,
                heuristic=lambda p, g: 0
            )

            if not baseline_path:
                unreachable += 1
                continue

            map_id = len(records) + 1

            obstacle_count = sum(
                cell == "X"
                for row in grid
                for cell in row
            )

            record = {
                "map_id": map_id,
                "grid": grid,
                "methods": {}
            }

            for name, heuristic in methods.items():

                result = evaluate_map(grid, heuristic)

                record["methods"][name] = result

                detail_rows.append({
                    "density": density,
                    "map_id": map_id,
                    "attempts": attempts,
                    "obstacle_count": obstacle_count,
                    "method": name,
                    "search_nodes": result["search_nodes"],
                    "expanded_nodes": result["expanded_nodes"],
                    "generated_nodes": result["generated_nodes"],
                    "path_length": result["path_length"],
                    "runtime_ms": result["runtime_ms"],
                    "map_layout": json.dumps(
                        grid,
                        ensure_ascii=False
                    )
                })

            # 两种方法必须找到相同的最短路径长度
            assert (
                record["methods"]["h=0"]["path_length"]
                == record["methods"]["Manhattan"]["path_length"]
            )

            records.append(record)

        # =========================
        # 5. 汇总数据
        # =========================

        print(f"\n===== 障碍密度 {density:.0%} =====")
        print(f"尝试生成：{attempts}")
        print(f"可达地图：{len(records)}")
        print(f"无解地图：{unreachable}")

        for name in methods:

            metric_names = [
                "search_nodes",
                "expanded_nodes",
                "generated_nodes",
                "path_length",
                "runtime_ms"
            ]

            summary = {}

            for metric in metric_names:

                values = [
                    r["methods"][name][metric]
                    for r in records
                ]

                summary[metric] = statistics.mean(values)

            summary_rows.append({
                "density": density,
                "method": name,
                "attempts": attempts,
                "reachable_maps": len(records),
                "unreachable_maps": unreachable,
                "unreachable_rate": unreachable / attempts,
                **summary
            })

            print(f"\n方法：{name}")

            for metric, value in summary.items():
                print(f"{metric}: {value:.4f}")

    # =========================
    # 6. 保存详细数据
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
    # 7. 保存汇总数据
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

    print("\n===== 文件保存完成 =====")
    print(f"详细数据：{DETAIL_FILE}")
    print(f"汇总数据：{SUMMARY_FILE}")
    print(f"详细记录数：{len(detail_rows)}")
    print(f"汇总记录数：{len(summary_rows)}")


if __name__ == "__main__":
    run_experiment()
