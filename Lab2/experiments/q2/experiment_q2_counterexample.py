
import random
import json
import time
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

# 第一轮先进行10000次搜索，不盲目执行50万次
MAX_TRIALS = 500000

SEED = 20261101

PROGRESS_INTERVAL = 500

START = (0, 0)
GOAL = (ROWS - 1, COLS - 1)

RESULT_DIR = PROJECT_ROOT / "results" / "q2"
RESULT_DIR.mkdir(parents=True, exist_ok=True)


# =========================
# 2. 启发函数
# =========================

def ordinary_manhattan(pos, goal):
    """普通曼哈顿距离"""

    return (
        abs(pos[0] - goal[0])
        + abs(pos[1] - goal[1])
    )


def zero_heuristic(pos, goal):
    """h=0，即 Dijkstra"""

    return 0


# =========================
# 3. 随机地图生成
# =========================

def generate_terrain(rng):
    """生成随机加权地形"""

    grid = []

    for r in range(ROWS):
        row = []

        for c in range(COLS):

            if (r, c) == START:
                row.append('S')

            elif (r, c) == GOAL:
                row.append('E')

            else:
                row.append(rng.choice([1, 2, 3, 8]))

        grid.append(row)

    return grid


# =========================
# 4. 保存反例
# =========================

def save_counterexample(trial, terrain, results):

    output = {
        "seed": SEED,
        "trial": trial,
        "rows": ROWS,
        "cols": COLS,
        "start": START,
        "goal": GOAL,
        "terrain": terrain,
        "results": results
    }

    output_file = RESULT_DIR / "q2_counterexample.json"

    with open(output_file, "w", encoding="utf-8") as f:

        json.dump(
            output,
            f,
            ensure_ascii=False,
            indent=4
        )

    print("\n反例数据已保存：", output_file)


# =========================
# 5. 打印算法结果
# =========================

def print_result(name, path, cost, stats):

    print(f"\n{name}")

    print("路径：", path)
    print("总代价：", cost)
    print("搜索节点：", stats["search_nodes"])
    print("扩展节点：", stats["expanded_nodes"])
    print("生成节点：", stats["generated_nodes"])
    print("运行时间(ms)：", stats["runtime_ms"])


# =========================
# 6. 执行反例搜索
# =========================

def main():

    rng = random.Random(SEED)

    begin_time = time.perf_counter()

    print("===== 普通曼哈顿距离反例搜索 =====")
    print("随机种子：", SEED)
    print("地图尺寸：", f"{ROWS}x{COLS}")
    print("最大尝试次数：", MAX_TRIALS)
    print("地形权值：1、2、3、8")
    print("-" * 45)

    found = False

    for trial in range(1, MAX_TRIALS + 1):

        terrain = generate_terrain(rng)

        # -------------------------
        # Dijkstra：最优代价参照
        # -------------------------

        path_d, cost_d, stats_d = weighted_astar_search(
            START,
            GOAL,
            terrain,
            heuristic=zero_heuristic
        )

        # -------------------------
        # 普通曼哈顿距离
        # -------------------------

        path_o, cost_o, stats_o = weighted_astar_search(
            START,
            GOAL,
            terrain,
            heuristic=ordinary_manhattan
        )

        # -------------------------
        # 判断是否找到反例
        # -------------------------

        if cost_o > cost_d:

            found = True

            print("\n" + "=" * 45)
            print("发现反例！")
            print("尝试次数：", trial)

            print("\n地图：")

            for row in terrain:
                print(row)

            print_result(
                "Dijkstra",
                path_d,
                cost_d,
                stats_d
            )

            print_result(
                "普通曼哈顿距离",
                path_o,
                cost_o,
                stats_o
            )

            # 只有找到反例后才运行修正曼哈顿距离
            path_c, cost_c, stats_c = weighted_astar_search(
                START,
                GOAL,
                terrain,
                heuristic=weighted_manhattan_distance
            )

            print_result(
                "修正曼哈顿距离",
                path_c,
                cost_c,
                stats_c
            )

            # 验证修正启发函数仍然找到最优解
            assert cost_c == cost_d

            print("\n反例验证：PASS")

            save_counterexample(
                trial,
                terrain,
                {
                    "Dijkstra": {
                        "path": path_d,
                        "cost": cost_d,
                        "stats": stats_d
                    },
                    "ordinary_manhattan": {
                        "path": path_o,
                        "cost": cost_o,
                        "stats": stats_o
                    },
                    "corrected_manhattan": {
                        "path": path_c,
                        "cost": cost_c,
                        "stats": stats_c
                    }
                }
            )

            break

        # -------------------------
        # 定期输出进度
        # -------------------------

        if trial % PROGRESS_INTERVAL == 0:

            elapsed = time.perf_counter() - begin_time

            print(
                f"已尝试：{trial}/{MAX_TRIALS} | "
                f"耗时：{elapsed:.2f}s | "
                f"当前：未发现反例",
                flush=True
            )

    # =========================
    # 7. 最终实验总结
    # =========================

    total_time = time.perf_counter() - begin_time

    print("\n" + "=" * 45)
    print("===== 实验结束 =====")

    print("实际尝试次数：", trial)
    print("总耗时：", f"{total_time:.2f}s")

    if found:
        print("实验结论：发现普通曼哈顿距离非最优反例")
    else:
        print("实验结论：本轮未发现反例")
        print("注意：未发现反例不代表普通曼哈顿距离具有最优性保证。")


if __name__ == "__main__":
    main()
