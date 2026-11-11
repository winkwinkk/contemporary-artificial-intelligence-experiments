
from algorithms.weighted_astar import (
    find_minimum_cost,
    weighted_astar_search,
    weighted_manhattan_distance
)


# =========================
# 1. 标准测试地图
# =========================

terrain = [
    ['S', 1, 1, 8, 1],
    [1, 8, 3, 1, 1],
    [1, 8, 1, 8, 1],
    [1, 1, 1, 8, 1],
    [8, 8, 1, 1, 'E']
]

start = (0, 0)
goal = (4, 4)


# =========================
# 2. 路径代价独立验证
# =========================

def calculate_path_cost(path, grid):
    """根据路径重新计算进入各格子的总代价"""

    if not path:
        return -1

    total_cost = 0

    for r, c in path[1:]:
        cell = grid[r][c]

        if cell == 'X':
            raise ValueError("路径经过了障碍物")

        if cell == 'E':
            total_cost += 0
        else:
            total_cost += int(cell)

    return total_cost


def validate_path(path, grid, start, goal):
    """检查路径是否合法"""

    if not path:
        return False

    if path[0] != start or path[-1] != goal:
        return False

    for i in range(len(path) - 1):

        r1, c1 = path[i]
        r2, c2 = path[i + 1]

        # 只能上下左右移动
        if abs(r1 - r2) + abs(c1 - c2) != 1:
            return False

        if grid[r2][c2] == 'X':
            return False

    return True


# =========================
# 3. 测试正式接口
# =========================

print("===== 题目二标准测试 =====")

result = find_minimum_cost(
    start,
    goal,
    terrain
)

print("最小总代价：", result)

assert result == 7

print("标准测试用例：PASS")


# =========================
# 4. 定义三种启发函数
# =========================

def zero_heuristic(pos, target):
    """h=0：Dijkstra 基线"""
    return 0


def ordinary_manhattan_distance(pos, target):
    """普通曼哈顿距离"""
    return (
        abs(pos[0] - target[0])
        + abs(pos[1] - target[1])
    )


heuristics = {
    "Dijkstra (h=0)": zero_heuristic,

    "普通曼哈顿距离": ordinary_manhattan_distance,

    "修正曼哈顿距离": weighted_manhattan_distance
}


# =========================
# 5. 三种启发函数对照实验
# =========================

print("\n===== 三种启发函数对照实验 =====")

results = {}

for name, heuristic in heuristics.items():

    path, cost, stats = weighted_astar_search(
        start,
        goal,
        terrain,
        heuristic=heuristic
    )

    # 保存实验结果
    results[name] = {
        "path": path,
        "cost": cost,
        "search_nodes": stats["search_nodes"],
        "expanded_nodes": stats["expanded_nodes"],
        "generated_nodes": stats["generated_nodes"],
        "runtime_ms": stats["runtime_ms"]
    }

    print(f"\n{name}")
    print("路径：", path)
    print("最小总代价：", cost)
    print("搜索节点数：", stats["search_nodes"])
    print("扩展节点数：", stats["expanded_nodes"])
    print("生成节点数：", stats["generated_nodes"])
    print("运行时间(ms)：", stats["runtime_ms"])


# =========================
# 6. 路径正确性与总代价验证
# =========================

print("\n===== 路径与总代价验证 =====")

for name, result_data in results.items():

    path = result_data["path"]
    cost = result_data["cost"]

    # 检查路径合法性
    assert validate_path(
        path, terrain, start, goal
    ), f"{name} 路径不合法"

    # 独立计算路径总代价
    actual_cost = calculate_path_cost(
        path, terrain
    )

    assert actual_cost == cost, (
        f"{name} 返回代价与实际路径代价不一致"
    )

    print(f"{name}：")
    print("路径合法性：PASS")
    print("实际路径代价：", actual_cost)
    print("算法返回代价：", cost)


# =========================
# 7. 最优代价核对
# =========================

# Dijkstra 作为最优代价参照
optimal_cost = results["Dijkstra (h=0)"]["cost"]

assert optimal_cost == 7

# 修正曼哈顿距离应当保持最优
assert results["修正曼哈顿距离"]["cost"] == optimal_cost

print("\n===== 最优代价核对 =====")

print("Dijkstra 最优代价：", optimal_cost)

print(
    "普通曼哈顿距离代价：",
    results["普通曼哈顿距离"]["cost"]
)

print(
    "修正曼哈顿距离代价：",
    results["修正曼哈顿距离"]["cost"]
)

print("修正曼哈顿距离最优性验证：PASS")

# =========================
# 8. 搜索性能汇总
# =========================

print("\n===== 搜索性能汇总 =====")

print(
    f"{'方法':<22}"
    f"{'搜索节点':>10}"
    f"{'扩展节点':>10}"
    f"{'生成节点':>10}"
    f"{'总代价':>10}"
)

for name, data in results.items():

    print(
        f"{name:<22}"
        f"{data['search_nodes']:>10}"
        f"{data['expanded_nodes']:>10}"
        f"{data['generated_nodes']:>10}"
        f"{data['cost']:>10}"
    )


print("\n所有题目二基础及对照测试完成！")
