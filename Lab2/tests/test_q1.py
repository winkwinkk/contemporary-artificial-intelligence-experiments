
from algorithms.astar import (
    find_path,
    astar_search,
    manhattan_distance
)


maze = [
    ['S', '.', '.', 'X', '.'],
    ['.', 'X', '.', '.', '.'],
    ['.', 'X', '.', 'X', '.'],
    ['.', '.', '.', 'X', 'E'],
    ['X', 'X', '.', '.', '.']
]

start = (0, 0)
goal = (3, 4)


# 1. 测试正式接口
path = find_path(start, goal, maze)

print("===== 正式接口测试 =====")
print("找到的路径：", path)
print("路径坐标数量：", len(path))
print("移动步数：", len(path) - 1 if path else 0)

assert path[0] == start
assert path[-1] == goal
assert len(path) == 8

for i in range(len(path) - 1):
    r1, c1 = path[i]
    r2, c2 = path[i + 1]

    assert abs(r1 - r2) + abs(c1 - c2) == 1
    assert maze[r2][c2] != 'X'

print("标准测试用例：PASS")


# 2. 对照实验：h=0
baseline_path, baseline_stats = astar_search(
    start, goal, maze,
    heuristic=lambda pos, target: 0
)


# 3. 对照实验：曼哈顿距离
astar_path, astar_stats = astar_search(
    start, goal, maze,
    heuristic=manhattan_distance
)


print("\n===== 搜索效率对照 =====")

for name, path, stats in [
    ("h=0", baseline_path, baseline_stats),
    ("曼哈顿距离", astar_path, astar_stats)
]:
    print(f"\n{name}")
    print("路径：", path)
    print("搜索节点数：", stats["search_nodes"])
    print("扩展节点数：", stats["expanded_nodes"])
    print("生成节点数：", stats["generated_nodes"])
    print("运行时间(ms)：", stats["runtime_ms"])


assert len(baseline_path) == len(astar_path)

print("\n对照实验正确性检查：PASS")

print("\n===== 特殊情况测试 =====")

# 1. 起点等于终点
path, stats = astar_search(
    (0, 0), (0, 0),
    [['S', '.'], ['.', '.']],
    heuristic=manhattan_distance
)
assert path == [(0, 0)]
print("起点等于终点：PASS")

# 2. 无法到达
path, stats = astar_search(
    (0, 0), (0, 2),
    [['S', 'X', 'E']],
    heuristic=manhattan_distance
)
assert path == []
print("无解地图：PASS")

# 3. 起点是障碍物
path, stats = astar_search(
    (0, 0), (0, 1),
    [['X', 'E']],
    heuristic=manhattan_distance
)
assert path == []
print("起点障碍：PASS")

# 4. 终点是障碍物
path, stats = astar_search(
    (0, 0), (0, 1),
    [['S', 'X']],
    heuristic=manhattan_distance
)
assert path == []
print("终点障碍：PASS")