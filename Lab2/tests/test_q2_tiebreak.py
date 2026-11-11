
from algorithms.weighted_astar import (
    weighted_astar_search,
    weighted_manhattan_distance
)


terrain = [
    [1 for _ in range(10)]
    for _ in range(10)
]

terrain[0][0] = 'S'
terrain[9][9] = 'E'

start = (0, 0)
goal = (9, 9)


# 原始排序
path1, cost1, stats1 = weighted_astar_search(
    start,
    goal,
    terrain,
    heuristic=weighted_manhattan_distance,
    tie_break="fifo"
)


# 优化排序
path2, cost2, stats2 = weighted_astar_search(
    start,
    goal,
    terrain,
    heuristic=weighted_manhattan_distance,
    tie_break="lower_h"
)


print("===== A* 排序优化测试 =====")

print("\n原始 A*：")
print("总代价：", cost1)
print("路径长度：", len(path1) - 1)
print("搜索节点：", stats1["search_nodes"])
print("扩展节点：", stats1["expanded_nodes"])
print("生成节点：", stats1["generated_nodes"])

print("\n优化 A*：")
print("总代价：", cost2)
print("路径长度：", len(path2) - 1)
print("搜索节点：", stats2["search_nodes"])
print("扩展节点：", stats2["expanded_nodes"])
print("生成节点：", stats2["generated_nodes"])

assert cost1 == cost2 == 17

assert stats2["expanded_nodes"] <= stats1["expanded_nodes"]

print("\n排序优化正确性验证：PASS")
