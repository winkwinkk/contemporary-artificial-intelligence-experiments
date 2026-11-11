
from collections import deque

from algorithms.weighted_astar import weighted_manhattan_distance


# =========================
# 1. 定义测试地图
# =========================

terrain = [
    ['S', 1, 1, 8, 1],
    [1, 8, 3, 1, 1],
    [1, 8, 1, 8, 1],
    [1, 1, 1, 8, 1],
    [8, 8, 1, 1, 'E']
]

goal = (4, 4)


# =========================
# 2. 计算进入格子的代价
# =========================

def enter_cost(pos):

    cell = terrain[pos[0]][pos[1]]

    if cell == 'E':
        return 0

    if cell == 'X':
        return float('inf')

    if cell == 'S':
        return 0

    return int(cell)


# =========================
# 3. 使用 Dijkstra 计算真实剩余代价
# =========================

def true_remaining_cost(start):

    if start == goal:
        return 0

    queue = [(0, start)]
    best = {start: 0}

    import heapq

    while queue:

        cost, current = heapq.heappop(queue)

        if cost != best[current]:
            continue

        if current == goal:
            return cost

        for dr, dc in [
            (-1, 0), (0, 1), (1, 0), (0, -1)
        ]:

            nr = current[0] + dr
            nc = current[1] + dc

            if not (
                0 <= nr < len(terrain)
                and 0 <= nc < len(terrain[0])
            ):
                continue

            neighbor = (nr, nc)

            step = enter_cost(neighbor)

            if step == float('inf'):
                continue

            new_cost = cost + step

            if new_cost < best.get(
                neighbor, float('inf')
            ):

                best[neighbor] = new_cost

                heapq.heappush(
                    queue,
                    (new_cost, neighbor)
                )

    return float('inf')


# =========================
# 4. 检查启发函数可采纳性
# =========================

def ordinary_manhattan(pos, target):

    return (
        abs(pos[0] - target[0])
        + abs(pos[1] - target[1])
    )


def main():

    print("===== 启发函数可采纳性验证 =====")

    ordinary_violations = []
    corrected_violations = []

    for r in range(len(terrain)):

        for c in range(len(terrain[0])):

            pos = (r, c)

            if terrain[r][c] == 'X':
                continue

            actual = true_remaining_cost(pos)

            h1 = ordinary_manhattan(pos, goal)

            h2 = weighted_manhattan_distance(
                pos, goal
            )

            if h1 > actual:
                ordinary_violations.append(
                    (pos, h1, actual)
                )

            if h2 > actual:
                corrected_violations.append(
                    (pos, h2, actual)
                )

    print("\n普通曼哈顿距离")

    print("违反可采纳性的位置数量：",
          len(ordinary_violations))

    for item in ordinary_violations:
        print(
            f"位置：{item[0]}，"
            f"h={item[1]}，"
            f"真实剩余代价={item[2]}"
        )

    print("\n修正曼哈顿距离")

    print("违反可采纳性的位置数量：",
          len(corrected_violations))

    assert len(ordinary_violations) > 0
    assert len(corrected_violations) == 0

    print("\n实验结论：PASS")


if __name__ == "__main__":
    main()
