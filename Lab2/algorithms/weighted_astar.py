
import heapq
import itertools
import time


def weighted_manhattan_distance(pos, goal):
    """
    加权地形启发函数。

    考虑终点 E 的进入代价为 0，
    其他可通行格子的进入代价至少为 1。

    h(n) = max(0, Manhattan(n, goal) - 1)
    """
    distance = (
        abs(pos[0] - goal[0])
        + abs(pos[1] - goal[1])
    )

    return max(0, distance - 1)


def weighted_astar_search(
    start,
    goal,
    grid,
    heuristic=None,
    tie_break="fifo"
):
    """
    加权 A* 搜索算法。

    参数：
        start: 起点坐标 (row, col)
        goal: 终点坐标 (row, col)
        grid: 加权地形地图
        heuristic: 启发函数
        tie_break:
            fifo：相同 f 值按照插入顺序处理
            lower_h：相同 f 值优先选择较小 h

    返回：
        path: 最优路径，无解返回 []
        cost: 最小总代价，无解返回 -1
        stats: 搜索过程统计信息
    """

    if heuristic is None:
        heuristic = weighted_manhattan_distance

    if tie_break not in ("fifo", "lower_h"):
        raise ValueError(
            "tie_break 只能为 'fifo' 或 'lower_h'"
        )

    rows = len(grid)

    if rows == 0:
        return [], -1, {}

    cols = len(grid[0])

    if cols == 0 or any(len(row) != cols for row in grid):
        return [], -1, {}

    def in_bounds(pos):
        r, c = pos
        return 0 <= r < rows and 0 <= c < cols

    def enter_cost(pos):
        """计算进入目标格子的代价"""

        cell = grid[pos[0]][pos[1]]

        if cell == 'X':
            return float('inf')

        if cell == 'E' or cell == 'S':
            return 0

        return int(cell)

    # 检查起点和终点
    if not in_bounds(start) or not in_bounds(goal):
        return [], -1, {}

    if enter_cost(start) == float('inf'):
        return [], -1, {}

    if enter_cost(goal) == float('inf'):
        return [], -1, {}

    # 起点与终点相同
    if start == goal:
        return [start], 0, {
            "search_nodes": 0,
            "expanded_nodes": 0,
            "generated_nodes": 1,
            "runtime_ms": 0.0
        }

    # 上、右、下、左
    directions = [
        (-1, 0),
        (0, 1),
        (1, 0),
        (0, -1)
    ]

    # 优先队列：
    # (f, tie_value, 序号, g, 坐标)
    counter = itertools.count()
    open_heap = []

    start_g = 0
    start_h = heuristic(start, goal)
    start_f = start_g + start_h

    start_tie = (
        start_h if tie_break == "lower_h" else 0
    )

    heapq.heappush(
        open_heap,
        (
            start_f,
            start_tie,
            next(counter),
            start_g,
            start
        )
    )

    # 当前已知最优路径代价
    g_score = {start: 0}

    # 路径来源记录
    came_from = {}

    search_nodes = 0
    expanded_nodes = 0
    generated_nodes = 1

    begin_time = time.perf_counter()

    while open_heap:

        f, _, _, current_g, current = heapq.heappop(
            open_heap
        )

        # 跳过已经过期的队列记录
        if current_g != g_score.get(current):
            continue

        search_nodes += 1

        # 到达终点，恢复路径
        if current == goal:

            path = [goal]

            while path[-1] != start:
                path.append(came_from[path[-1]])

            path.reverse()

            runtime_ms = (
                time.perf_counter() - begin_time
            ) * 1000

            return path, current_g, {
                "search_nodes": search_nodes,
                "expanded_nodes": expanded_nodes,
                "generated_nodes": generated_nodes,
                "runtime_ms": runtime_ms
            }

        expanded_nodes += 1

        # 扩展四个方向
        for dr, dc in directions:

            neighbor = (
                current[0] + dr,
                current[1] + dc
            )

            if not in_bounds(neighbor):
                continue

            step_cost = enter_cost(neighbor)

            if step_cost == float('inf'):
                continue

            # 进入邻居格子的实际代价
            tentative_g = current_g + step_cost

            if tentative_g < g_score.get(
                neighbor, float('inf')
            ):

                is_new = neighbor not in g_score

                g_score[neighbor] = tentative_g
                came_from[neighbor] = current

                h = heuristic(neighbor, goal)
                new_f = tentative_g + h

                # 同 f 值节点的排序策略
                tie_value = (
                    h if tie_break == "lower_h" else 0
                )

                heapq.heappush(
                    open_heap,
                    (
                        new_f,
                        tie_value,
                        next(counter),
                        tentative_g,
                        neighbor
                    )
                )

                if is_new:
                    generated_nodes += 1

    runtime_ms = (
        time.perf_counter() - begin_time
    ) * 1000

    return [], -1, {
        "search_nodes": search_nodes,
        "expanded_nodes": expanded_nodes,
        "generated_nodes": generated_nodes,
        "runtime_ms": runtime_ms
    }


def find_minimum_cost(start, goal, grid):
    """
    题目二正式接口。

    返回从起点到终点的最小总代价。
    """

    _, cost, _ = weighted_astar_search(
        start,
        goal,
        grid,
        heuristic=weighted_manhattan_distance
    )

    return cost
