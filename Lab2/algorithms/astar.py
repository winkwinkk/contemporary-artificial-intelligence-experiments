
import heapq
import itertools
import time


def manhattan_distance(pos, goal):
    return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])


def astar_search(start, goal, grid, heuristic=None):
    """
    A*搜索算法。

    参数：
        start: 起点坐标 (row, col)
        goal: 终点坐标 (row, col)
        grid: 二维地图，X表示障碍物
        heuristic: 启发函数，默认使用零启发函数（h=0）

    返回：
        path: 从起点到终点的坐标列表，无解返回 []
        stats: 搜索过程统计信息
    """

    if heuristic is None:
        heuristic = lambda pos, target: 0

    rows = len(grid)
    cols = len(grid[0])

    # 检查坐标是否合法
    def in_bounds(pos):
        r, c = pos
        return 0 <= r < rows and 0 <= c < cols

    if not in_bounds(start) or not in_bounds(goal):
        return [], {}

    if grid[start[0]][start[1]] == 'X':
        return [], {}

    if grid[goal[0]][goal[1]] == 'X':
        return [], {}

    # 起点和终点相同
    if start == goal:
        return [start], {
            "search_nodes": 0,
            "expanded_nodes": 0,
            "generated_nodes": 1,
            "runtime_ms": 0.0
        }

    # 四个移动方向：上、右、下、左
    directions = [
        (-1, 0),
        (0, 1),
        (1, 0),
        (0, -1)
    ]

    # 优先队列：(f, 序号, g, 当前坐标)
    counter = itertools.count()
    open_heap = []

    start_h = heuristic(start, goal)

    heapq.heappush(
        open_heap,
        (start_h, next(counter), 0, start)
    )

    # 记录到达每个节点的最优已知代价
    g_score = {start: 0}

    # 记录路径来源
    came_from = {}

    search_nodes = 0
    expanded_nodes = 0
    generated_nodes = 1

    begin_time = time.perf_counter()

    while open_heap:

        f, _, current_g, current = heapq.heappop(open_heap)

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

            return path, {
                "search_nodes": search_nodes,
                "expanded_nodes": expanded_nodes,
                "generated_nodes": generated_nodes,
                "runtime_ms": runtime_ms
            }

        expanded_nodes += 1

        # 扩展当前节点的四个邻居
        for dr, dc in directions:
            neighbor = (current[0] + dr, current[1] + dc)

            if not in_bounds(neighbor):
                continue

            nr, nc = neighbor

            if grid[nr][nc] == 'X':
                continue

            tentative_g = current_g + 1

            if tentative_g < g_score.get(neighbor, float('inf')):

                is_new = neighbor not in g_score

                g_score[neighbor] = tentative_g
                came_from[neighbor] = current

                h = heuristic(neighbor, goal)
                new_f = tentative_g + h

                heapq.heappush(
                    open_heap,
                    (new_f, next(counter), tentative_g, neighbor)
                )

                if is_new:
                    generated_nodes += 1

    runtime_ms = (
        time.perf_counter() - begin_time
    ) * 1000

    return [], {
        "search_nodes": search_nodes,
        "expanded_nodes": expanded_nodes,
        "generated_nodes": generated_nodes,
        "runtime_ms": runtime_ms
    }


def find_path(start, goal, grid):
    """题目一正式接口：返回最短路径"""
    path, _ = astar_search(
        start,
        goal,
        grid,
        heuristic=manhattan_distance
    )
    return path

