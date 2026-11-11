
import time

from algorithms.weighted_astar import (
    weighted_astar_search,
    weighted_manhattan_distance
)


# =====================================
# 1. 地图定义
# =====================================

GRID = [
    [1, 1, 1, 'X', 2, 1, 1],
    [1, 'X', 6, 'X', 1, 'X', 1],
    [1, 1, 1, 1, 1, 1, 1],
    ['X', 1, 'X', 8, 'X', 1, 'X'],
    [1, 1, 1, 1, 1, 1, 1],
    [1, 'X', 9, 'X', 1, 'X', 1],
    [1, 1, 1, 'X', 1, 1, 1]
]


# =====================================
# 2. 玩家和宝藏位置
# =====================================

POSITIONS = {
    "A": (0, 0),
    "B": (6, 6),
    "T1": (0, 6),
    "T2": (2, 2),
    "T3": (4, 4),
    "T4": (6, 0)
}


TREASURE_VALUES = {
    "T1": 9,
    "T2": 6,
    "T3": 7,
    "T4": 7
}


# =====================================
# 3. 计算两个位置之间的最小移动代价
# =====================================

def get_distance(source, target):

    start = POSITIONS[source]
    goal = POSITIONS[target]

    path, cost, stats = weighted_astar_search(
        start,
        goal,
        GRID,
        heuristic=weighted_manhattan_distance
    )

    if not path:
        raise ValueError(
            f"{source} 到 {target} 无法到达"
        )

    return cost


# =====================================
# 4. 计算 6×6 距离矩阵
# =====================================

def calculate_distance_matrix():

    names = list(POSITIONS.keys())

    matrix = {}

    for source in names:

        matrix[source] = {}

        for target in names:

            if source == target:
                cost = 0
            else:
                cost = get_distance(source, target)

            matrix[source][target] = cost

    return matrix


# =====================================
# 5. 博弈状态初始化
# =====================================

def create_initial_state():

    return {
        "pos_a": "A",
        "pos_b": "B",
        "remaining": tuple(TREASURE_VALUES.keys()),
        "score_a": 0,
        "score_b": 0,
        "current_player": "A"
    }


# =====================================
# 6. 状态转移
# =====================================

def apply_action(state, target, distance_matrix):
    """
    当前玩家选择一个宝藏，生成新的博弈状态。
    不直接修改原状态。
    """

    if target not in state["remaining"]:
        raise ValueError("该宝藏已经被获取")

    player = state["current_player"]

    new_state = state.copy()

    # 当前玩家的位置
    if player == "A":
        current_pos = state["pos_a"]
    else:
        current_pos = state["pos_b"]

    # 计算本次行动收益
    distance = distance_matrix[current_pos][target]

    gain = TREASURE_VALUES[target] - distance

    # 更新玩家位置和累计收益
    if player == "A":

        new_state["pos_a"] = target
        new_state["score_a"] += gain
        new_state["current_player"] = "B"

    else:

        new_state["pos_b"] = target
        new_state["score_b"] += gain
        new_state["current_player"] = "A"

    # 删除已经获得的宝藏
    new_state["remaining"] = tuple(
        item for item in state["remaining"]
        if item != target
    )

    return new_state


# =====================================
# 7. 博弈状态评价函数
# =====================================

def evaluate_state(state):

    return state["score_a"] - state["score_b"]


# =====================================
# 8. Alpha-Beta 节点排序策略
# =====================================

def get_action_gain(state, target, distance_matrix):
    """
    计算当前玩家选择某个宝藏的即时收益。
    """

    if state["current_player"] == "A":
        current_pos = state["pos_a"]
    else:
        current_pos = state["pos_b"]

    distance = distance_matrix[current_pos][target]

    return TREASURE_VALUES[target] - distance


def order_actions(
    state,
    distance_matrix,
    strategy="fifo"
):
    """
    根据指定策略调整宝藏访问顺序。

    fifo:
        按照原始宝藏顺序访问。

    reverse:
        按照原始宝藏逆序访问。

    greedy:
        按照即时收益从高到低访问。

    注意：
        Greedy 只使用当前行动的即时收益，
        不提前计算未来博弈结果。
    """

    remaining = list(state["remaining"])

    if strategy == "fifo":
        return remaining

    if strategy == "reverse":
        return list(reversed(remaining))

    if strategy == "greedy":

        return sorted(
            remaining,
            key=lambda target: -get_action_gain(
                state,
                target,
                distance_matrix
            )
        )

    raise ValueError(
        f"未知节点排序策略：{strategy}"
    )


# =====================================
# 9. Minimax 博弈搜索
# =====================================

def minimax_game(state, distance_matrix, depth):
    """
    Minimax 博弈搜索。

    depth: 剩余搜索深度，表示双方累计行动次数。

    返回：
        best_value: 当前状态下的最优博弈值
        best_action: 当前玩家的最优行动
        stats: 搜索统计信息
    """

    if depth < 0:
        raise ValueError("搜索深度不能为负数")

    stats = {
        "visited_nodes": 0,
        "expanded_nodes": 0,
        "leaf_evaluations": 0,
        "runtime_ms": 0.0
    }

    def search(current_state, remaining_depth):

        stats["visited_nodes"] += 1

        remaining = current_state["remaining"]

        # 到达搜索深度或游戏结束
        if remaining_depth == 0 or not remaining:

            stats["leaf_evaluations"] += 1

            return evaluate_state(current_state), None

        stats["expanded_nodes"] += 1

        player = current_state["current_player"]

        # A 为 MAX，B 为 MIN
        is_maximizing = (player == "A")

        if is_maximizing:
            best_value = float("-inf")
        else:
            best_value = float("inf")

        best_action = None

        # 按照剩余宝藏的原始顺序访问
        for target in remaining:

            next_state = apply_action(
                current_state,
                target,
                distance_matrix
            )

            child_value, _ = search(
                next_state,
                remaining_depth - 1
            )

            if is_maximizing:

                if child_value > best_value:
                    best_value = child_value
                    best_action = target

            else:

                if child_value < best_value:
                    best_value = child_value
                    best_action = target

        return best_value, best_action

    begin_time = time.perf_counter()

    best_value, best_action = search(state, depth)

    stats["runtime_ms"] = (
        time.perf_counter() - begin_time
    ) * 1000

    return best_value, best_action, stats


# =====================================
# 10. Alpha-Beta 剪枝博弈搜索
# =====================================

def alpha_beta_game(
    state,
    distance_matrix,
    depth,
    ordering="fifo"
):
    """
    Alpha-Beta 剪枝博弈搜索。

    参数：
        state: 当前博弈状态
        distance_matrix: 关键位置之间的距离矩阵
        depth: 搜索深度
        ordering: 节点排序策略

            fifo: 原始顺序
            reverse: 逆序
            greedy: 即时收益优先

    返回：
        best_value: 最优博弈值
        best_action: 当前玩家的最优行动
        stats: 搜索统计信息
    """

    if depth < 0:
        raise ValueError("搜索深度不能为负数")

    if ordering not in ("fifo", "reverse", "greedy"):
        raise ValueError(
            "ordering 只能为 fifo、reverse 或 greedy"
        )

    stats = {
        "visited_nodes": 0,
        "expanded_nodes": 0,
        "leaf_evaluations": 0,
        "pruned_branches": 0,
        "cutoffs": 0,
        "runtime_ms": 0.0
    }

    def search(
        current_state,
        remaining_depth,
        alpha,
        beta
    ):

        stats["visited_nodes"] += 1

        remaining = current_state["remaining"]

        # 搜索终止条件
        if remaining_depth == 0 or not remaining:

            stats["leaf_evaluations"] += 1

            return evaluate_state(current_state), None

        stats["expanded_nodes"] += 1

        player = current_state["current_player"]

        is_maximizing = (player == "A")

        if is_maximizing:
            best_value = float("-inf")
        else:
            best_value = float("inf")

        best_action = None

        # 根据排序策略调整当前节点的访问顺序
        ordered_targets = order_actions(
            current_state,
            distance_matrix,
            ordering
        )

        for index, target in enumerate(ordered_targets):

            next_state = apply_action(
                current_state,
                target,
                distance_matrix
            )

            child_value, _ = search(
                next_state,
                remaining_depth - 1,
                alpha,
                beta
            )

            if is_maximizing:

                if child_value > best_value:
                    best_value = child_value
                    best_action = target

                alpha = max(alpha, best_value)

            else:

                if child_value < best_value:
                    best_value = child_value
                    best_action = target

                beta = min(beta, best_value)

            # Alpha-Beta 剪枝
            if alpha >= beta:

                remaining_branches = (
                    len(ordered_targets) - index - 1
                )

                if remaining_branches > 0:

                    stats["cutoffs"] += 1

                    stats["pruned_branches"] += (
                        remaining_branches
                    )

                break

        return best_value, best_action

    begin_time = time.perf_counter()

    best_value, best_action = search(
        state,
        depth,
        float("-inf"),
        float("inf")
    )

    stats["runtime_ms"] = (
        time.perf_counter() - begin_time
    ) * 1000

    return best_value, best_action, stats


# =====================================
# 11. 输出距离矩阵
# =====================================

if __name__ == "__main__":

    distance_matrix = calculate_distance_matrix()

    names = list(POSITIONS.keys())

    print("===== 六个关键位置之间的最小代价矩阵 =====")

    print("位置", *names, sep="\t")

    for source in names:

        values = [
            distance_matrix[source][target]
            for target in names
        ]

        print(source, *values, sep="\t")
