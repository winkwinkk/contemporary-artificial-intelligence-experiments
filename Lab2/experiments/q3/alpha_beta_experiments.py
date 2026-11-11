
import time


def get_children(node):
    """读取节点的子节点，兼容 TreeNode 和字典。"""
    if isinstance(node, dict):
        return node.get("children", [])
    return node.children


def get_value(node):
    """读取叶子节点的评价值。"""
    if isinstance(node, dict):
        return node.get("value")
    return node.value


def minimax_value(node, is_maximizing=True):
    """
    计算节点的真实 Minimax 值。

    仅用于 Oracle 排序。
    该函数的计算时间由调用方单独统计。
    """

    children = get_children(node)

    if not children:
        value = get_value(node)

        if value is None:
            raise ValueError("叶子节点必须提供 value")

        return value

    values = [
        minimax_value(child, not is_maximizing)
        for child in children
    ]

    if is_maximizing:
        return max(values)

    return min(values)


def order_children(
    children,
    is_maximizing,
    strategy,
    stats=None
):
    """
    根据指定策略调整子节点访问顺序。

    fifo:
        保持原始顺序。

    reverse:
        反向访问。

    oracle:
        MAX 优先访问真实 Minimax 值较大的节点；
        MIN 优先访问真实 Minimax 值较小的节点。

    Oracle 的 Minimax 预计算及排序时间
    单独累计到 oracle_precompute_ms。
    """

    if strategy == "fifo":
        return list(children)

    if strategy == "reverse":
        return list(reversed(children))

    if strategy == "oracle":

        precompute_begin = time.perf_counter()

        scored = []

        for index, child in enumerate(children):

            child_value = minimax_value(
                child,
                not is_maximizing
            )

            if is_maximizing:
                key = -child_value
            else:
                key = child_value

            scored.append((key, index, child))

        scored.sort(key=lambda item: (item[0], item[1]))

        ordered = [
            item[2] for item in scored
        ]

        precompute_elapsed = (
            time.perf_counter() - precompute_begin
        ) * 1000

        if stats is not None:
            stats["oracle_precompute_ms"] += precompute_elapsed

        return ordered

    raise ValueError(f"未知排序策略：{strategy}")


def alpha_beta_ordered_search(root, strategy="fifo"):
    """
    带节点排序策略的 Alpha-Beta 实验算法。

    参数：
        root: 博弈树根节点
        strategy: fifo / reverse / oracle

    返回：
        best_value: 最佳博弈值
        stats: 搜索统计信息

    时间指标：
        search_runtime_ms:
            不包含 Oracle 预计算的正式搜索时间。

        oracle_precompute_ms:
            Oracle 真实 Minimax 值计算和排序时间。

        total_runtime_ms:
            整个算法流程的总耗时。

        runtime_ms:
            兼容旧代码，等于 total_runtime_ms。
    """

    stats = {
        "visited_nodes": 0,
        "expanded_nodes": 0,
        "leaf_evaluations": 0,
        "pruned_branches": 0,
        "cutoffs": 0,

        "search_runtime_ms": 0.0,
        "oracle_precompute_ms": 0.0,
        "total_runtime_ms": 0.0,
        "runtime_ms": 0.0
    }

    begin_time = time.perf_counter()

    def search(node, alpha, beta, is_maximizing):

        stats["visited_nodes"] += 1

        children = get_children(node)

        # 叶子节点
        if not children:

            value = get_value(node)

            if value is None:
                raise ValueError("叶子节点必须提供 value")

            stats["leaf_evaluations"] += 1

            return value

        stats["expanded_nodes"] += 1

        # 根据策略调整访问顺序
        ordered = order_children(
            children,
            is_maximizing,
            strategy,
            stats
        )

        if is_maximizing:

            best_value = float("-inf")

            for index, child in enumerate(ordered):

                child_value = search(
                    child,
                    alpha,
                    beta,
                    False
                )

                best_value = max(
                    best_value,
                    child_value
                )

                alpha = max(alpha, best_value)

                if alpha >= beta:

                    remaining = len(ordered) - index - 1

                    if remaining > 0:
                        stats["cutoffs"] += 1
                        stats["pruned_branches"] += remaining

                    break

            return best_value

        else:

            best_value = float("inf")

            for index, child in enumerate(ordered):

                child_value = search(
                    child,
                    alpha,
                    beta,
                    True
                )

                best_value = min(
                    best_value,
                    child_value
                )

                beta = min(beta, best_value)

                if alpha >= beta:

                    remaining = len(ordered) - index - 1

                    if remaining > 0:
                        stats["cutoffs"] += 1
                        stats["pruned_branches"] += remaining

                    break

            return best_value

    best_value = search(
        root,
        float("-inf"),
        float("inf"),
        True
    )

    # 完整算法流程耗时
    total_runtime = (
        time.perf_counter() - begin_time
    ) * 1000

    stats["total_runtime_ms"] = total_runtime

    # Oracle 预计算时间从总耗时中分离
    stats["search_runtime_ms"] = max(
        0.0,
        total_runtime - stats["oracle_precompute_ms"]
    )

    # 兼容原有 runtime_ms 字段
    stats["runtime_ms"] = total_runtime

    return best_value, stats
