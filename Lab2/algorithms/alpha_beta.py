
import time


class TreeNode:
    """
    博弈树节点。

    value: 叶子节点的评价值
    children: 子节点列表
    is_maximizing_player: 当前节点是否属于 MAX 玩家
    """

    def __init__(
        self,
        value=None,
        children=None,
        is_maximizing_player=None
    ):
        self.value = value
        self.children = children if children is not None else []
        self.is_maximizing_player = is_maximizing_player


def get_node_info(node, inherited_role):

    if isinstance(node, dict):
        children = node.get("children", [])
        value = node.get("value")
        role = node.get("is_maximizing_player")
    else:
        children = node.children
        value = node.value
        role = node.is_maximizing_player

    if role is None:
        role = inherited_role

    return children, value, role


def minimax_search(root):
    """
    Minimax 极大极小搜索。

    根节点默认为 MAX。
    子节点按照从左到右的顺序访问。

    返回：
        best_value: 最佳博弈值
        stats: 搜索统计信息
    """

    stats = {
        "visited_nodes": 0,
        "expanded_nodes": 0,
        "leaf_evaluations": 0,
        "pruned_branches": 0,
        "cutoffs": 0,
        "runtime_ms": 0.0
    }

    begin_time = time.perf_counter()

    def search(node, inherited_role):

        stats["visited_nodes"] += 1

        children, value, is_max = get_node_info(
            node, inherited_role
        )

        # 叶子节点直接返回评价值
        if not children:
            if value is None:
                raise ValueError("叶子节点必须提供 value")

            stats["leaf_evaluations"] += 1
            return value

        stats["expanded_nodes"] += 1

        # MAX 节点选择最大值
        if is_max:
            best_value = float("-inf")

            for child in children:
                child_value = search(child, not is_max)
                best_value = max(best_value, child_value)

            return best_value

        # MIN 节点选择最小值
        else:
            best_value = float("inf")

            for child in children:
                child_value = search(child, not is_max)
                best_value = min(best_value, child_value)

            return best_value

    best_value = search(root, True)

    stats["runtime_ms"] = (
        time.perf_counter() - begin_time
    ) * 1000

    return best_value, stats


def alpha_beta_search(root):
    """
    Alpha-Beta 剪枝搜索。

    根节点默认为 MAX。
    子节点按照从左到右的顺序访问。

    alpha: MAX 当前已知的下界
    beta: MIN 当前已知的上界

    当 alpha >= beta 时停止搜索当前节点的剩余子分支。

    返回：
        best_value: 最佳博弈值
        stats: 搜索统计信息
    """

    stats = {
        "visited_nodes": 0,
        "expanded_nodes": 0,
        "leaf_evaluations": 0,
        "pruned_branches": 0,
        "cutoffs": 0,
        "runtime_ms": 0.0
    }

    begin_time = time.perf_counter()

    def search(
        node,
        alpha,
        beta,
        inherited_role
    ):

        stats["visited_nodes"] += 1

        children, value, is_max = get_node_info(
            node, inherited_role
        )

        # 叶子节点
        if not children:
            if value is None:
                raise ValueError("叶子节点必须提供 value")

            stats["leaf_evaluations"] += 1
            return value

        stats["expanded_nodes"] += 1

        if is_max:

            best_value = float("-inf")

            for index, child in enumerate(children):

                child_value = search(
                    child,
                    alpha,
                    beta,
                    not is_max
                )

                best_value = max(
                    best_value,
                    child_value
                )

                alpha = max(alpha, best_value)

                # 剪枝条件
                if alpha >= beta:

                    remaining = len(children) - index - 1

                    if remaining > 0:
                        stats["cutoffs"] += 1
                        stats["pruned_branches"] += remaining

                    break
                
            return best_value

        else:

            best_value = float("inf")

            for index, child in enumerate(children):

                child_value = search(
                    child,
                    alpha,
                    beta,
                    not is_max
                )

                best_value = min(
                    best_value,
                    child_value
                )

                beta = min(beta, best_value)

                # 剪枝条件
                if alpha >= beta:

                    remaining = len(children) - index - 1

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

    stats["runtime_ms"] = (
        time.perf_counter() - begin_time
    ) * 1000

    return best_value, stats


def minimax(root):
    """Minimax 正式接口，仅返回最佳博弈值。"""
    value, _ = minimax_search(root)
    return value


def alpha_beta(root):
    """Alpha-Beta 正式接口，仅返回最佳博弈值。"""
    value, _ = alpha_beta_search(root)
    return value
