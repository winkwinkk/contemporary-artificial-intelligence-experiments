
from algorithms.alpha_beta import (
    TreeNode,
    minimax,
    alpha_beta,
    minimax_search,
    alpha_beta_search
)


# =====================================
# 1. 构造标准博弈树
# =====================================

# 叶子节点
leaf1 = TreeNode(value=3)
leaf2 = TreeNode(value=5)
leaf3 = TreeNode(value=6)
leaf4 = TreeNode(value=9)
leaf5 = TreeNode(value=1)
leaf6 = TreeNode(value=2)

# MAX 层
max1 = TreeNode(children=[leaf1, leaf2])
max2 = TreeNode(children=[leaf3, leaf4])
max3 = TreeNode(children=[leaf5, leaf6])

# MIN 层
min1 = TreeNode(children=[max1, max2])
min2 = TreeNode(children=[max2, max3])

# 根节点 MAX
root = TreeNode(children=[min1, min2])


# =====================================
# 2. Minimax 测试
# =====================================

print("===== Minimax 标准测试 =====")

value_minimax, stats_minimax = minimax_search(root)

print("最佳博弈值：", value_minimax)

assert value_minimax == 5

print("Minimax 正确性：PASS")


# =====================================
# 3. Alpha-Beta 测试
# =====================================

print("\n===== Alpha-Beta 标准测试 =====")

value_ab, stats_ab = alpha_beta_search(root)

print("最佳博弈值：", value_ab)

assert value_ab == 5

print("Alpha-Beta 正确性：PASS")


# =====================================
# 4. 两种算法结果对照
# =====================================

print("\n===== 算法结果对照 =====")

assert value_minimax == value_ab

assert minimax(root) == 5
assert alpha_beta(root) == 5

print("Minimax 与 Alpha-Beta 结果一致：PASS")


# =====================================
# 5. 搜索过程统计
# =====================================

print("\n===== 搜索性能对照 =====")

for name, stats in [
    ("Minimax", stats_minimax),
    ("Alpha-Beta", stats_ab)
]:

    print(f"\n{name}")

    print("访问节点数：", stats["visited_nodes"])
    print("扩展节点数：", stats["expanded_nodes"])
    print("叶节点评估数：", stats["leaf_evaluations"])
    print("剪枝分支数：", stats["pruned_branches"])
    print("剪枝发生次数：", stats["cutoffs"])
    print("运行时间(ms)：", stats["runtime_ms"])


# =====================================
# 6. 最终验证
# =====================================

assert stats_ab["pruned_branches"] > 0

print("\n===== 所有题目三基础测试通过！=====")
