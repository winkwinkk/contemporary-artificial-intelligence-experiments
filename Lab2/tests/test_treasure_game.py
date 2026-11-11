
from algorithms.treasure_game import (
    calculate_distance_matrix,
    create_initial_state,
    minimax_game,
    alpha_beta_game
)


def run_test():

    distance_matrix = calculate_distance_matrix()

    print("===== 题目四 Minimax 与 Alpha-Beta 测试 =====")

    for depth in [2, 4]:

        print(f"\n===== 搜索深度 H={depth} =====")

        state = create_initial_state()

        # Minimax
        minimax_value, minimax_action, minimax_stats = (
            minimax_game(
                state,
                distance_matrix,
                depth
            )
        )

        print("\nMinimax")
        print("最优博弈值：", minimax_value)
        print("当前玩家最优行动：", minimax_action)
        print("访问节点数：", minimax_stats["visited_nodes"])
        print("扩展节点数：", minimax_stats["expanded_nodes"])
        print("叶节点评估数：", minimax_stats["leaf_evaluations"])
        print("运行时间(ms)：", minimax_stats["runtime_ms"])

        # Alpha-Beta
        ab_value, ab_action, ab_stats = (
            alpha_beta_game(
                state,
                distance_matrix,
                depth
            )
        )

        print("\nAlpha-Beta")
        print("最优博弈值：", ab_value)
        print("当前玩家最优行动：", ab_action)
        print("访问节点数：", ab_stats["visited_nodes"])
        print("扩展节点数：", ab_stats["expanded_nodes"])
        print("叶节点评估数：", ab_stats["leaf_evaluations"])
        print("剪枝分支数：", ab_stats["pruned_branches"])
        print("剪枝发生次数：", ab_stats["cutoffs"])
        print("运行时间(ms)：", ab_stats["runtime_ms"])

        # 正确性检查
        assert minimax_value == ab_value

        print("\n最优博弈值一致性：PASS")

        if depth == 2:
            assert minimax_value == -1
            assert minimax_action == "T2"
            assert ab_action == "T2"

            print("H=2 标准结果验证：PASS")

        if depth == 4:
            assert minimax_value == 3
            assert ab_value == 3
            assert minimax_action == "T3"
            assert ab_action == "T3"

            print("H=4 标准结果验证：PASS")

    print("\n===== 题目四基础测试全部完成！=====")


if __name__ == "__main__":
    run_test()
