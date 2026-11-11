
from algorithms.weighted_astar import weighted_manhattan_distance


terrain = [
    ['S', 1, 1, 8, 1],
    [1, 8, 3, 1, 1],
    [1, 8, 1, 8, 1],
    [1, 1, 1, 8, 1],
    [8, 8, 1, 1, 'E']
]

GOAL = (4, 4)

DIRECTIONS = [
    (-1, 0),
    (0, 1),
    (1, 0),
    (0, -1)
]


def enter_cost(pos):

    cell = terrain[pos[0]][pos[1]]

    if cell == 'X':
        return float('inf')

    if cell == 'S' or cell == 'E':
        return 0

    return int(cell)


def check_consistency(heuristic):

    violations = []

    rows = len(terrain)
    cols = len(terrain[0])

    for r in range(rows):

        for c in range(cols):

            current = (r, c)

            if enter_cost(current) == float('inf'):
                continue

            h_current = heuristic(current, GOAL)

            for dr, dc in DIRECTIONS:

                neighbor = (r + dr, c + dc)

                nr, nc = neighbor

                if not (
                    0 <= nr < rows
                    and 0 <= nc < cols
                ):
                    continue

                cost = enter_cost(neighbor)

                if cost == float('inf'):
                    continue

                h_neighbor = heuristic(neighbor, GOAL)

                # 一致性条件
                if h_current > cost + h_neighbor:

                    violations.append({
                        "current": current,
                        "neighbor": neighbor,
                        "h_current": h_current,
                        "cost": cost,
                        "h_neighbor": h_neighbor
                    })

    return violations


def main():

    print("===== 启发函数一致性验证 =====")

    violations = check_consistency(
        weighted_manhattan_distance
    )

    print("检查地图：5x5")
    print("检查启发函数：修正曼哈顿距离")
    print("违反一致性的边数：", len(violations))

    for item in violations:
        print(item)

    assert len(violations) == 0

    print("\n一致性验证：PASS")


if __name__ == "__main__":
    main()
