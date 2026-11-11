import time
import statistics

from algorithms.astar import astar_search, manhattan_distance


maze = [
    ['S', '.', '.', 'X', '.'],
    ['.', 'X', '.', '.', '.'],
    ['.', 'X', '.', 'X', '.'],
    ['.', '.', '.', 'X', 'E'],
    ['X', 'X', '.', '.', '.']
]

start = (0, 0)
goal = (3, 4)

methods = {
    "h=0": lambda p, g: 0,
    "Manhattan": manhattan_distance
}

REPEATS = 30
BATCH_SIZE = 1000

print("===== 重复运行时间实验 =====")

for name, heuristic in methods.items():

    times = []

    for _ in range(REPEATS):

        begin = time.perf_counter()

        for _ in range(BATCH_SIZE):
            astar_search(
                start, goal, maze,
                heuristic=heuristic
            )

        elapsed = time.perf_counter() - begin

        times.append(
            elapsed * 1000 / BATCH_SIZE
        )

    print(f"\n{name}")
    print(f"平均时间：{statistics.mean(times):.6f} ms")
    print(f"中位数：{statistics.median(times):.6f} ms")
    print(f"最小时间：{min(times):.6f} ms")
    print(f"最大时间：{max(times):.6f} ms")