# env/make_map_30.py
import sys
import os
import numpy as np

# 프로젝트 루트 디렉토리를 sys.path에 추가
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from planning.cpp import CoveragePlanner, HeuristicType


def make_map_30(seed: int = 0):
    """
    30x30 크기의 랜덤 맵을 생성하고 CPP 알고리즘으로 웨이포인트를 생성함.
    - 1: 장애물
    - 2: 시작 위치
    - 0: 빈 공간
    반환값: (grid, waypoints, start_rc)
    """
    rng = np.random.default_rng(seed)
    size = 30
    g = np.zeros((size, size), dtype=int)

    # 장애물 블록 2~4개 랜덤 생성
    for _ in range(rng.integers(7, 10)):
        h = rng.integers(3, 6)
        w = rng.integers(3, 6)
        r0 = rng.integers(2, size - h - 2)
        c0 = rng.integers(2, size - w - 2)
        g[r0:r0 + h, c0:c0 + w] = 1

    # 시작 위치 (왼쪽 아래에서 첫 번째 빈칸)
    start = None
    for r in range(size):
        for c in range(size):
            if g[r, c] == 0:
                g[r, c] = 2
                start = (r, c)
                break
        if start:
            break

    # CPP 경로 생성
    cp = CoveragePlanner(g)
    cp.start(initial_orientation=0, cp_heuristic=HeuristicType.VERTICAL)
    cp.compute()
    _, _, _, traj, xy = cp.result()

    # (row, col) 웨이포인트만 추출
    waypoints = [(t[2], t[1]) for t in traj]

    return g, np.array(waypoints, dtype=np.int32), start


if __name__ == "__main__":
    grid, waypoints, start = make_map_30(seed=123)
    np.save("map_grid.npy", grid)
    np.save("waypoints.npy", waypoints)
    print("[DONE] Saved map_grid.npy and waypoints.npy")

    # (선택) 시각화
    import matplotlib.pyplot as plt
    plt.imshow(grid, cmap="gray_r", origin="lower")
    wx, wy = zip(*waypoints)
    plt.plot(wx, wy, color='red', linewidth=1)
    plt.scatter(start[1], start[0], c='blue', marker='o')
    plt.title("30x30 CPP Map & Path")
    plt.show()