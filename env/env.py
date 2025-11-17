# env/make_map_30.py
import sys
import os
import numpy as np

# 프로젝트 루트 디렉토리를 sys.path에 추가
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from planning.cpp import CoveragePlanner, HeuristicType


def make_map_30(seed: int | None = 0, size: int = 30):
    """
    size x size 크기의 랜덤 맵을 생성하고 CPP 알고리즘으로 웨이포인트를 생성함.
    - 1: 장애물
    - 2: 시작 위치
    - 0: 빈 공간
    반환값: (grid, waypoints, start_rc)
    """
    if seed is None:
        # 실행마다 다른 결과를 원할 때
        rng = np.random.default_rng()
    else:
        rng = np.random.default_rng(seed)

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
    import argparse
    parser = argparse.ArgumentParser(description="Generate grid/waypoints .npy via CoveragePlanner")
    parser.add_argument("--seed", type=int, default=None, help="Random seed (None => different each run)")
    parser.add_argument("--size", type=int, default=30, help="Map size (size x size)")
    parser.add_argument("--out-grid", type=str, default="map_grid.npy", help="Output npy for grid")
    parser.add_argument("--out-wps", type=str, default="waypoints.npy", help="Output npy for waypoints")
    parser.add_argument("--show", action="store_true", help="Show plot after saving")
    args = parser.parse_args()

    grid, waypoints, start = make_map_30(seed=args.seed, size=args.size)
    np.save(args.out_grid, grid)
    np.save(args.out_wps, waypoints)
    print(f"[DONE] Saved {args.out_grid} and {args.out_wps} (seed={args.seed})")

    if args.show:
        import matplotlib.pyplot as plt
        plt.imshow(grid, cmap="gray_r", origin="lower")
        if len(waypoints) > 0:
            wx, wy = zip(*waypoints)
            plt.plot(wx, wy, color='red', linewidth=1)
        plt.scatter(start[1], start[0], c='blue', marker='o')
        plt.title(f"CPP Map & Path (size={args.size}, seed={args.seed})")
        plt.show()