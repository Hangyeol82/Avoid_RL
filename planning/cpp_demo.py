# cpp_demo.py  —  실행용 데모 (화살표 시각화 + 메인 포함)
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D

# 네가 저장해둔 CoveragePlanner 모듈
from cpp import CoveragePlanner, HeuristicType, PlannerStatus


# ----------------- 화살표 기반 CPP 시각화 -----------------
def plot_cpp_arrows(
    target_map,
    trajectory,
    title="Coverage Path Planning (arrows)",
    save_path=None,
    a_star_offset=0.18,   # A* 구간 살짝 평행 이동(겹침 방지)
    arrow_width=0.08,
    head_length=0.18,
    inset_s=0.18,         # 화살표 시작을 셀 중앙에서 dir*inset만큼 이동
    inset_e=0.82,         # 화살표 끝을 셀 중앙에서 dir*inset_e까지 (양 끝을 살짝 잘라줌)
):
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    from matplotlib.lines import Line2D
    from cpp import PlannerStatus

    movement = [[-1,0],[0,-1],[1,0],[0,1]]    # (drow, dcol)
    action   = [-1, 0, 1, 2]                  # R # L B

    start_position_color    = 'gold'
    start_orientation_color = 'deeppink'
    status_color_ref = {
        PlannerStatus.STANDBY:                 'black',
        PlannerStatus.COVERAGE_SEARCH:         'royalblue',
        PlannerStatus.NEARST_UNVISITED_SEARCH: 'darkturquoise',
        PlannerStatus.FOUND:                   'mediumseagreen',
        PlannerStatus.NOT_FOUND:               'red'
    }

    cmap = mpl.colors.ListedColormap([
        'w','k', start_position_color,
        status_color_ref[PlannerStatus.FOUND],
        status_color_ref[PlannerStatus.NOT_FOUND],
    ])
    norm = mpl.colors.BoundaryNorm([0,1,2,3,4,5], cmap.N)
    status_to_cmap_pos = {PlannerStatus.FOUND:3, PlannerStatus.NOT_FOUND:4}

    H, W = target_map.shape
    m = target_map.copy()
    last = trajectory[-1]
    if last[6] in status_to_cmap_pos:
        m[last[1], last[2]] = status_to_cmap_pos[last[6]]

    fig, ax = plt.subplots(figsize=(7,7))
    ax.imshow(m, origin='lower', cmap=cmap, norm=norm,
              interpolation='nearest', extent=[0, W, 0, H])
    ax.set_title(title); ax.set_xlabel("X"); ax.set_ylabel("Y")
    ax.set_xlim(0, W); ax.set_ylim(0, H)

    # 셀 중앙 좌표: (col+0.5, row+0.5)
    for i in range(len(trajectory)-1):
        _, r, c, o, _, next_action, state = trajectory[i]
        if next_action is None: 
            continue
        mov_idx = (o + action[next_action]) % 4
        drow, dcol = movement[mov_idx]

        # 셀 중앙
        cx = c + 0.5
        cy = r + 0.5

        # A* 구간이면 평행 이동(겹침 줄이기)
        offx, offy = 0.0, 0.0
        if state == PlannerStatus.NEARST_UNVISITED_SEARCH and a_star_offset > 0:
            if mov_idx % 2:  # 좌우 이동이면 세로로 살짝
                offy = -a_star_offset
            else:            # 상하 이동이면 가로로 살짝
                offx =  a_star_offset

        # 양 끝을 잘라주는 inset 적용
        # 시작: 중앙 + dir*inset_s, 끝: 중앙 + dir*inset_e
        sx = cx + offx + dcol * inset_s
        sy = cy + offy + drow * inset_s
        ex = cx + offx + dcol * inset_e
        ey = cy + offy + drow * inset_e

        ax.arrow(sx, sy, ex - sx, ey - sy,
                 width=arrow_width,
                 color=status_color_ref.get(state, 'gray'),
                 length_includes_head=True, head_length=head_length)

    # 시작 orientation 화살표도 중앙 기준으로
    ir, ic = trajectory[0][1], trajectory[0][2]
    io = trajectory[0][3]
    vrow, vcol = movement[io]
    cx0, cy0 = ic + 0.5, ir + 0.5
    sx0 = cx0 + vcol * (inset_s*0.6)
    sy0 = cy0 + vrow * (inset_s*0.6)
    ex0 = cx0 + vcol * (inset_e*0.6)
    ey0 = cy0 + vrow * (inset_e*0.6)
    ax.arrow(sx0, sy0, ex0 - sx0, ey0 - sy0,
             width=arrow_width*0.9, color=start_orientation_color,
             length_includes_head=True, head_length=head_length*0.8)

    legend_elements = [
        Line2D([0],[0], color=status_color_ref[PlannerStatus.COVERAGE_SEARCH], lw=1, marker='>',
               markerfacecolor=status_color_ref[PlannerStatus.COVERAGE_SEARCH], label='Coverage Search'),
        Line2D([0],[0], color=status_color_ref[PlannerStatus.NEARST_UNVISITED_SEARCH], lw=1, marker='>',
               markerfacecolor=status_color_ref[PlannerStatus.NEARST_UNVISITED_SEARCH], label='A* to Nearest Unvisited'),
        Line2D([0],[0], color='w', lw=1, marker='>', markerfacecolor='deeppink', label='Start Orientation'),
        Line2D([0],[0], marker='s', color='w', label='Start', markerfacecolor='gold', markersize=12),
        Line2D([0],[0], marker='s', color='w', label='Obstacle', markerfacecolor='k', markersize=12),
    ]
    ax.legend(handles=legend_elements, bbox_to_anchor=(1.02, 1.0), loc='upper left')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=160, bbox_inches='tight')
    plt.show()


# ----------------- 간단 랜덤 맵 생성기 -----------------
def make_simple_grid(size=30, seed=None):
    """
    0=빈칸, 1=장애물(블록), 2=시작점
    장애물 몇 개를 블록으로 뿌리고, 시작점은 좌하단 쪽 첫 빈칸.
    """
    rng = np.random.default_rng(seed)
    g = np.zeros((size, size), dtype=int)

    # 블록 장애물 2~4개
    for _ in range(rng.integers(2, 5)):
        h = rng.integers(3, 6)
        w = rng.integers(3, 6)
        r0 = rng.integers(2, size - h - 2)
        c0 = rng.integers(2, size - w - 2)
        g[r0:r0+h, c0:c0+w] = 1

    # 시작점 (좌하단부터 탐색)
    start = None
    for r in range(size):
        for c in range(size):
            if g[r, c] == 0:
                g[r, c] = 2
                start = (r, c)
                break
        if start: break
    if start is None:
        raise RuntimeError("빈칸이 없습니다. 장애물이 너무 많음.")
    return g


# ----------------- 메인 -----------------
def main():
    import argparse
    parser = argparse.ArgumentParser(description="CoveragePlanner Demo (arrow viz)")
    parser.add_argument("--size", type=int, default=30)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--cp", type=str, default="VERTICAL",
                        choices=["VERTICAL","HORIZONTAL","MANHATTAN","CHEBYSHEV"])
    parser.add_argument("--orient", type=int, default=0, choices=[0,1,2,3],
                        help="초기 orientation (0:up, 1:left, 2:down, 3:right)")
    parser.add_argument("--save", type=str, default=None, help="PNG 저장 경로")
    args = parser.parse_args()

    grid = make_simple_grid(size=args.size, seed=args.seed)

    # 플래너 실행
    cp = CoveragePlanner(grid)
    cp.start(initial_orientation=args.orient,
             cp_heuristic=getattr(HeuristicType, args.cp))
    cp.compute()

    found, steps, cost, traj, xy = cp.result()
    print(f"found={found}, steps={steps}, cost={cost:.2f}")

    # 화살표 시각화
    plot_cpp_arrows(grid, traj, title=f"CPP Demo (steps={steps})", save_path=args.save)


if __name__ == "__main__":
    main()