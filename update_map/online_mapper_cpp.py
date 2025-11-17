# update_map/online_mapper_cpp.py
# -*- coding: utf-8 -*-
"""
Online Mapper (CPP + 온라인 맵 업데이트 데모; LiDAR 가시선 센싱 버전)
- 외부 맵 파일(map_grid.npy)을 로드해 시작
- 16빔 LiDAR로 센싱: 각 빔은 첫 장애물까지만 관측(가림 적용)
- 알려진 free/obstacle만 갱신; unknown은 기본적으로 통과 허용(탐사형) 옵션
- CPP 목표는 sweep(지그재그) 타깃; A*(4-방향)로 타깃 연결
- 경로 막힘/소진/타임아웃 시 리플랜
"""

import os
import argparse
from typing import List, Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt
import heapq
import math

# =========================
# 유틸: 맵 로드
# =========================
def load_map_grid(path: str = "map_grid.npy") -> np.ndarray:
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} 가 없습니다.")
    grid = np.load(path)
    if grid.ndim != 2:
        raise ValueError("map_grid.npy는 2D 배열이어야 합니다.")
    grid = grid.astype(int, copy=False)

    # 시작지점(2) 없으면 첫 free를 2로 설정
    if (grid == 2).sum() == 0:
        ys, xs = np.where(grid == 0)
        if len(ys) == 0:
            raise RuntimeError("free 셀이 없습니다. map_grid를 확인하세요.")
        grid[ys[0], xs[0]] = 2
    return grid

# =========================
# A* (4-connected)
# =========================
def astar_4(start: Tuple[int, int], goal: Tuple[int, int], trav: np.ndarray) -> Optional[List[Tuple[int, int]]]:
    H, W = trav.shape
    sy, sx = start
    gy, gx = goal
    if not (0 <= sy < H and 0 <= sx < W and 0 <= gy < H and 0 <= gx < W):
        return None
    if not trav[sy, sx] or not trav[gy, gx]:
        return None

    def h(y, x):  # Manhattan
        return abs(y - gy) + abs(x - gx)

    pq = [(h(sy, sx), 0, sy, sx)]
    came = {(sy, sx): None}
    gscore = {(sy, sx): 0}

    while pq:
        _, g, y, x = heapq.heappop(pq)
        if (y, x) == (gy, gx):
            path = []
            cur = (y, x)
            while cur is not None:
                path.append(cur)
                cur = came[cur]
            return path[::-1]
        for dy, dx in [(-1,0),(1,0),(0,-1),(0,1)]:
            ny, nx = y+dy, x+dx
            if not (0 <= ny < H and 0 <= nx < W): continue
            if not trav[ny, nx]: continue
            ng = g + 1
            if ng < gscore.get((ny, nx), 1e18):
                gscore[(ny, nx)] = ng
                came[(ny, nx)] = (y, x)
                heapq.heappush(pq, (ng + h(ny, nx), ng, ny, nx))
    return None

# =========================
# Bresenham for ray grid traversal
# =========================
def bresenham_cells(y0:int, x0:int, y1:int, x1:int) -> List[Tuple[int,int]]:
    """(y0,x0)→(y1,x1) inclusive grid cells along a line."""
    cells = []
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    y, x = y0, x0
    while True:
        cells.append((y, x))
        if x == x1 and y == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x += sx
        if e2 <= dx:
            err += dx
            y += sy
    return cells

# =========================
# CPP 타깃 생성(지그재그)
# =========================
def sweep_targets(trav_mask: np.ndarray, visited: np.ndarray) -> List[Tuple[int, int]]:
    H, W = trav_mask.shape
    targets: List[Tuple[int, int]] = []
    for y in range(0, H):
        rng = range(0, W) if (y % 2 == 0) else range(W-1, -1, -1)
        for x in rng:
            if trav_mask[y, x] and not visited[y, x]:
                targets.append((y, x))
    return targets

# =========================
# 에이전트
# =========================
class OnlineMapperAgent:
    def __init__(self,
                 base_grid: np.ndarray,
                 sense_num_rays: int = 16,
                 sense_fov_deg: float = 360.0,
                 sense_max_range_cells: int = 16,
                 unknown_is_blocked: bool = False,
                 robot_radius_cells: int = 0,
                 replan_timeout: int = 200,
                 draw_rays: bool = True):
        """
        base_grid: 0 free, 1 obstacle, 2 start
        LiDAR:
          - sense_num_rays: 빔 개수(기본 16)
          - sense_fov_deg: 시야각(기본 360도)
          - sense_max_range_cells: 최대 사거리(셀)
          - unknown_is_blocked: True면 관측 안된 셀로는 계획/이동 금지(보수적)
        """
        self.grid = base_grid.copy().astype(int)
        self.H, self.W = self.grid.shape

        ys, xs = np.where(self.grid == 2)
        if len(ys) == 0:
            raise RuntimeError("시작 위치(2) 없음")
        self.pos = (int(ys[0]), int(xs[0]))  # (y,x)

        # 관측 상태
        self.known_obst = np.zeros_like(self.grid, dtype=bool)
        self.known_free = np.zeros_like(self.grid, dtype=bool)
        sy, sx = self.pos
        self.known_free[sy, sx] = True  # 시작셀은 free

        self.visited = np.zeros_like(self.grid, dtype=bool)
        self.visited[sy, sx] = True

        # LiDAR 설정
        self.num_rays = int(sense_num_rays)
        self.fov_deg = float(sense_fov_deg)
        self.max_range = int(sense_max_range_cells)
        self.unknown_is_blocked = bool(unknown_is_blocked)

        # 플래닝/이동
        self.robot_radius = int(robot_radius_cells)
        self.current_path: List[Tuple[int,int]] = []
        self.target_list: List[Tuple[int,int]] = []
        self.target_idx = 0
        self.steps_since_replan = 0
        self.replan_timeout = int(replan_timeout)

        # 시각화용
        self.draw_rays = bool(draw_rays)
        self._last_rays_segments: List[Tuple[Tuple[int,int], Tuple[int,int]]] = []

        # 최초 센싱 후 리플랜
        self.sense_update()
        self.replan("init(no-obs)")

    # ---------- 장애물 팽창(선택) ----------
    def _inflate_obstacles(self, obst: np.ndarray) -> np.ndarray:
        if self.robot_radius <= 0:
            return obst.copy()
        H, W = obst.shape
        out = obst.copy()
        coords = np.argwhere(obst)
        r = self.robot_radius
        for (y, x) in coords:
            y0 = max(0, y - r); y1 = min(H, y + r + 1)
            x0 = max(0, x - r); x1 = min(W, x + r + 1)
            out[y0:y1, x0:x1] |= True
        return out

    # ---------- LiDAR 가시선 센싱 ----------
    def sense_update(self):
        """빔별로 첫 장애물까지만 관측. 레이 경로는 known_free, 첫 hit은 known_obst."""
        y0, x0 = self.pos
        self._last_rays_segments = []

        # 360도 기준 시작각(라디안). fov로 제한 가능
        if self.fov_deg >= 360 - 1e-6:
            angles = [2*math.pi * i / self.num_rays for i in range(self.num_rays)]
        else:
            fov = math.radians(self.fov_deg)
            start = -fov/2.0
            angles = [start + fov * i / (self.num_rays - 1) for i in range(self.num_rays)]

        H, W = self.H, self.W

        for th in angles:
            # 그리드(y+,x+) 좌표계: dx=cos, dy=sin
            dx = math.cos(th)
            dy = math.sin(th)

            # 최대 사거리 끝점(그리드 내부로 클램프)
            x1 = int(round(x0 + dx * self.max_range))
            y1 = int(round(y0 + dy * self.max_range))
            x1 = max(0, min(W-1, x1))
            y1 = max(0, min(H-1, y1))

            # Bresenham 셀 나열
            cells = bresenham_cells(y0, x0, y1, x1)
            hit = None
            path_cells = []

            for (yy, xx) in cells[1:]:  # 시작셀 제외
                if self.grid[yy, xx] == 1:
                    hit = (yy, xx)
                    break
                else:
                    self.known_free[yy, xx] = True
                    path_cells.append((yy, xx))

            if hit is not None:
                hy, hx = hit
                self.known_obst[hy, hx] = True
                self._last_rays_segments.append(((y0, x0), (hy, hx)))
            else:
                # 히트 없으면 레이 끝까지 free
                if path_cells:
                    self._last_rays_segments.append(((y0, x0), path_cells[-1]))

    # ---------- 통과 가능 마스크 ----------
    def traversable_mask(self) -> np.ndarray:
        """통과 가능(bool). unknown 처리 정책에 따라 달라짐."""
        obst_inf = self._inflate_obstacles(self.known_obst)
        if self.unknown_is_blocked:
            # 본 적 있는 free만 통과 (보수적)
            trav = self.known_free & (~obst_inf)
        else:
            # 알려진 장애물만 막고 unknown은 통과 허용(탐사형)
            trav = ~obst_inf
        return trav

    # ---------- CPP 타깃/플랜 ----------
    def _build_targets(self, trav: np.ndarray) -> List[Tuple[int,int]]:
        return sweep_targets(trav, self.visited)

    def _connect_with_astar(self, pts: List[Tuple[int,int]], trav: np.ndarray) -> List[Tuple[int,int]]:
        out: List[Tuple[int,int]] = []
        cur = self.pos
        for tgt in pts:
            path = astar_4(cur, tgt, trav)
            if path is None or len(path) == 0:
                continue
            if out and path[0] == out[-1]:
                out.extend(path[1:])
            else:
                out.extend(path)
            cur = tgt
        return out

    def replan(self, reason: str):
        trav = self.traversable_mask()
        targets = self._build_targets(trav)
        self.target_list = targets
        self.target_idx = 0
        if len(targets) == 0:
            self.current_path = []
            print(f"[REPLAN] {reason} | targets=0 (done)")
            return
        full_path = self._connect_with_astar(targets, trav)
        self.current_path = full_path
        self.steps_since_replan = 0
        print(f"[REPLAN] {reason} | targets={len(targets)} | path_len={len(full_path)}")

    # ---------- 스텝 ----------
    def step_once(self) -> dict:
        # 1) 센싱(가림 적용)
        self.sense_update()

        # 2) 경로 상태 점검 & 리플랜 결정
        need_replan = False
        reason = ""
        trav = self.traversable_mask()

        if self.steps_since_replan >= self.replan_timeout:
            need_replan, reason = True, "timeout"

        if not need_replan and len(self.current_path) == 0:
            need_replan, reason = True, "path_empty"

        if not need_replan and len(self.current_path) > 0:
            if self.current_path[0] != self.pos:
                self.current_path.insert(0, self.pos)
            if len(self.current_path) >= 2:
                ny, nx = self.current_path[1]
                # 알려진 통과 가능 + 물리 맵 free 둘 다 만족해야 실제 전진 가능
                if (not trav[ny, nx]) or (self.grid[ny, nx] == 1):
                    need_replan, reason = True, "next_blocked"

        if need_replan:
            self.replan(reason)

        # 3) 이동
        moved = False
        if len(self.current_path) >= 2:
            ny, nx = self.current_path[1]
            # 이동 전에도 물리 충돌 체크(unknown이라도 실제 맵은 뚫을 수 없음)
            if trav[ny, nx] and (self.grid[ny, nx] != 1):
                self.pos = (ny, nx)
                self.current_path.pop(0)
                moved = True

        y, x = self.pos
        self.visited[y, x] = True
        self.known_free[y, x] = True

        self.steps_since_replan += 1

        info = {
            "pos": self.pos,
            "moved": moved,
            "path_remain": len(self.current_path),
            "targets_remain": int((~self.visited & self.traversable_mask()).sum()),
        }
        return info

# =========================
# 시각화 데모
# =========================
def run_demo(map_path: str,
             start_corner: Optional[str] = None,  # None이면 맵의 2(시작) 사용
             seed: int = 0,
             max_steps: int = 4000,
             show: bool = True,
             start_paused: bool = False,
             num_rays: int = 16,
             fov_deg: float = 360.0,
             max_range: int = 16,
             unknown_blocked: bool = False,
             draw_rays: bool = True):
    np.random.seed(seed)

    base_grid = load_map_grid(map_path)
    H, W = base_grid.shape

    # 선택적으로 시작 모서리 강제
    if start_corner:
        corners = {
            "tl": (0, 0), "tr": (0, W-1),
            "bl": (H-1, 0), "br": (H-1, W-1),
        }
        sel = corners.get(start_corner.lower(), (0, 0))
        base_grid[base_grid == 2] = 0
        base_grid[sel[0], sel[1]] = 2
        print(f"[MAP] start corner={start_corner} -> {sel}")

    print(f"[MAP] '{map_path}' shape={base_grid.shape} (0 free, 1 obst, 2 start)")

    agent = OnlineMapperAgent(
        base_grid=base_grid,
        sense_num_rays=num_rays,
        sense_fov_deg=fov_deg,
        sense_max_range_cells=max_range,
        unknown_is_blocked=unknown_blocked,
        robot_radius_cells=0,
        replan_timeout=200,
        draw_rays=draw_rays,
    )

    if not show:
        for _ in range(max_steps):
            info = agent.step_once()
            if info["targets_remain"] == 0:
                print("[DONE] coverage complete.")
                break
        return

    # 인터랙션
    plt.ion()
    fig, ax = plt.subplots(figsize=(8, 8))
    state = {"paused": bool(start_paused), "step": False, "quit": False}

    def on_key(event):
        k = (event.key or "").lower()
        if k in (" ", "space", "p"):
            state["paused"] = not state["paused"]
        elif k in ("n", "right"):
            state["step"] = True
        elif k in ("q", "escape"):
            state["quit"] = True

    fig.canvas.mpl_connect('key_press_event', on_key)

    step = 0
    info = {"pos": agent.pos, "moved": False,
            "path_remain": len(agent.current_path),
            "targets_remain": int((~agent.visited & agent.traversable_mask()).sum())}

    while step < max_steps:
        if state["quit"]:
            break
        if (not state["paused"]) or state["step"]:
            step += 1
            info = agent.step_once()
            state["step"] = False

        ax.clear()
        ax.imshow(
            base_grid, cmap="Greys", origin="upper",
            vmin=0, vmax=2, extent=[-0.5, W-0.5, H-0.5, -0.5],
            interpolation="nearest",
        )

        # 관측 free/obst 시각화
        fy, fx = np.where(agent.known_free)
        if len(fy) > 0:
            ax.scatter(fx, fy, s=6, c="#C8EFFF", marker="s", label="Known Free", alpha=0.8)
        oy, ox = np.where(agent.known_obst)
        if len(oy) > 0:
            ax.scatter(ox, oy, s=10, c="#333333", marker="s", label="Known Obst", alpha=0.9)

        # 현재 경로
        if len(agent.current_path) > 1:
            py = [p[0] for p in agent.current_path]
            px = [p[1] for p in agent.current_path]
            ax.plot(px, py, "g-", lw=1.5, alpha=0.8, label="Current Path")

        # 라이다 레이(선택)
        if agent.draw_rays and agent._last_rays_segments:
            for (p0, p1) in agent._last_rays_segments:
                (y0, x0), (y1, x1) = p0, p1
                ax.plot([x0, x1], [y0, y1], lw=0.8, alpha=0.5)

        # 로봇
        ry, rx = agent.pos
        ax.scatter([rx], [ry], s=80, c="orange", edgecolors="k", linewidths=1.0, label="Robot")

        ax.set_xlim(-0.5, W-0.5); ax.set_ylim(H-0.5, -0.5)
        ax.set_aspect("equal")
        title = f"LiDAR Online CPP | step={step} remain={info['targets_remain']}  "
        title += f"[rays={agent.num_rays}, range={agent.max_range}, unknown_blocked={agent.unknown_is_blocked}]"
        if state["paused"]:
            title += "  [PAUSED] (space/p toggle, n step, q quit)"
        ax.set_title(title)
        ax.legend(loc="upper right", fontsize=8)
        plt.pause(0.02)

        if info["targets_remain"] == 0:
            ax.set_title(f"Coverage Complete at step={step}")
            plt.pause(0.5)
            break

    plt.ioff()
    plt.show()

# =========================
# main
# =========================
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--map", type=str, default="map_grid.npy")
    p.add_argument("--start_corner", type=str, default=None, help="tl|tr|bl|br (지정 시 맵의 2를 무시하고 모서리로 시작)")
    p.add_argument("--max_steps", type=int, default=4000)
    p.add_argument("--pause", action="store_true")
    p.add_argument("--no_show", action="store_true")
    p.add_argument("--num_rays", type=int, default=16)
    p.add_argument("--fov_deg", type=float, default=360.0)
    p.add_argument("--max_range", type=int, default=16)
    p.add_argument("--unknown_blocked", action="store_true", help="관측 안된 영역 통과금지(보수적 탐사)")
    p.add_argument("--no_rays", action="store_true", help="레이 시각화 끄기")
    args = p.parse_args()

    run_demo(
        map_path=args.map,
        start_corner=args.start_corner,
        seed=0,
        max_steps=args.max_steps,
        show=not args.no_show,
        start_paused=args.pause,
        num_rays=args.num_rays,
        fov_deg=args.fov_deg,
        max_range=args.max_range,
        unknown_blocked=args.unknown_blocked,
        draw_rays=(not args.no_rays),
    )

if __name__ == "__main__":
    main()