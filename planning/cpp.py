# planning/cpp_serpentine.py
# -*- coding: utf-8 -*-

import numpy as np
import heapq
from enum import Enum, auto


class PlannerStatus(Enum):
    STANDBY = auto()
    COVERAGE_SEARCH = auto()
    NEAREST_FRONTIER_SEARCH = auto()
    FOUND = auto()
    NOT_FOUND = auto()


class HeuristicType(Enum):
    MANHATTAN = auto()
    CHEBYSHEV = auto()
    VERTICAL = auto()
    HORIZONTAL = auto()


class CoveragePlanner:
    """
    지그재그(부스트로페돈) 선호 + 프론티어 점프 결합형 CPP

    - visited: 장애물(1)과 분리된 bool 마스크
    - 커버리지 한 스텝: 현재 위치 주변의 미방문 free 중
        * 지그재그 선호(pref) 가중치로 비용 낮춤
        * (선택) dir_bias * heuristic 으로 미세 편향(기본 0)
    - 프론티어: free & not visited & 4-이웃 중 visited=True 존재
        * BFS로 같은 거리 레벨(layer)에서 찾고,
          그 중 "지그재그 순번(serp_rank)"이 앞서는 타깃을 선택
    - A*: 상태 = (y,x,orientation), 액션 비용(턴/전진) 반영
    - FSM은 네가 준 로직 그대로 반영
    """

    # orientation: 0=up, 1=left, 2=down, 3=right
    MOVES = [(-1, 0),  (0, -1),  (1, 0),  (0, 1)]   # up, left, down, right
    ACTION_DELTA = [-1, 0, 1, 2]  # Right turn, Forward, Left turn, Back(180)
    ACTION_NAME  = ['R', '#', 'L', 'B']
    ACTION_COST  = [0.2, 0.1, 0.2, 0.4]  # (R, F, L, B)

    def __init__(self, map_open: np.ndarray):
        # 맵 규약: 0=free, 1=obstacle, 2=start
        self.map_grid = map_open.astype(int, copy=True)
        self.H, self.W = self.map_grid.shape

        # 시작 위치 (orientation=0 기본)
        self.current_pos = self.get_start_position()
        if self.current_pos is None:
            ys, xs = np.where(self.map_grid == 0)
            if len(ys) == 0:
                raise RuntimeError("free 셀이 없습니다.")
            self.map_grid[ys[0], xs[0]] = 2
            self.current_pos = [int(ys[0]), int(xs[0]), 0]

        # 방문 마스크(장애물과 분리)
        self.visited = np.zeros_like(self.map_grid, dtype=bool)

        # 결과 누적
        # [v,y,x,o,a_in,a_next,state]
        self.current_trajectory = []
        self.current_trajectory_annotations = []

        # FSM
        self.state_ = PlannerStatus.STANDBY

        # 휴리스틱 설정
        self.a_star_heuristic = HeuristicType.MANHATTAN
        self.cp_heuristic = HeuristicType.VERTICAL

        # 지그재그 스윕/가중치
        self.sweep_mode = 'row'   # 'row' = 가로 지그재그, 'col' = 세로 지그재그
        self.serp_weight = 10.0   # 지그재그 선호 가중치(클수록 지그재그에 집착)
        self.dir_bias = 0.0       # 미세 편향(0~0.02 권장). 큼=나선 유발 가능

        self.debug_level = -1
        self._cp_heuristic_map = None  # 수직/수평 고정 그라디언트 캐시

    # ================= 퍼블릭 API =================

    def set_debug_level(self, level: int):
        self.debug_level = int(level)

    def start(self, initial_orientation=0, a_star_heuristic=None, cp_heuristic=None):
        y, x, _ = self.get_start_position(orientation=initial_orientation)
        self.current_pos = [y, x, initial_orientation]
        self.visited[:] = False
        self.current_trajectory.clear()
        self.current_trajectory_annotations.clear()

        if cp_heuristic is not None:
            self.cp_heuristic = cp_heuristic
        if a_star_heuristic is not None:
            self.a_star_heuristic = a_star_heuristic

        # 수직/수평 편향은 (0,0) 앵커 기준 고정 그라디언트 캐시
        if self.cp_heuristic in (HeuristicType.VERTICAL, HeuristicType.HORIZONTAL):
            self._cp_heuristic_map = self._create_heuristic([0, 0], self.cp_heuristic)
        else:
            self._cp_heuristic_map = None

        self.state_ = PlannerStatus.COVERAGE_SEARCH
        self._printd("start", f"start={self.current_pos}")

    def compute(self):
        while self.compute_non_blocking():
            pass
        return self.state_

    def compute_non_blocking(self):
        searching = False
        st = self.state_

        if st == PlannerStatus.COVERAGE_SEARCH:
            # 국소 커버리지 확장(지그재그 강제)
            heuristic = self._cp_heuristic_map if self._cp_heuristic_map is not None \
                        else self._create_heuristic(self.current_pos, self.cp_heuristic)

            ok, traj = self._coverage_search_step(self.current_pos, heuristic, self.dir_bias)

            # 방문/위치 반영
            if traj:
                self._append_trajectory(traj, "CS")
                self.current_pos = [traj[-1][1], traj[-1][2], traj[-1][3]]

            # ---- 네가 준 FSM 로직 적용 ----
            if ok:
                # 아직 커버리지를 계속 진행해야 함. 완료 여부만 체크.
                if self.check_full_coverage():
                    self.state_ = PlannerStatus.FOUND
                    if self.current_trajectory:
                        self.current_trajectory[-1][6] = PlannerStatus.FOUND
                    searching = False
                else:
                    # 더 진행할 수 있으므로 다음 루프 계속
                    self.state_ = PlannerStatus.COVERAGE_SEARCH
                    searching = True
            else:
                # 주변에 미방문 free 없음 → 프론티어로 점프 시도
                self.state_ = PlannerStatus.NEAREST_FRONTIER_SEARCH
                searching = True

        elif st == PlannerStatus.NEAREST_FRONTIER_SEARCH:
            found, goal = self._nearest_frontier_bfs(tuple(self.current_pos[:2]))
            if not found:
                # 더 갈 프론티어가 없으면 커버 완료인지 확인
                if self.check_full_coverage():
                    self.state_ = PlannerStatus.FOUND
                    if self.current_trajectory:
                        self.current_trajectory[-1][6] = PlannerStatus.FOUND
                else:
                    self.state_ = PlannerStatus.NOT_FOUND
                    if self.current_trajectory:
                        self.current_trajectory[-1][6] = PlannerStatus.NOT_FOUND
            else:
                # 방향 인지 A*로 goal까지 연결
                ok2, traj2 = self._astar_oriented(self.current_pos, goal)
                if ok2 and traj2:
                    self._append_trajectory(traj2, "A*")
                    self.current_pos = [traj2[-1][1], traj2[-1][2], traj2[-1][3]]
                    self.state_ = PlannerStatus.COVERAGE_SEARCH
                    searching = True
                else:
                    # 이 프론티어는 막혔거나 경로가 안나옴 → 다음 루프에서 BFS 재시도
                    searching = True

        else:
            self._printd("compute_non_blocking", f"Invalid state: {st}")
            self.state_ = PlannerStatus.NOT_FOUND

        return searching

    def result(self):
        found = self.state_ == PlannerStatus.FOUND
        total_steps = max(0, len(self.current_trajectory) - 1)
        total_cost = self._trajectory_cost(self.current_trajectory)
        xy_trajectory = self._xy_traj(self.current_trajectory)
        return [found, total_steps, total_cost, self.current_trajectory, xy_trajectory]

    def show_results(self):
        self._printd("show_results", f"Final: {self.state_.name}")
        self._printd("show_results", f"Steps: {max(0, len(self.current_trajectory)-1)}")
        self._printd("show_results", f"Cost : {self._trajectory_cost(self.current_trajectory):.2f}")
        if self.debug_level > 0:
            self._print_trajectory(self.current_trajectory)
        self.print_policy_map()

    # ================= 내부 구현 =================

    # ---- 기본 유틸 ----
    def get_start_position(self, orientation=0):
        ys, xs = np.where(self.map_grid == 2)
        if len(ys) == 0:
            return None
        return [int(ys[0]), int(xs[0]), int(orientation)]

    def _is_free(self, y, x):
        return 0 <= y < self.H and 0 <= x < self.W and self.map_grid[y, x] == 0

    def _neighbors4(self, y, x):
        for dy, dx in self.MOVES:
            ny, nx = y + dy, x + dx
            if 0 <= ny < self.H and 0 <= nx < self.W:
                yield ny, nx

    def _printd(self, f, m, level=1):
        if level <= self.debug_level:
            print(f"[{f}] {m}")

    # ---- 트래젝토리/표시 ----
    def _append_trajectory(self, new_traj, tag):
        if not new_traj:
            return
        # 앞뒤 중복 위치 연결
        if self.current_trajectory:
            new_traj[0][4] = self.current_trajectory[-1][4]
            self.current_trajectory_annotations.append([new_traj[0][1], new_traj[0][2], tag])
            self.current_trajectory.pop()
        self.current_trajectory.extend(new_traj)

        # 방문 마킹
        for _, y, x, *_ in new_traj:
            if self._is_free(y, x):
                self.visited[y, x] = True

    def _trajectory_cost(self, traj):
        cost = 0.0
        for t in traj:
            a_next = t[5]
            if a_next is not None:
                cost += self.ACTION_COST[a_next]
        return cost

    def _xy_traj(self, trajectory):
        if not trajectory:
            return []
        return [t[1:3] for t in trajectory]

    def _print_trajectory(self, trajectory):
        print("l_cost\ty\tx\tori\tact_in\ta_next\tstate")
        for t in trajectory:
            print(f"{t[0]:.2f}\t{t[1]}\t{t[2]}\t{t[3]}\t{t[4]}\t{t[5]}\t{t[6].name}")

    def print_map(self, M):
        for r in M:
            print("[" + ",\t".join(f"{v}" if isinstance(v, str) else f"{v:.1f}" for v in r) + "]")

    def print_policy_map(self, trajectory=None, trajectory_annotations=None):
        if trajectory is None:
            trajectory = self.current_trajectory
        if trajectory_annotations is None:
            trajectory_annotations = list(self.current_trajectory_annotations)

        policy = [[" " for _ in range(self.W)] for _ in range(self.H)]
        # 장애물 표기
        for y in range(self.H):
            for x in range(self.W):
                if self.map_grid[y, x] == 1:
                    policy[y][x] = "XXXXXX"

        # 액션 오버레이
        for t in trajectory:
            y, x = t[1], t[2]
            a = t[5]
            if a is not None and 0 <= y < self.H and 0 <= x < self.W:
                policy[y][x] += self.ACTION_NAME[a]

        # 태그
        if trajectory:
            trajectory_annotations = trajectory_annotations + [
                [trajectory[0][1], trajectory[0][2], "STA"],
                [trajectory[-1][1], trajectory[-1][2], "END"],
            ]

        for (y, x, name) in trajectory_annotations:
            if 0 <= y < self.H and 0 <= x < self.W:
                policy[y][x] += f"@{name}"

        self._printd("policy", "Policy Map:")
        if self.debug_level > 0:
            self.print_map(policy)

    # ---- 휴리스틱 ----
    def _create_heuristic(self, target_point, htype: HeuristicType):
        # target_point가 [y,x] 또는 [y,x,o] 모두 허용
        if isinstance(target_point, (list, tuple)):
            ty, tx = target_point[:2]
        else:
            ty, tx = (0, 0)

        H = np.zeros_like(self.map_grid, dtype=float)
        for y in range(self.H):
            for x in range(self.W):
                if htype == HeuristicType.MANHATTAN:
                    H[y, x] = abs(y - ty) + abs(x - tx)
                elif htype == HeuristicType.CHEBYSHEV:
                    H[y, x] = max(abs(y - ty), abs(x - tx))
                elif htype == HeuristicType.HORIZONTAL:
                    H[y, x] = abs(y - ty)
                elif htype == HeuristicType.VERTICAL:
                    H[y, x] = abs(x - tx)
        return H

    # ---- 지그재그 선호/순위 ----
    def _serp_pref(self, y, x, o, ny, nx, o2):
        """
        비용에 더해지는 '선호' 값(음수=좋음). row 스윕 기준:
        - 같은 행에서 원하는 방향으로 전진: -1.0
        - 행을 바꾸는 전이: -0.5
        - 그 외: 0.0
        col 스윕은 행/열을 바꿔서 동일한 논리.
        """
        if self.sweep_mode == 'row':
            desired_o = 3 if (y % 2 == 0) else 1  # 짝수행: →(3), 홀수행: ←(1)
            if ny == y and o2 == desired_o:
                return -1.0
            if ny != y:
                return -0.5
        else:  # 'col'
            desired_o = 2 if (x % 2 == 0) else 0  # 짝수열: ↓(2), 홀수열: ↑(0)
            if nx == x and o2 == desired_o:
                return -1.0
            if nx != x:
                return -0.5
        return 0.0

    def _serp_rank(self, y, x):
        """
        지그재그 순번 (작을수록 먼저 커버해야 하는 순서).
        row 스윕: y행 내부에서 좌→우 / 우→좌 번갈아.
        col 스윕: x열 내부에서 상→하 / 하→상 번갈아.
        """
        if self.sweep_mode == 'row':
            return y * self.W + (x if (y % 2 == 0) else (self.W - 1 - x))
        else:
            return x * self.H + (y if (x % 2 == 0) else (self.H - 1 - y))

    # ---- 커버리지 한 스텝 ----
    def _coverage_search_step(self, initial_pos, heuristic_map, dir_bias=0.0):
        """
        현재 위치에서 '한 구간' 전진.
        - 미방문 free만 후보
        - 비용 = action_cost + dir_bias*heuristic + serp_weight*pref
          (pref는 음수일수록 선호)
        """
        y, x, o = initial_pos
        if not self._is_free(y, x):
            return False, []

        # 현 위치 방문
        self.visited[y, x] = True
        traj = [[0.0, y, x, o, None, None, self.state_]]

        # 후보 모으기
        cand = []
        for a_idx, d in enumerate(self.ACTION_DELTA):
            o2 = (o + d) % 4
            dy, dx = self.MOVES[o2]
            ny, nx = y + dy, x + dx
            if self._is_free(ny, nx) and not self.visited[ny, nx]:
                pref = self._serp_pref(y, x, o, ny, nx, o2)
                v2 = (self.ACTION_COST[a_idx]
                      + (dir_bias * float(heuristic_map[ny, nx]))
                      + self.serp_weight * pref)
                cand.append((v2, ny, nx, o2, a_idx))

        if not cand:
            # 주변에 미방문 free 없음
            return False, traj

        cand.sort(key=lambda t: t[0])
        v2, ny, nx, o2, a_idx = cand[0]
        # 현재 스텝의 next_action 메모, 다음 스텝 append
        traj[-1][5] = a_idx
        traj.append([v2, ny, nx, o2, a_idx, None, self.state_])
        # 방문 마킹
        self.visited[ny, nx] = True
        return True, traj

    # ---- 프론티어(BFS) ----
    def _nearest_frontier_bfs(self, start_yx):
        """
        frontier = free & not visited & (4-이웃 중 visited=True 존재)
        BFS 레벨(layer)별로 탐색해서, 같은 거리에서 발견된 프론티어 중
        지그재그 순번(serp_rank)이 가장 앞선 타깃을 반환.
        """
        sy, sx = start_yx
        if not (0 <= sy < self.H and 0 <= sx < self.W):
            return False, None

        from collections import deque
        q = deque([(sy, sx)])
        seen = np.zeros((self.H, self.W), dtype=bool)
        seen[sy, sx] = True

        def is_frontier(y, x):
            if not (self._is_free(y, x) and not self.visited[y, x]):
                return False
            for ny, nx in self._neighbors4(y, x):
                if self._is_free(ny, nx) and self.visited[ny, nx]:
                    return True
            return False

        has_any_visited = bool(self.visited.any())

        while q:
            layer = list(q)
            q.clear()
            layer_frontiers = []

            for y, x in layer:
                if has_any_visited:
                    if is_frontier(y, x):
                        layer_frontiers.append((y, x))
                else:
                    # 시작 직후엔 아무 free 도 OK
                    if self._is_free(y, x) and not self.visited[y, x]:
                        layer_frontiers.append((y, x))

                for ny, nx in self._neighbors4(y, x):
                    if not seen[ny, nx] and self._is_free(ny, nx):
                        seen[ny, nx] = True
                        q.append((ny, nx))

            if layer_frontiers:
                best = min(layer_frontiers, key=lambda p: self._serp_rank(p[0], p[1]))
                return True, best

        return False, None

    # ---- 방향 인지 A* ----
    def _astar_oriented(self, start_yxo, goal_yx):
        sy, sx, so = start_yxo
        gy, gx = goal_yx

        def h(y, x):
            # 맨해튼 * 전진비용
            return (abs(y - gy) + abs(x - gx)) * self.ACTION_COST[1]

        # f,g, y,x,o
        pq = [(h(sy, sx), 0.0, sy, sx, so)]
        # came[(y,x,o)] = (py,px,po, a_idx)
        came = {(sy, sx, so): None}
        gscore = {(sy, sx, so): 0.0}

        while pq:
            f, g, y, x, o = heapq.heappop(pq)
            if (y, x) == (gy, gx):
                # reconstruct
                path = []
                cur = (y, x, o)
                while cur is not None:
                    prev = came[cur]
                    if prev is None:
                        path.append([0.0, cur[0], cur[1], cur[2], None, None, self.state_])
                    else:
                        py, px, po, a_idx = prev
                        path.append([gscore[cur], cur[0], cur[1], cur[2], a_idx, None, self.state_])
                    cur = None if prev is None else (prev[0], prev[1], prev[2])
                path.reverse()
                # next_action 채우기
                for i in range(len(path)-1):
                    path[i][5] = path[i+1][4]
                return True, path

            # 확장: 네 가지 액션
            for a_idx, d in enumerate(self.ACTION_DELTA):
                o2 = (o + d) % 4
                dy, dx = self.MOVES[o2]
                ny, nx = y + dy, x + dx
                if not self._is_free(ny, nx):
                    continue
                ng = g + self.ACTION_COST[a_idx]
                key = (ny, nx, o2)
                if ng < gscore.get(key, 1e30):
                    gscore[key] = ng
                    came[key] = (y, x, o, a_idx)
                    heapq.heappush(pq, (ng + h(ny, nx), ng, ny, nx, o2))

        return False, []

    # ---- 커버 완료 판정 ----
    def check_full_coverage(self):
        free = (self.map_grid == 0)
        return np.all(~free | self.visited)