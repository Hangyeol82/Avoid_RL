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
    개선 포인트
    - visited을 별도의 bool 마스크로 관리(장애물=1과 혼동 제거)
    - 프론티어(frontier) = (free & not visited) 이면서 4-이웃 중 visited가 하나 이상
    - '가장 가까운 프론티어'는 BFS로 추출
    - 프론티어까지 경로는 (x,y,orientation) 상태의 A*로 탐색(턴 비용 반영)
    - coverage_search는 약한 방향편향(dir_bias)을 더한 국소 탐욕(기존 구조 유지)
    """

    # ---- 움직임/액션 정의(기존과 동일 의미) ----
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
            # 시작이 없으면 첫 free를 start로
            ys, xs = np.where(self.map_grid == 0)
            if len(ys) == 0:
                raise RuntimeError("free 셀이 없습니다.")
            self.map_grid[ys[0], xs[0]] = 2
            self.current_pos = [int(ys[0]), int(xs[0]), 0]

        # 방문 마스크(장애물과 분리!)
        self.visited = np.zeros_like(self.map_grid, dtype=bool)

        # 결과 누적
        self.current_trajectory = []              # [v,x,y,o,a_in,a_next,state]
        self.current_trajectory_annotations = []  # 시각화용 태그

        # FSM
        self.state_ = PlannerStatus.STANDBY

        # 휴리스틱 설정(coverage는 수직/수평 약한 편향, A*는 맨해튼)
        self.a_star_heuristic = HeuristicType.MANHATTAN
        self.cp_heuristic = HeuristicType.VERTICAL
        self.dir_bias = 0.05   # coverage 탐욕에서 휴리스틱 가중(작게!)
        self.debug_level = -1

        # 캐시(수직/수평 편향은 고정 앵커 기준으로 1회 생성)
        self._cp_heuristic_map = None

    # ============== 퍼블릭 API ==============

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

        # 수직/수평 편향은 (0,0) 앵커 기준으로 캐시(고정 그라디언트)
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
            # 국소 커버리지 확장(방향 편향은 약하게)
            heuristic = self._cp_heuristic_map if self._cp_heuristic_map is not None \
                        else self._create_heuristic(self.current_pos, self.cp_heuristic)

            ok, traj = self._coverage_search_step(self.current_pos, heuristic, self.dir_bias)

            # 방문/위치 반영
            if traj:
                self._append_trajectory(traj, "CS")
                self.current_pos = [traj[-1][1], traj[-1][2], traj[-1][3]]

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
                # 방향 인지 A*로 goal까지 잇기
                ok, traj = self._astar_oriented(self.current_pos, goal)
                if ok and traj:
                    self._append_trajectory(traj, "A*")
                    self.current_pos = [traj[-1][1], traj[-1][2], traj[-1][3]]
                    # 도착했으면 다시 커버리지 모드
                    self.state_ = PlannerStatus.COVERAGE_SEARCH
                    searching = True
                else:
                    # 이 프론티어는 막혔음 → 다음 compute에서 다시 다른 프론티어 탐색
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
        # 반환 형식:
        #  - found: bool
        #  - total_steps: int
        #  - total_cost: float
        #  - trajectory: list[[v, y, x, o, a_in, a_next, state], ...]
        #  - xy_trajectory: list[(x, y), ...]  ← 주의: (x,y) 순서
        return (found, total_steps, total_cost, self.current_trajectory, xy_trajectory)

    def show_results(self):
        self._printd("show_results", f"Final: {self.state_.name}")
        self._printd("show_results", f"Steps: {max(0, len(self.current_trajectory)-1)}")
        self._printd("show_results", f"Cost : {self._trajectory_cost(self.current_trajectory):.2f}")
        if self.debug_level > 0:
            self._print_trajectory(self.current_trajectory)
        self._print_policy_map()

    # ============== 내부 구현 ==============

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
        # (y,x) → (x,y)로 변환하여 반환
        return [(t[2], t[1]) for t in trajectory]

    def _print_trajectory(self, trajectory):
        print("l_cost\ty\tx\tori\tact_in\ta_next\tstate")
        for t in trajectory:
            print(f"{t[0]:.2f}\t{t[1]}\t{t[2]}\t{t[3]}\t{t[4]}\t{t[5]}\t{t[6].name}")

    def _print_map(self, M):
        for r in M:
            print("[" + ",\t".join(f"{v}" for v in r) + "]")

    def _print_policy_map(self, trajectory=None, annotations=None):
        if trajectory is None:
            trajectory = self.current_trajectory
        if annotations is None:
            annotations = list(self.current_trajectory_annotations)

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
            annotations = annotations + [
                [trajectory[0][1], trajectory[0][2], "STA"],
                [trajectory[-1][1], trajectory[-1][2], "END"],
            ]

        for (y, x, name) in annotations:
            if 0 <= y < self.H and 0 <= x < self.W:
                policy[y][x] += f"@{name}"

        self._printd("policy", "Policy Map:")
        if self.debug_level > 0:
            self._print_map(policy)

    # ---- 휴리스틱 ----
    def _create_heuristic(self, target_point, htype: HeuristicType):
        ty, tx = target_point if isinstance(target_point, (list, tuple)) else (0, 0)
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

    # ---- 커버리지 국소 탐욕 한 스텝 ----
    def _coverage_search_step(self, initial_pos, heuristic_map, dir_bias=0.05):
        """
        현재 위치에서 '한 구간' 전진.
        - 방문하지 않은 free만 우선(없으면 종료)
        - 비용 = action_cost + dir_bias * heuristic
        반환:
          ok, traj(list)
        """
        y, x, o = initial_pos
        if not self._is_free(y, x):
            return False, []

        # 현 위치 방문
        self.visited[y, x] = True
        traj = [[0.0, y, x, o, None, None, self.state_]]

        # 다음 후보(미방문 free) 모으기
        cand = []
        for a_idx, d in enumerate(self.ACTION_DELTA):
            o2 = (o + d) % 4
            dy, dx = self.MOVES[o2]
            ny, nx = y + dy, x + dx
            if self._is_free(ny, nx) and not self.visited[ny, nx]:
                v2 = self.ACTION_COST[a_idx] + (dir_bias * float(heuristic_map[ny, nx]))
                cand.append((v2, ny, nx, o2, a_idx))

        if not cand:
            # 주변에 미방문 free 없음 → 커버리지 확장 종료
            return False, traj

        cand.sort(key=lambda t: t[0])
        v2, ny, nx, o2, a_idx = cand[0]
        # 현재 스텝의 next_action 메모, 다음 스텝 append
        traj[-1][5] = a_idx
        traj.append([v2, ny, nx, o2, a_idx, None, self.state_])
        # 방문 마킹
        self.visited[ny, nx] = True
        return True, traj

    # ---- 프론티어 탐색(BFS: 가장 가까운 미방문 접점) ----
    def _nearest_frontier_bfs(self, start_yx):
        """
        frontier = free & not visited & (4-이웃 중 visited=True 존재)
        start에서 free를 통과하며 BFS로 가장 가까운 frontier 하나를 찾는다.
        """
        sy, sx = start_yx
        if not (0 <= sy < self.H and 0 <= sx < self.W):
            return False, None

        from collections import deque
        q = deque()
        q.append((sy, sx))
        seen = np.zeros((self.H, self.W), dtype=bool)
        seen[sy, sx] = True

        def is_frontier(y, x):
            if not (self._is_free(y, x) and not self.visited[y, x]):
                return False
            for ny, nx in self._neighbors4(y, x):
                if self._is_free(ny, nx) and self.visited[ny, nx]:
                    return True
            return False

        # 특례: 전체가 미방문이면 단순히 가장 가까운 free 도 OK
        has_any_visited = bool(self.visited.any())

        while q:
            y, x = q.popleft()
            if has_any_visited:
                if is_frontier(y, x):
                    return True, (y, x)
            else:
                # 시작 직후엔 아무 데나 free로
                if self._is_free(y, x) and not self.visited[y, x]:
                    return True, (y, x)

            for ny, nx in self._neighbors4(y, x):
                if not seen[ny, nx] and self._is_free(ny, nx):
                    seen[ny, nx] = True
                    q.append((ny, nx))

        return False, None

    # ---- 방향 인지 A* (상태: y,x,orientation) ----
    def _astar_oriented(self, start_yxo, goal_yx):
        sy, sx, so = start_yxo
        gy, gx = goal_yx

        def h(y, x):
            # 간단: 맨해튼 * 전진비용
            return (abs(y - gy) + abs(x - gx)) * self.ACTION_COST[1]

        # g,f, y,x,o
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
                        # [v,y,x,o,a_in,a_next,state]
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