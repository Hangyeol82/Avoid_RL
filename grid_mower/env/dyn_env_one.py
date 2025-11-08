# env/dyn_env_one.py
from typing import Optional, List, Tuple
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from collections import deque

from env.moving_object import MovingObj
from utils_timing import estimate_robot_timeline


class DynAvoidOneObjEnv(gym.Env):
    """
    - 좌표계: grid[y, x]  (y: 행, 아래로 증가 / x: 열, 오른쪽으로 증가)
    - waypoints: shape (N,2) = (x, y)  ← 주의! (y,x) 아님
    - 하이브리드 모드:
        FOLLOW_CPP: CPP 경로 추종(기본)
        AVOID     : PPO가 회피 액션
    - 관측:
        [ goal_dist/Rg, cos(goal), sin(goal),
          obj_dist/Ro, cos(obj), sin(obj),
          obj_speed/vmax, ttc/Tcap,
          lidar_0/Rray, ... lidar_{N-1}/Rray ]
    """
    metadata = {"render_modes": ["human"]}

    # ----------------------- 초기화 -----------------------
    def __init__(self, grid: np.ndarray, waypoints: np.ndarray, seed: int = 0, cell_size_m: float = 0.20):
        super().__init__()
        assert grid.ndim == 2
        self.rng = np.random.default_rng(seed)

        # ==== (1) 월드 전체 패딩: 테두리를 '진짜 벽(1)'로 ====
        self.grid, self.offset = self._pad_world(grid)  # grid는 패딩된 맵으로 교체
        self.H, self.W = self.grid.shape

        # 웨이포인트도 +offset
        self.waypoints = waypoints.astype(np.float32).copy()
        self.waypoints[:, 0] += self.offset[1]  # x += +1
        self.waypoints[:, 1] += self.offset[0]  # y += +1

        # 시작 위치(원본 grid==2가 있었다면 그 좌표를 +offset, 없으면 첫 free)
        self.agent_rc = self._find_start_rc(grid)

        # --- 셀→미터 변환 비율
        self.cell_size_m = float(cell_size_m)

        # --- 센서/정규화 사양
        self.R_goal_m = 5.0
        self.R_obj_m  = 5.0
        self.R_ray_m  = 6.0
        self.T_cap_s  = 5.0
        self.vmax_obj = 1.5

        # --- 라이다(정적 장애물만)
        self.ray_count  = 16
        self.ray_angles = np.linspace(-np.pi, np.pi, self.ray_count, endpoint=False)

        # --- 액션/관측 공간
        self.action_space = spaces.Discrete(5)  # 0=상,1=좌,2=하,3=우,4=정지
        self.obs_dim = 3 + 3 + 1 + 1 + self.ray_count  # = 25
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32)

        # 이동 벡터(상,좌,하,우)
        self.moves = [(-1,0), (0,-1), (1,0), (0,1)]

        # 상태 변수
        self.wp_idx: int = 0
        self.steps: int = 0
        self.deviated_from_cpp: bool = False
        self.prev_obj_dist_m: Optional[float] = None
        self.prev_agent_rc: Optional[np.ndarray] = self.agent_rc.copy()
        self.avoiding: bool = False
        self.visited: Optional[np.ndarray] = np.zeros(len(self.waypoints), dtype=bool)
        self.dynamic_objs: List[MovingObj] = []

        # =========== 경로 방해(Route-Blocked) 모니터 ===========
        # (A) 목표-객체 '원 교집합' 기반의 방해율
        self.goal_radius_m = 0.35
        self.obj_radius_m  = 0.35
        self.block_win = 25
        self.block_rate_thresh = 0.80            # 윈도우 평균(now) 임계
        self.block_future_horizon = 6            # t=1..H
        self.block_future_rate_thresh = 0.7     # 미래 교차율 임계
        self.rb_block_hist = deque(maxlen=self.block_win)
        self.rb_future_hist = deque(maxlen=self.block_win)

        # (B) 진행 품질(평균이동 함정 회피): 효율/역진동/목표정렬
        self.rb_win = 20
        self.rb_hist_vec = deque(maxlen=self.rb_win)  # (dy,dx) 누적
        # 보수적(원하면 더 빡세게 낮추거나/높이기)
        self.rb_eff_thresh = 0.25           # 낮으면 진행효율 나쁨
        self.rb_backtrack_thresh = 0.40     # 높으면 역진동 심함
        self.rb_prog_ratio_thresh = 0.10    # 낮으면 목표정렬 저조

        # 최근 관측 캐시(디버그용)
        self._last_d_goal_m = 0.0
        self._last_ttc = 1.0
        self._last_rays = np.ones(self.ray_count, dtype=np.float32)
        self._last_goal_angle_math = 0.0

        # 실제 동적 객체 스폰/초기화
        self.reset()

        # 렌더 옵션
        self._render_on = False
        self._fig = None
        self._ax = None

    # ----------------------- 월드 패딩 -----------------------
    def _pad_world(self, grid_raw: np.ndarray):
        """grid_raw(0/1/2)를 테두리 1로 패딩한 새 grid를 반환. offset=(+1,+1)."""
        H, W = grid_raw.shape
        pad = np.ones((H+2, W+2), dtype=int)
        pad[1:-1, 1:-1] = grid_raw
        # 시각 확인용 로그(원하면 주석)
        # print(f"[WORLD PAD] OK: {pad.shape} border=1 applied.")
        return pad, (1,1)  # (dy,dx) offset

    def _find_start_rc(self, grid_raw: np.ndarray) -> np.ndarray:
        """원본 grid에서 2(시작)를 찾아 +offset. 없으면 패딩된 맵에서 첫 free."""
        locs = np.argwhere(grid_raw == 2)
        if len(locs) > 0:
            r, c = locs[0]
            return np.array([float(r + self.offset[0]), float(c + self.offset[1])], dtype=float)
        # fallback: 패딩된 맵에서 free(0) 첫 칸
        free = np.argwhere(self.grid == 0)
        if len(free) == 0:
            raise RuntimeError("No free cell to start.")
        r, c = free[0]
        return np.array([float(r), float(c)], dtype=float)

    # ----------------------- 유틸 -----------------------
    def _cells_to_m(self, d_cells: float) -> float:
        return float(d_cells) * self.cell_size_m

    def _m_to_cells(self, d_m: float) -> float:
        return float(d_m) / self.cell_size_m

    def _is_free(self, r, c):
        return 0 <= r < self.H and 0 <= c < self.W and self.grid[int(r), int(c)] != 1

    @staticmethod
    def _wrap_pi(a):
        out = (a + np.pi) % (2*np.pi)
        if out <= 0:
            out += 2*np.pi
        return out - np.pi

    def _goal_angle_math(self, dy_cells, dx_cells):
        """라이다가 쓰는 수학 좌표(+x가 0rad, CCW+)에 맞춘 goal 방향 각도."""
        return float(np.arctan2(-dy_cells, dx_cells))

    # ----------------------- 라이다(정적만) -----------------------
    def _raycast_static(self, origin_yx, theta, max_range_m):
        """패딩된 self.grid를 그대로 사용(테두리=벽)."""
        grid = self.grid
        H, W = grid.shape
        max_cells = self._m_to_cells(max_range_m)

        y = float(origin_yx[0])
        x = float(origin_yx[1])

        dx = np.cos(theta)
        dy = np.sin(theta)
        dy_grid = -dy

        t = 0.0
        step = 0.25
        while t <= max_cells:
            yy = int(round(y + dy_grid * t))
            xx = int(round(x + dx       * t))
            if yy < 0 or yy >= H or xx < 0 or xx >= W:
                return self._cells_to_m(max(t - step, 0.0))
            if grid[yy, xx] == 1:
                return self._cells_to_m(t)
            t += step
        return max_range_m

    # ----------------------- 동적 객체 거리 -----------------------
    def _distance_to_nearest_obj_m(self):
        if not hasattr(self, "dynamic_objs") or len(self.dynamic_objs) == 0:
            return float("inf")
        ry, rx = self.agent_rc
        d_cells = min(np.hypot(obj.p[0]-ry, obj.p[1]-rx) for obj in self.dynamic_objs)
        return self._cells_to_m(d_cells)

    # ----------------------- 이동 -----------------------
    def _move_toward(self, wp_xy):
        """웨이포인트를 향해 대각선 1칸(격자 기준) 이동."""
        tx, ty = wp_xy
        y, x = self.agent_rc
        dy = np.sign(float(ty) - float(y))
        dx = np.sign(float(tx) - float(x))
        new_y = int(y + dy)
        new_x = int(x + dx)
        if self._is_free(new_y, new_x):
            self.agent_rc = np.array([new_y, new_x], dtype=float)
        # 반환: 실행된 '격자 이동'을 4방 액션 중 가장 가까운 것으로(로깅용)
        if dy == -1 and dx == 0: return 0
        if dy == 0 and dx == -1: return 1
        if dy == 1 and dx == 0: return 2
        if dy == 0 and dx == 1: return 3
        return 4  # 대각선/정지 → 정지로 표기

    def _move_agent(self, action):
        if action is None:
            return True
        dy, dx = {0:(-1,0), 1:(0,-1), 2:(1,0), 3:(0,1), 4:(0,0)}[int(action)]
        ny = int(self.agent_rc[0] + dy)
        nx = int(self.agent_rc[1] + dx)
        if self._is_free(ny, nx):
            self.agent_rc = np.array([ny, nx], dtype=float)
            return True
        return False

    # ----------------------- 웨이포인트 -----------------------
    def _reached_waypoint(self):
        if self.wp_idx >= len(self.waypoints):
            return False
        gx, gy = self.waypoints[self.wp_idx]
        ry, rx = self.agent_rc
        return np.hypot(gx - rx, gy - ry) < 1.0

    # ----------------------- 관측 -----------------------
    def _obs(self):
        ry, rx = self.agent_rc
        if self.wp_idx >= len(self.waypoints):
            self.wp_idx = len(self.waypoints) - 1
        gx, gy = self.waypoints[self.wp_idx]  # (x,y)

        # 목표(relative): 거리, 각도
        dy_cells = gy - ry
        dx_cells = gx - rx
        d_goal_m = self._cells_to_m(np.hypot(dy_cells, dx_cells))
        angle_goal = np.arctan2(dy_cells, dx_cells)
        goal_feats = np.array([
            min(d_goal_m, self.R_goal_m) / self.R_goal_m,
            np.cos(angle_goal),
            np.sin(angle_goal)
        ], dtype=np.float32)

        # 동적(relative)
        if hasattr(self, "dynamic_objs") and len(self.dynamic_objs) > 0:
            dists_cells = [np.hypot(obj.p[0]-ry, obj.p[1]-rx) for obj in self.dynamic_objs]
            k = int(np.argmin(dists_cells))
            obj = self.dynamic_objs[k]
            oy, ox = obj.p
            ovy, ovx = obj.v

            ody_cells = oy - ry
            odx_cells = ox - rx
            d_obj_m = self._cells_to_m(np.hypot(ody_cells, odx_cells))
            angle_obj = np.arctan2(ody_cells, odx_cells)

            obj_pos_feats = np.array([
                min(d_obj_m, self.R_obj_m) / self.R_obj_m,
                np.cos(angle_obj),
                np.sin(angle_obj)
            ], dtype=np.float32)

            obj_speed_mps = self._cells_to_m(np.hypot(ovx, ovy))
            obj_speed_feat = np.array([min(obj_speed_mps, self.vmax_obj) / self.vmax_obj], dtype=np.float32)

            rel_p_m = np.array([self._cells_to_m(odx_cells), self._cells_to_m(ody_cells)], dtype=np.float64)
            rel_v_mps = np.array([self._cells_to_m(ovx), self._cells_to_m(ovy)], dtype=np.float64)
            denom = float(np.dot(rel_v_mps, rel_v_mps))
            if denom > 1e-9:
                ttc = max(0.0, -float(np.dot(rel_p_m, rel_v_mps)) / denom)
            else:
                ttc = self.T_cap_s
            ttc_feat = np.array([min(ttc, self.T_cap_s) / self.T_cap_s], dtype=np.float32)
        else:
            obj_pos_feats = np.zeros(3, dtype=np.float32)
            obj_speed_feat = np.zeros(1, dtype=np.float32)
            ttc_feat = np.array([1.0], dtype=np.float32)

        # 라이다
        origin = (ry, rx)
        rays_m = [self._raycast_static(origin, ang, self.R_ray_m) for ang in self.ray_angles]
        rays   = np.array([r / self.R_ray_m for r in rays_m], dtype=np.float32)

        # === 캐시(경로 방해/디버그용) ===
        self._last_d_goal_m = float(d_goal_m)
        self._last_goal_angle_math = self._goal_angle_math(dy_cells, dx_cells)
        self._last_rays = rays.copy()
        self._last_ttc  = float(ttc_feat[0])

        obs = np.concatenate([goal_feats, obj_pos_feats, obj_speed_feat, ttc_feat, rays], axis=0)
        assert obs.shape[0] == self.obs_dim
        return obs

    # ----------------------- 경로 방해 유틸 -----------------------
    @staticmethod
    def _circle_overlap(c1, r1, c2, r2) -> bool:
        """원(c1,r1)과 원(c2,r2)가 겹치면 True. c: (x,y) in meters."""
        dx = c1[0] - c2[0]
        dy = c1[1] - c2[1]
        return (dx*dx + dy*dy) <= (r1 + r2) * (r1 + r2)

    def _goal_xy_m(self):
        gx, gy = self.waypoints[self.wp_idx]
        return np.array([self._cells_to_m(gx), self._cells_to_m(gy)], dtype=float)

    def _obj_xy_m(self, obj: MovingObj):
        return np.array([self._cells_to_m(obj.p[1]), self._cells_to_m(obj.p[0])], dtype=float)

    def _motion_metrics(self, goal_xy_cells):
        """효율/역진동/목표정렬 진행률 계산."""
        v = np.array(self.rb_hist_vec, dtype=float)  # (k,2) with (dy,dx)
        k = len(v)
        if k == 0:
            return 0.0, 0.0, 0.0

        seg_len = np.linalg.norm(v, axis=1).sum()
        net_vec = v.sum(axis=0)
        eff = float(np.linalg.norm(net_vec) / (seg_len + 1e-6))

        back = 0
        for i in range(1, k):
            a = v[i-1]; b = v[i]
            if np.allclose(a + b, 0.0):
                back += 1
        backrate = float(back / max(1, k-1))

        ry, rx = self.agent_rc
        gx, gy = goal_xy_cells
        g = np.array([gy - ry, gx - rx], dtype=float)  # (dy,dx)
        g_norm = np.linalg.norm(g)
        if g_norm < 1e-9 or seg_len < 1e-9:
            prog_ratio = 0.0
        else:
            g_hat = g / g_norm
            proj = float((v @ g_hat).sum())
            prog_ratio = float(proj / seg_len)

        return eff, backrate, prog_ratio

    # ----------------------- reset -----------------------
    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        # 시간-교차 동적 객체(예: 혼합 스폰). 필요시 사용자 코드로 교체.
        self.dynamic_objs = self._default_spawn(self.grid, self.waypoints, self.rng, v_robot=1.0)

        # 북키핑 리셋
        self.wp_idx   = 0
        self.steps    = 0
        self.deviated_from_cpp = False
        self.prev_obj_dist_m   = None
        # 시작 위치는 __init__에서 정함(패딩 반영)
        self.prev_agent_rc     = self.agent_rc.copy()
        self.avoiding          = False
        self.visited           = np.zeros(len(self.waypoints), dtype=bool)

        # 버퍼 초기화
        self.rb_block_hist.clear()
        self.rb_future_hist.clear()
        self.rb_hist_vec.clear()

        return self._obs(), {}

    # (데모용) 기본 스폰
    @staticmethod
    def _default_spawn(occ_grid, waypoints, rng, v_robot=1.0, v_obj_range=(0.6,1.2), k_min=1, k_max=2, max_retry=50):
        H, W = occ_grid.shape
        objs = []
        t_robot = estimate_robot_timeline(waypoints, v_robot_cells_per_step=v_robot)
        candidates = ["cv", "random_walk", "sin", "circle"]
        alias = {"random_walk":"ou", "sin":"patrol", "circle":"patrol"}

        K = int(rng.integers(k_min, k_max+1))
        for _ in range(K):
            raw_kind = rng.choice(candidates)
            kind = alias.get(raw_kind, raw_kind)
            created = False
            if kind == "cv":
                for _a in range(max_retry):
                    idx = int(rng.integers(3, max(4, len(waypoints)-3))) if len(waypoints) >= 6 else int(rng.integers(0, len(waypoints)))
                    wp_xy = np.array(waypoints[idx], float)
                    target_rc = np.array([wp_xy[1], wp_xy[0]], float)
                    t = float(t_robot[idx]) if len(t_robot) > idx else float(rng.uniform(8.0,20.0))
                    vmag = rng.uniform(*v_obj_range)
                    theta = rng.uniform(0, 2*np.pi)
                    vy, vx = np.sin(theta)*vmag, np.cos(theta)*vmag
                    start_rc = target_rc - np.array([vy, vx])*t
                    sy, sx = start_rc
                    if 1 < sy < H-2 and 1 < sx < W-2 and occ_grid[int(sy), int(sx)] == 0:
                        objs.append(MovingObj(pos=(float(sy), float(sx)),
                                              vel=np.array([vy, vx], float),
                                              vmax=max(v_obj_range[1],1.2),
                                              kind="cv",
                                              seed=int(rng.integers(1e9))))
                        created = True
                        break
            else:
                for _a in range(max_retry):
                    sy = int(rng.integers(1, H-1)); sx = int(rng.integers(1, W-1))
                    if occ_grid[sy, sx] != 0: continue
                    vmag = rng.uniform(*v_obj_range)
                    theta = rng.uniform(0, 2*np.pi)
                    vy, vx = np.sin(theta)*vmag, np.cos(theta)*vmag
                    objs.append(MovingObj(pos=(float(sy), float(sx)),
                                          vel=np.array([vy, vx], float),
                                          vmax=max(v_obj_range[1],1.2),
                                          kind=kind,
                                          seed=int(rng.integers(1e9))))
                    created = True
                    break
            if not created:
                # 백업
                while True:
                    sy = int(rng.integers(1, H-1)); sx = int(rng.integers(1, W-1))
                    if occ_grid[sy, sx] == 0: break
                vmag = rng.uniform(*v_obj_range)
                theta = rng.uniform(0, 2*np.pi)
                vy, vx = np.sin(theta)*vmag, np.cos(theta)*vmag
                objs.append(MovingObj(pos=(float(sy), float(sx)),
                                      vel=np.array([vy, vx], float),
                                      vmax=max(v_obj_range[1],1.2),
                                      kind="cv",
                                      seed=int(rng.integers(1e9))))
        return objs

    # ----------------------- step -----------------------
    def step(self, action_from_ppo=None):
        reward = 0.0
        done   = False
        info   = {}

        # 동적 객체 이동
        for obj in getattr(self, "dynamic_objs", []):
            obj.move(self.grid)

        # 임계값
        COLLISION_M = 0.15
        DANGER_M    = 1.05
        SAFE_M      = 1.35

        dist_to_obj_m = self._distance_to_nearest_obj_m()

        # 모드 결정
        mode = "AVOID" if dist_to_obj_m < DANGER_M else "FOLLOW_CPP"
        info["mode"] = mode

        executed_action = 4  # 로깅용(정지)

        # ---------------- FOLLOW_CPP ----------------
        if mode == "FOLLOW_CPP":
            if getattr(self, "deviated_from_cpp", False):
                ry, rx = self.agent_rc
                curr_xy = np.array([rx, ry], dtype=float)
                unvisited = np.where(~self.visited)[0]
                if len(unvisited) > 0:
                    cand = self.waypoints[unvisited]
                    d = np.linalg.norm(cand - curr_xy[None, :], axis=1)
                    self.wp_idx = int(unvisited[int(np.argmin(d))])
                else:
                    self.wp_idx = len(self.waypoints)-1
                self.deviated_from_cpp = False

            if self.wp_idx < len(self.waypoints):
                executed_action = self._move_toward(self.waypoints[self.wp_idx])
                reward += 0.05

        # ---------------- AVOID ----------------
        else:
            self.avoiding = True
            moved_successfully = True
            if action_from_ppo is not None:
                executed_action = int(action_from_ppo)
                moved_successfully = self._move_agent(executed_action)
            if not moved_successfully:
                reward -= 0.25

            # 간단 미래 충돌 패널티
            ry, rx = self.agent_rc
            future_pen = 0.0
            for obj in self.dynamic_objs:
                future_pos = obj.p + obj.v * 5.0
                future_pos_m = np.array([
                    self._cells_to_m(future_pos[1] - rx),
                    self._cells_to_m(future_pos[0] - ry)
                ])
                fdist = np.linalg.norm(future_pos_m)
                if fdist < 1.5:
                    future_pen += (1.5 - fdist) * 0.2
            reward -= future_pen

            # 거리 변화 보상
            prev = self.prev_obj_dist_m if (self.prev_obj_dist_m is not None) else dist_to_obj_m
            delta = dist_to_obj_m - prev
            self.prev_obj_dist_m = dist_to_obj_m
            reward += 0.2 * np.clip(delta, -1.0, 1.0)

            # 충돌/위험/안전
            if dist_to_obj_m <= COLLISION_M:
                reward -= 2.0
                done = True
            elif dist_to_obj_m >= SAFE_M:
                reward += 0.3
                self.avoiding = False
            else:
                reward -= 0.05 * (SAFE_M - dist_to_obj_m)

            # 거의 안 움직였으면 패널티(소폭)
            move_dist = float(np.linalg.norm(self.agent_rc - self.prev_agent_rc))
            if move_dist < 0.1:
                reward -= 0.1

            """ 대강 막혀 있으면 경로 우회 하는 코드 """
            # if self.wp_idx < len(self.waypoints):
            #     wx, wy = self.waypoints[self.wp_idx]
            #     ay, ax = self.agent_rc
            #     cpp_dist_cells = np.hypot(wy - ay, wx - ax)
            #     if cpp_dist_cells > 3.0:
            #         self.deviated_from_cpp = True

        # === 이동 벡터 기록(공통) ===
        dyx = self.agent_rc - self.prev_agent_rc  # (dy,dx)
        self.rb_hist_vec.append(dyx.copy())
        self.prev_agent_rc = self.agent_rc.copy()

        # === 경로 방해(now/future) 계산 ===
        block_now = False
        block_future = False
        goal_xy_m = self._goal_xy_m()
        if len(self.dynamic_objs) > 0 and self.wp_idx < len(self.waypoints):
            for obj in self.dynamic_objs:
                obj_xy_m = self._obj_xy_m(obj)
                # now
                if self._circle_overlap(goal_xy_m, self.goal_radius_m, obj_xy_m, self.obj_radius_m):
                    block_now = True
                # future (t=1..H)
                for t in range(1, self.block_future_horizon+1):
                    fut = obj_xy_m + np.array([self._cells_to_m(obj.v[1]*t),
                                               self._cells_to_m(obj.v[0]*t)], dtype=float)
                    if self._circle_overlap(goal_xy_m, self.goal_radius_m, fut, self.obj_radius_m):
                        block_future = True
                        break

        self.rb_block_hist.append(1.0 if block_now else 0.0)
        self.rb_future_hist.append(1.0 if block_future else 0.0)
        block_rate = float(np.mean(self.rb_block_hist)) if len(self.rb_block_hist) > 0 else 0.0
        block_future_rate = float(np.mean(self.rb_future_hist)) if len(self.rb_future_hist) > 0 else 0.0

        # === 진행 품질 지표 ===
        if self.wp_idx < len(self.waypoints):
            goal_xy_cells = self.waypoints[self.wp_idx]
        else:
            goal_xy_cells = self.waypoints[-1]
        eff, backrate, prog_ratio = self._motion_metrics(goal_xy_cells)
        move_mean_m = 0.0
        if len(self.rb_hist_vec) > 0:
            move_mean_m = float(np.mean(np.linalg.norm(np.array(self.rb_hist_vec), axis=1))) * self.cell_size_m

        # 최종 경로 방해 판정: (교차율 높음) AND (진행 품질 나쁨)
        progress_abnormal = (eff <= self.rb_eff_thresh) or (backrate >= self.rb_backtrack_thresh) or (prog_ratio <= self.rb_prog_ratio_thresh)
        route_blocked = ((block_rate >= self.block_rate_thresh) or (block_future_rate >= self.block_future_rate_thresh)) and progress_abnormal

        # info 채우기(시각화/디버그)
        info["route_blocked"] = bool(route_blocked)
        info["block_now"] = bool(block_now)
        info["block_rate"] = block_rate
        info["block_future_rate"] = block_future_rate
        info["move_mean_m"] = move_mean_m
        # 현재 목표-가장 가까운 객체 간 거리(참고)
        if len(self.dynamic_objs) > 0:
            d_now = min(np.linalg.norm(self._obj_xy_m(o) - goal_xy_m) for o in self.dynamic_objs)
            info["goal_obj_dist_now_m"] = float(d_now)
        else:
            info["goal_obj_dist_now_m"] = None

        info["efficiency"] = eff
        info["backtrack_rate"] = backrate
        info["prog_ratio"] = prog_ratio

        # 웨이포인트 처리
        if self._reached_waypoint():
            self.visited[self.wp_idx] = True
            reward += 0.4
            if self.wp_idx >= len(self.waypoints) - 1:
                done = True
                reward += 2.0
            else:
                self.wp_idx += 1

        reward = float(np.clip(reward, -1.0, 1.0))
        self.steps += 1
        return self._obs(), reward, done, False, info

    # ----------------------- 렌더 -----------------------
    def render(self):
        import matplotlib.pyplot as plt
        if not self._render_on:
            self._render_on = True
            self._fig, self._ax = plt.subplots(figsize=(6,6))
        ax = self._ax
        ax.clear()
        ax.imshow(self.grid, cmap="Greys", origin="upper")
        if len(self.waypoints) > 0:
            ax.plot(self.waypoints[:,0], self.waypoints[:,1], "b--", alpha=0.6, label="CPP")
        ax.scatter(self.agent_rc[1], self.agent_rc[0], c="royalblue", s=60, label="Robot")
        for i, obj in enumerate(getattr(self, "dynamic_objs", [])):
            ax.scatter(obj.p[1], obj.p[0], c="crimson", s=50, label="Obj" if i==0 else None)
        ax.set_xlim(0, self.W); ax.set_ylim(self.H, 0)
        ax.legend(loc="upper right")
        ax.set_title(f"step {self.steps} | wp {self.wp_idx}")
        plt.pause(0.01)