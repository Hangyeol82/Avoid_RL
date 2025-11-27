# env/dyn_env_one.py
from typing import Optional, List, Tuple
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from collections import deque

from env.moving_object import MovingObj
from planning.danger_zone import DangerZoneMap
from planning.cpp import CoveragePlanner
from utils_timing import estimate_robot_timeline


class DynAvoidOneObjEnv(gym.Env):
    """
    Hybrid: FOLLOW_CPP (기본) / AVOID (근접 시)
    - 좌표계: grid[y, x]  (y: 행↓, x: 열→)
    - waypoints: shape (N,2) = (x, y)
    - 관측(obs):
        [ goal(3) ,  per_obj(5*max_objs) , lidar(ray_count) ]
        goal: [dist/Rg, cos(goal), sin(goal)]
        per_obj(j): [dist/Ro, cos, sin, speed/vmax, ttc/Tcap]
        → '가려진(occluded)' 동적 객체는 관측에서 제외(패딩 값으로 대체)
    - 임계(칸 단위): collision_cells / danger_cells / safe_cells
    """

    metadata = {"render_modes": ["human"]}

    # ----------------------- 초기화 -----------------------
    def __init__(
        self,
        grid: np.ndarray,
        waypoints: np.ndarray,
        seed: int = 0,
        cell_size_m: float = 0.20,
        max_objs: int = 3,
        ray_count: int = 64,
        consider_occlusion_in_obs: bool = True,
        consider_occlusion_in_mode: bool = True,
        use_escape_subpolicy: bool = False,
    ):
        super().__init__()
        assert grid.ndim == 2
        self.rng = np.random.default_rng(seed)

        # ==== (1) 월드 패딩: 테두리 = 벽(1) ====
        self.grid, self.offset = self._pad_world(grid)
        self.H, self.W = self.grid.shape

        # 웨이포인트(+offset)
        self.waypoints = waypoints.astype(np.float32).copy()
        self.waypoints[:, 0] += self.offset[1]  # x
        self.waypoints[:, 1] += self.offset[0]  # y

        # 시작 위치
        self.agent_rc = self._find_start_rc(grid)

        # 스케일
        self.cell_size_m = float(cell_size_m)

        # ----- 센서/정규화 사양 (미터 기반) -----
        self.R_goal_m = 5.0
        self.R_obj_m  = 5.0
        self.R_ray_m  = 6.0
        self.T_cap_s  = 5.0
        self.vmax_obj = 1.5

        # ----- 라이다(정적 장애물 전용) -----
        self.ray_count  = int(max(1, ray_count))
        self.ray_angles = np.linspace(-np.pi, np.pi, self.ray_count, endpoint=False)

        # ----- 동적 객체 관측 슬롯 -----
        self.max_objs = int(max(1, max_objs))
        self.obj_feat_len = 5  # [dist_norm, cos, sin, speed_norm, ttc_norm]
        self.danger_feat_len = 2  # [current danger, nearby max]
        self.danger_lidar_bins = 16

        # ----- 액션/관측 공간 -----
        self.action_space = spaces.Discrete(5)  # 0=상,1=좌,2=하,3=우,4=정지
        self.obs_dim = 3 + (self.obj_feat_len * self.max_objs) + self.ray_count + self.danger_feat_len + self.danger_lidar_bins
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32)

        # 이동 벡터(상,좌,하,우)
        self.moves = [(-1,0), (0,-1), (1,0), (0,1)]

        # ----- 상태 변수 -----
        self.wp_idx: int = 0
        self.steps: int = 0
        self.deviated_from_cpp: bool = False
        self.prev_obj_dist_m: Optional[float] = None
        self.prev_agent_rc: Optional[np.ndarray] = self.agent_rc.copy()
        self.avoiding: bool = False
        self.visited: Optional[np.ndarray] = np.zeros(len(self.waypoints), dtype=bool)
        self.dynamic_objs: List[MovingObj] = []

        # ----- stuck 상태 추적 (보수적으로 상향) -----
        self.stuck_window = 16
        self.stuck_progress_min = 0.05
        self.stuck_eff_min = 0.15
        self.stuck_move_min_m = 0.05
        self.stuck_stagnation_steps = 16
        self.stuck_block_min = 0.45

        self.stuck_goal_dist_min_cells = 2.0
        self.stuck_stagnation_radius_cells = 1.0
        
        self.stuck_score_min = 1.5  # score_sum은 참고용
        self._stuck_hist = deque(maxlen=self.stuck_window)
        self._stuck_pos_hist = deque(maxlen=self.stuck_stagnation_steps)
        self._stuck_state = False
        self._prev_stuck_state = False
        self._last_replan_reason = None

        self.danger_zone_map = DangerZoneMap(self.grid.shape, decay=0.92, hard_thr=0.6)
        self.danger_block_threshold = 0.85
        self.obj_history_len = 25
        self.obj_histories = {}
        self.override_path = deque()
        self.override_target = None
        self.override_path_full = None
        self.covered_mask = np.zeros_like(self.grid, dtype=bool)
        self.visible_obj_ids = set()
        self.avoid_seen_obj_ids = set()
        self.danger_stuck_radius = 8.0
        self.danger_regions = {}
        self.danger_goal_points = {}  # key -> [목표 좌표 스냅샷들]
        self.danger_goal_links = {}
        self.danger_release_radius = 3.0
        self.danger_release_steps = 6
        self._danger_release_timers = {}
        self._last_danger_feats = np.zeros(2, dtype=np.float32)
        self._last_danger_lidar = np.zeros(self.danger_lidar_bins, dtype=np.float32)
        # 소프트 위험 마스크 임계(경로/커버리지에서 회피). danger_block_threshold는 종료/패널티용.
        self.danger_soft_block = 0.30
        self.goal_timeout_steps = 120
        self.goal_stagnation_timer = 0
        self.use_escape_subpolicy = bool(use_escape_subpolicy)
        self.escape_active = False
        self.escape_release_steps = 3
        self._escape_release_counter = 0

        # ====== 모드 전환 임계(칸 단위) ======
        self.collision_cells = 0.75
        self.danger_cells    = 5.0
        self.safe_cells      = 7.0

        # 모드 전환 시 가림막 고려 여부
        self.consider_occlusion_in_obs  = bool(consider_occlusion_in_obs)
        self.consider_occlusion_in_mode = bool(consider_occlusion_in_mode)

        # =========== 경로 방해/정체 모니터 ===========
        self.block_dyn_radius_cells = 3.5
        self.block_horizon = 6
        self.block_route_threshold = 0.65
        self.block_route_avg_threshold = 0.55
        self.block_hist = deque(maxlen=25)

        self.rb_hist_vec = deque(maxlen=20)
        self.rb_eff_thresh = 0.25
        self.rb_backtrack_thresh = 0.40
        self.rb_prog_ratio_thresh = 0.10

        # 최근 관측 캐시
        self._last_d_goal_m = 0.0
        self._last_ttc = 1.0
        self._last_rays = np.ones(self.ray_count, dtype=np.float32)
        self._last_goal_angle_math = 0.0

        # 근접 패널티(튜닝)
        self.prox_pen_scale = 0.8   # 0.5 -> 0.8
        self.prox_pen_pow   = 1.2   # 1.3 -> 1.2

        # AVOID/FOLLOW 보상계수
        self.future_pen_coef = 0.07
        self.avoid_base_pen  = -0.2
        self.follow_step_r   = 0.15
        self.delta_dist_coef = 0.15
        self.progress_coef   = 0.10
        self.progress_bonus  = 0.05
        self.avoid_follow_bonus = self.follow_step_r * 0.6

        # 목표 진행도 캐시
        self._prev_goal_dist_cells = None
        self._escape_release_counter = 0

        # 동적 객체 스폰
        self.reset()

        # 렌더 옵션x
        self._render_on = False
        self._fig = None
        self._ax = None

    # ----------------------- 월드 패딩 -----------------------
    def _pad_world(self, grid_raw: np.ndarray):
        H, W = grid_raw.shape
        pad = np.ones((H+2, W+2), dtype=int)
        pad[1:-1, 1:-1] = grid_raw
        return pad, (1,1)  # (dy,dx)

    def _find_start_rc(self, grid_raw: np.ndarray) -> np.ndarray:
        locs = np.argwhere(grid_raw == 2)
        if len(locs) > 0:
            r, c = locs[0]
            return np.array([float(r + self.offset[0]), float(c + self.offset[1])], dtype=float)
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

    def _agent_inside_danger(self):
        if self.danger_zone_map is None:
            return False
        soft = getattr(self.danger_zone_map, "soft", None)
        if soft is None:
            return False
        yy = int(round(self.agent_rc[0]))
        xx = int(round(self.agent_rc[1]))
        if not (0 <= yy < soft.shape[0] and 0 <= xx < soft.shape[1]):
            return False
        return bool(soft[yy, xx] >= float(self.danger_block_threshold))

    def _agent_inside_soft_danger(self):
        """Soft 임계(danger_soft_block) 기준 위험 여부."""
        if self.danger_zone_map is None:
            return False
        soft = getattr(self.danger_zone_map, "soft", None)
        if soft is None:
            return False
        yy = int(round(self.agent_rc[0]))
        xx = int(round(self.agent_rc[1]))
        if not (0 <= yy < soft.shape[0] and 0 <= xx < soft.shape[1]):
            return False
        thr = float(getattr(self, "danger_soft_block", 0.3))
        return bool(soft[yy, xx] >= thr)

    @staticmethod
    def _wrap_pi(a):
        out = (a + np.pi) % (2*np.pi)
        if out <= 0:
            out += 2*np.pi
        return out - np.pi

    def _goal_angle_math(self, dy_cells, dx_cells):
        return float(np.arctan2(-dy_cells, dx_cells))

    def _danger_lidar(self, bins=16, max_range=6.0):
        """
        위험 경계(soft < threshold)까지의 상대 거리.
        - 현재 이미 안전하면 1.0
        - 위험 영역 안이면, 가장 가까운 안전셀까지의 거리로 1 - (dist/max_range)
        - 끝까지 위험이면 0.0
        """
        if self.danger_zone_map is None or getattr(self.danger_zone_map, "soft", None) is None:
            return np.ones(bins, dtype=np.float32)
        soft = self.danger_zone_map.soft
        thr = float(getattr(self, "danger_soft_block", 0.3))
        angles = np.linspace(-np.pi, np.pi, bins, endpoint=False)
        ry, rx = self.agent_rc
        H, W = soft.shape
        readings = np.zeros(bins, dtype=np.float32)
        steps = int(max_range * 4)  # 세밀한 샘플링
        for i, ang in enumerate(angles):
            exit_found = False
            for t in range(1, steps + 1):
                dy = np.sin(ang) * (t * max_range / steps)
                dx = np.cos(ang) * (t * max_range / steps)
                yy = int(round(ry + dy))
                xx = int(round(rx + dx))
                if not (0 <= yy < H and 0 <= xx < W):
                    break
                if float(soft[yy, xx]) < thr:
                    dist = np.hypot(dy, dx)
                    readings[i] = max(0.0, 1.0 - dist / max_range)
                    exit_found = True
                    break
            if not exit_found:
                # 현재 이미 안전하면 1.0, 아니면 0.0
                readings[i] = 1.0 if float(soft[int(round(ry)), int(round(rx))]) < thr else 0.0
        return readings
    # ----------------------- 브레젠험: 가림막 판정 -----------------------
    @staticmethod
    def _bresenham_cells(y0: int, x0: int, y1: int, x1: int):
        """(y0,x0)→(y1,x1) 직선상의 격자 셀들을 생성(양 끝 제외 가능)."""
        dy = abs(y1 - y0)
        dx = abs(x1 - x0)
        sy = 1 if y0 < y1 else -1
        sx = 1 if x0 < x1 else -1
        err = dx - dy
        y, x = y0, x0
        cells = [(y, x)]
        while not (y == y1 and x == x1):
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy
            cells.append((y, x))
        return cells

    def is_occluded_by_static(self, origin_rc: np.ndarray, target_rc: np.ndarray, exclude_ends=True) -> bool:
        """origin_rc→target_rc 선분 경로 중 정적 장애물(1)이 끼어 있으면 True."""
        y0, x0 = int(round(origin_rc[0])), int(round(origin_rc[1]))
        y1, x1 = int(round(target_rc[0])), int(round(target_rc[1]))
        pts = self._bresenham_cells(y0, x0, y1, x1)
        if exclude_ends and len(pts) >= 2:
            pts = pts[1:-1]
        for (yy, xx) in pts:
            if not (0 <= yy < self.H and 0 <= xx < self.W):
                return True  # 맵 밖은 벽처럼 취급
            if self.grid[yy, xx] == 1:
                return True
        return False

    # ----------------------- 라이다(정적만) -----------------------
    def _raycast_static(self, origin_yx, theta, max_range_m):
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
    def _distance_to_nearest_obj_m(self, visible_only: bool = True):
        if not hasattr(self, "dynamic_objs") or len(self.dynamic_objs) == 0:
            return float("inf")
        ry, rx = self.agent_rc
        best = float("inf")
        for obj in self.dynamic_objs:
            if visible_only and self.consider_occlusion_in_mode:
                if self.is_occluded_by_static(self.agent_rc, obj.p):
                    continue
            d_cells = float(np.hypot(obj.p[0] - ry, obj.p[1] - rx))
            best = min(best, d_cells)
        if best == float("inf"):
            return float("inf")
        return self._cells_to_m(best)

    # ----------------------- 이동 -----------------------
    def _move_toward(self, wp_xy):
        tx, ty = wp_xy
        y, x = self.agent_rc
        dy = np.sign(float(ty) - float(y))
        dx = np.sign(float(tx) - float(x))
        new_y = int(y + dy)
        new_x = int(x + dx)
        if self._is_free(new_y, new_x):
            self.agent_rc = np.array([new_y, new_x], dtype=float)
            self.covered_mask[int(new_y), int(new_x)] = True
        if dy == -1 and dx == 0: return 0
        if dy == 0 and dx == -1: return 1
        if dy == 1 and dx == 0: return 2
        if dy == 0 and dx == 1: return 3
        return 4

    def _move_agent(self, action):
        if action is None:
            return True
        dy, dx = {0:(-1,0), 1:(0,-1), 2:(1,0), 3:(0,1), 4:(0,0)}[int(action)]
        ny = int(self.agent_rc[0] + dy)
        nx = int(self.agent_rc[1] + dx)
        if self._is_free(ny, nx):
            self.agent_rc = np.array([ny, nx], dtype=float)
            self.covered_mask[int(ny), int(nx)] = True
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

        # 목표(relative)
        dy_cells = gy - ry
        dx_cells = gx - rx
        d_goal_m = self._cells_to_m(np.hypot(dy_cells, dx_cells))
        angle_goal = np.arctan2(dy_cells, dx_cells)
        goal_feats = np.array([
            min(d_goal_m, self.R_goal_m) / self.R_goal_m,
            np.cos(angle_goal),
            np.sin(angle_goal)
        ], dtype=np.float32)

        # 동적 객체 관측(가림막 고려)
        per_obj_feats = []
        objs_metrics = []
        visible_ids = set()
        if hasattr(self, "dynamic_objs") and len(self.dynamic_objs) > 0:
            for obj in self.dynamic_objs:
                if self.consider_occlusion_in_obs and self.is_occluded_by_static(self.agent_rc, obj.p):
                    continue  # 보이지 않으면 관측에서 제외
                visible_ids.add(id(obj))
                oy, ox = obj.p
                move_vec = self._obj_move_vec_cells(obj)
                ovy, ovx = move_vec
                dist_cells = float(np.hypot(oy - ry, ox - rx))
                d_obj_m = self._cells_to_m(dist_cells)
                d_norm = min(d_obj_m, self.R_obj_m) / self.R_obj_m
                angle = float(np.arctan2(oy - ry, ox - rx))
                speed_mps = self._cells_to_m(np.hypot(ovx, ovy))
                speed_norm = min(speed_mps, self.vmax_obj) / self.vmax_obj
                # TTC
                rel_p_m = np.array([self._cells_to_m(ox - rx), self._cells_to_m(oy - ry)], dtype=np.float64)
                rel_v_mps = np.array([self._cells_to_m(ovx), self._cells_to_m(ovy)], dtype=np.float64)
                denom = float(np.dot(rel_v_mps, rel_v_mps))
                if denom > 1e-9:
                    ttc = max(0.0, -float(np.dot(rel_p_m, rel_v_mps)) / denom)
                else:
                    ttc = self.T_cap_s
                ttc_capped = float(min(ttc, self.T_cap_s))
                ttc_norm = ttc_capped / self.T_cap_s
                objs_metrics.append((ttc_capped >= self.T_cap_s - 1e-9, ttc_capped, dist_cells, angle, speed_norm, d_norm, ttc_norm))

            # 위협도 정렬: TTC 미캡 우선 → TTC → 거리
            objs_metrics.sort(key=lambda t: (t[0], t[1], t[2]))
            for j in range(min(self.max_objs, len(objs_metrics))):
                _, _, _, angle, speed_norm, d_norm, ttc_norm = objs_metrics[j]
                per_obj_feats.extend([
                    d_norm,
                    float(np.cos(angle)),
                    float(np.sin(angle)),
                    speed_norm,
                    ttc_norm,
                ])

        # 패딩
        while len(per_obj_feats) < self.obj_feat_len * self.max_objs:
            per_obj_feats.extend([1.0, 0.0, 0.0, 0.0, 1.0])

        # 라이다(정적)
        origin = (ry, rx)
        rays_m = [self._raycast_static(origin, ang, self.R_ray_m) for ang in self.ray_angles]
        rays   = np.array([r / self.R_ray_m for r in rays_m], dtype=np.float32)

        # 위험도 특징 + danger "라이다"
        danger_feats = np.zeros(self.danger_feat_len, dtype=np.float32)
        if self.danger_zone_map is not None:
            soft = getattr(self.danger_zone_map, "soft", None)
            if soft is not None:
                yy = int(round(ry))
                xx = int(round(rx))
                if 0 <= yy < soft.shape[0] and 0 <= xx < soft.shape[1]:
                    danger_here = float(np.clip(soft[yy, xx], 0.0, 1.0))
                    r = 3
                    y0 = max(0, yy - r)
                    y1 = min(soft.shape[0], yy + r + 1)
                    x0 = max(0, xx - r)
                    x1 = min(soft.shape[1], xx + r + 1)
                    patch = soft[y0:y1, x0:x1]
                    danger_near = float(np.clip(np.max(patch), 0.0, 1.0)) if patch.size > 0 else danger_here
                    danger_feats[:] = [danger_here, danger_near]
        danger_lidar = self._danger_lidar(bins=getattr(self, "danger_lidar_bins", 16), max_range=6.0)
        # 로그용 캐시
        self._last_danger_feats = danger_feats.copy()
        self._last_danger_lidar = danger_lidar.copy()

        # 캐시
        self._last_d_goal_m = float(d_goal_m)
        self._last_goal_angle_math = self._goal_angle_math(dy_cells, dx_cells)
        self._last_rays = rays.copy()
        if len(objs_metrics) > 0:
            self._last_ttc = float(objs_metrics[0][-1])
        else:
            self._last_ttc = 1.0
        self.visible_obj_ids = visible_ids

        obs = np.concatenate([goal_feats, np.array(per_obj_feats, dtype=np.float32), rays, danger_feats, danger_lidar], axis=0)
        assert obs.shape[0] == self.obs_dim
        return obs

    # ----------------------- 경로 방해 유틸 -----------------------
    @staticmethod
    def _circle_overlap(c1, r1, c2, r2) -> bool:
        dx = c1[0] - c2[0]
        dy = c1[1] - c2[1]
        return (dx*dx + dy*dy) <= (r1 + r2) * (r1 + r2)

    def _goal_xy_m(self):
        gx, gy = self.waypoints[self.wp_idx]
        return np.array([self._cells_to_m(gx), self._cells_to_m(gy)], dtype=float)

    def _obj_xy_m(self, obj: MovingObj):
        return np.array([self._cells_to_m(obj.p[1]), self._cells_to_m(obj.p[0])], dtype=float)

    @staticmethod
    def _obj_move_vec_cells(obj: MovingObj):
        vec = getattr(obj, "last_move", None)
        if vec is None:
            vec = obj.v
        return np.array(vec, dtype=float)

    def _static_density(self, center_rc, radius_cells=3.0):
        r = int(np.ceil(radius_cells))
        y0 = max(0, int(np.floor(center_rc[0] - r)))
        y1 = min(self.H, int(np.ceil(center_rc[0] + r + 1)))
        x0 = max(0, int(np.floor(center_rc[1] - r)))
        x1 = min(self.W, int(np.ceil(center_rc[1] + r + 1)))
        patch = self.grid[y0:y1, x0:x1]
        if patch.size == 0:
            return 0.0
        return float(np.mean(patch == 1))

    def _compute_block_metrics(self, goal_xy_cells):
        goal_rc = np.array([goal_xy_cells[1], goal_xy_cells[0]], dtype=float)
        severity_dyn = 0.0
        if len(self.dynamic_objs) > 0:
            for obj in self.dynamic_objs:
                move_vec = self._obj_move_vec_cells(obj)
                for t in range(0, self.block_horizon + 1):
                    pred = obj.p + move_vec * t
                    dist = float(np.linalg.norm(goal_rc - pred))
                    if dist <= self.block_dyn_radius_cells:
                        sev = 1.0 - (dist / max(self.block_dyn_radius_cells, 1e-6))
                        severity_dyn = max(severity_dyn, sev)
                        break
        static_density = self._static_density(self.agent_rc, radius_cells=3.0)
        severity = max(severity_dyn, static_density)
        return {
            "severity": severity,
            "dyn_severity": severity_dyn,
            "static_density": static_density,
        }

    def _update_obj_histories(self):
        self.danger_zone_map.decay_step()
        active = set()
        for obj in getattr(self, "dynamic_objs", []):
            key = id(obj)
            active.add(key)
            hist = self.obj_histories.get(key)
            if hist is None:
                hist = deque(maxlen=self.obj_history_len)
                self.obj_histories[key] = hist
            hist.append(np.array(obj.p, dtype=float))
        for key in list(self.obj_histories.keys()):
            if key not in active:
                del self.obj_histories[key]
        # danger_regions를 매 스텝 다시 찍어서 객체가 있는 동안 위험도가 사라지지 않게 유지
        if getattr(self, "danger_regions", None):
            self._rebuild_danger_map()

    def _follow_override_path(self):
        if not self.override_path:
            return None
        # 제거된 목표 지점 정리
        while self.override_path:
            target_xy = self.override_path[0]
            dist = np.hypot(self.agent_rc[1] - target_xy[0], self.agent_rc[0] - target_xy[1])
            if dist < 0.3:
                self.override_path.popleft()
            else:
                break
        if not self.override_path:
            self.deviated_from_cpp = False
            self.override_target = None
            self.override_path_full = None
            return None
        target_xy = self.override_path[0]
        action = self._move_toward(target_xy)
        if np.hypot(self.agent_rc[1] - target_xy[0], self.agent_rc[0] - target_xy[1]) < 0.3 and self.override_path:
            self.override_path.popleft()
        if not self.override_path:
            self.deviated_from_cpp = False
            self.override_target = None
            self.override_path_full = None
        return action

    def _plan_path_with_mask(self, mask, goal_xy_cells, start_rc=None):
        if start_rc is None:
            start_rc = self.agent_rc
        start = (int(round(start_rc[0])), int(round(start_rc[1])))
        goal = (int(round(goal_xy_cells[1])), int(round(goal_xy_cells[0])))
        if not (0 <= goal[0] < self.H and 0 <= goal[1] < self.W):
            return None
        mask = mask.copy()
        mask[start[0], start[1]] = False
        path_rc = self._bfs_path(start, goal, mask)
        if not path_rc or len(path_rc) < 2:
            return None
        # 경로를 (x,y) 좌표로 저장
        converted = []
        for r, c in path_rc[1:]:
            converted.append((float(c), float(r)))
        return converted

    def _find_path_to_unvisited(self, start_rc=None):
        if start_rc is None:
            start_rc = self.agent_rc

        sr = int(round(start_rc[0]))
        sc = int(round(start_rc[1]))
        if not (0 <= sr < self.H and 0 <= sc < self.W):
            return None

        danger_mask = self._danger_mask()
        target_mask = (~self.covered_mask) & (~danger_mask) & (self.grid != 1)
        if not np.any(target_mask):
            return None

        blocked = (self.grid == 1) | danger_mask
        blocked = blocked.copy()
        blocked[sr, sc] = False

        q = deque([(sr, sc)])
        came = {(sr, sc): None}
        found = None

        while q:
            r, c = q.popleft()
            if target_mask[r, c]:
                found = (r, c)
                break
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr, nc = r + dr, c + dc
                if not (0 <= nr < self.H and 0 <= nc < self.W):
                    continue
                if blocked[nr, nc] or (nr, nc) in came:
                    continue
                came[(nr, nc)] = (r, c)
                q.append((nr, nc))

        if found is None:
            return None

        path_rc = []
        cur = found
        while cur is not None:
            path_rc.append(cur)
            cur = came[cur]
        path_rc.reverse()

        return [(float(c), float(r)) for r, c in path_rc]

    def _bfs_path(self, start, goal, mask):
        if mask[start[0], start[1]]:
            return None
        H, W = mask.shape
        q = deque([start])
        came = {start: None}
        while q:
            r, c = q.popleft()
            if (r, c) == goal:
                break
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr, nc = r + dr, c + dc
                if not (0 <= nr < H and 0 <= nc < W):
                    continue
                if mask[nr, nc]:
                    continue
                if (nr, nc) in came:
                    continue
                came[(nr, nc)] = (r, c)
                q.append((nr, nc))
        if goal not in came:
            return None
        path = []
        cur = goal
        while cur is not None:
            path.append(cur)
            cur = came[cur]
        path.reverse()
        return path

    def _compute_cpp_mask(self, start_rc):
        mask = ((self.grid == 1) | self.covered_mask).copy()
        mask |= self._danger_mask()
        sr = int(round(start_rc[0]))
        sc = int(round(start_rc[1]))
        if 0 <= sr < self.H and 0 <= sc < self.W:
            mask[sr, sc] = False
        return mask

    def _clear_danger_along_path(self, path, radius_cells=1.5):
        if not path or self.danger_zone_map is None:
            return
        soft = self.danger_zone_map.soft
        H, W = soft.shape
        for x, y in path:
            cy = float(y)
            cx = float(x)
            r = float(radius_cells)
            y0 = max(0, int(np.floor(cy - r - 1)))
            y1 = min(H, int(np.ceil(cy + r + 2)))
            x0 = max(0, int(np.floor(cx - r - 1)))
            x1 = min(W, int(np.ceil(cx + r + 2)))
            if y0 >= y1 or x0 >= x1:
                continue
            yy, xx = np.ogrid[y0:y1, x0:x1]
            mask = (yy - cy)**2 + (xx - cx)**2 <= (r * r)
            region = soft[y0:y1, x0:x1]
            region[mask] = 0.0

    def _build_cpp_path(self, start_rc=None):
        if start_rc is None:
            start_rc = self.agent_rc
        components = self._unvisited_components()
        if not components:
            return []
        nav_mask = (self.grid == 1) | self._danger_mask()
        path = []
        curr = np.array(start_rc, dtype=float)

        for comp in components:
            comp_set = {(y, x) for (y, x) in comp}
            target = comp[0]
            curr_rc = (int(round(curr[0])), int(round(curr[1])))
            if curr_rc not in comp_set:
                bridge = self._plan_path_with_mask(nav_mask, (float(target[1]), float(target[0])), start_rc=curr)
                if not bridge:
                    continue
                if path and bridge[0] == path[-1]:
                    path.extend(bridge[1:])
                else:
                    path.extend(bridge)
                curr = np.array([bridge[-1][1], bridge[-1][0]], dtype=float)
                curr_rc = (int(round(curr[0])), int(round(curr[1])))
            if curr_rc not in comp_set:
                continue

            comp_mask = np.ones_like(self.grid, dtype=int)
            for (y, x) in comp_set:
                comp_mask[y, x] = 0
            comp_mask[curr_rc[0], curr_rc[1]] = 2
            try:
                planner = CoveragePlanner(comp_mask)
                planner.start()
                planner.compute()
                _, _, _, _, comp_traj = planner.result()
            except Exception:
                comp_traj = [(curr_rc[0], curr_rc[1])]
            comp_path = [(float(x), float(y)) for y, x in comp_traj]
            if not comp_path:
                continue
            if path and comp_path[0] == path[-1]:
                path.extend(comp_path[1:])
            else:
                path.extend(comp_path)
            curr = np.array([path[-1][1], path[-1][0]], dtype=float)

        return path

    def _apply_cpp_path(self, path_pts, *, clear_override=True, set_override_target=False):
        if not path_pts:
            return False
        new_wps = np.array([[p[0], p[1]] for p in path_pts], dtype=np.float32)
        if new_wps.ndim != 2 or new_wps.shape[1] != 2:
            return False
        self.waypoints = new_wps
        self.visited = np.zeros(len(self.waypoints), dtype=bool)
        self.wp_idx = 0
        self.deviated_from_cpp = False
        self.goal_stagnation_timer = 0
        if clear_override:
            self.override_path.clear()
            self.override_target = None
            self.override_path_full = None
        if set_override_target and len(self.waypoints) > 0:
            self.override_target = self.waypoints[0].copy()
        return True

    def _find_exit_path(self):
        if self.danger_zone_map is None:
            return None
        danger = self.danger_zone_map.hard
        sr = int(round(self.agent_rc[0]))
        sc = int(round(self.agent_rc[1]))
        if not (0 <= sr < self.H and 0 <= sc < self.W):
            return None
        if not danger[sr, sc]:
            return None
        blocked = (self.grid == 1)
        q = deque([(sr, sc)])
        came = {(sr, sc): None}
        target = None
        while q:
            r, c = q.popleft()
            if not danger[r, c]:
                target = (r, c)
                break
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr, nc = r + dr, c + dc
                if not (0 <= nr < self.H and 0 <= nc < self.W):
                    continue
                if blocked[nr, nc] or (nr, nc) in came:
                    continue
                came[(nr, nc)] = (r, c)
                q.append((nr, nc))
        if target is None:
            return None
        path = []
        cur = target
        while cur is not None:
            r, c = cur
            path.append((float(c), float(r)))
            cur = came[cur]
        path.reverse()
        return path

    def _danger_mask(self):
        if self.danger_zone_map is None:
            return np.zeros_like(self.grid, dtype=bool)
        soft = getattr(self.danger_zone_map, "soft", None)
        if soft is None:
            return np.zeros_like(self.grid, dtype=bool)
        thr = float(getattr(self, "danger_soft_block", 0.3))
        return soft >= thr

    def _unvisited_components(self):
        danger_mask = self._danger_mask()
        open_mask = (~self.covered_mask) & (~danger_mask) & (self.grid != 1)
        seen = np.zeros_like(open_mask, dtype=bool)
        comps = []
        for r in range(self.H):
            for c in range(self.W):
                if not open_mask[r, c] or seen[r, c]:
                    continue
                comp = []
                q = deque([(r, c)])
                seen[r, c] = True
                while q:
                    y, x = q.popleft()
                    comp.append((y, x))
                    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        ny, nx = y + dr, x + dc
                        if 0 <= ny < self.H and 0 <= nx < self.W and open_mask[ny, nx] and not seen[ny, nx]:
                            seen[ny, nx] = True
                            q.append((ny, nx))
                comps.append(comp)
        return comps

    def _stamp_danger_pts(self, pts):
        if self.danger_zone_map is None or not pts:
            return
        self.danger_zone_map.stamp_polyline(pts, radius_cells=1.2, val=0.9)
        for y, x in pts:
            for dy in range(-2, 3):
                for dx in range(-2, 3):
                    dist = max(abs(dy), abs(dx))
                    if dist == 0:
                        radius = 1.2; val = 0.95
                    elif dist == 1:
                        radius = 1.4; val = 0.9
                    else:
                        radius = 1.6; val = 0.8
                    self.danger_zone_map._stamp_disc(
                        self.danger_zone_map.soft,
                        y + dy,
                        x + dx,
                        radius,
                        val
                    )

    def _rebuild_danger_map(self):
        if self.danger_zone_map is None:
            return
        self.danger_zone_map.soft.fill(0.0)
        for pts in self.danger_regions.values():
            self._stamp_danger_pts(pts)
        # danger 생성 시점의 목표 좌표들을 3x3(반경≈1.0)으로 완만히 표시
        orphans = []
        for key, goals in self.danger_goal_points.items():
            if key not in self.danger_regions:
                orphans.append(key)
                continue
            for gy, gx in goals:
                gy_i = int(round(gy)); gx_i = int(round(gx))
                if 0 <= gy_i < self.H and 0 <= gx_i < self.W:
                    self.danger_zone_map._stamp_disc(self.danger_zone_map.soft, gy_i, gx_i, 1.5, 0.6)
            # 목표 지점이 같은 danger 영역과 이어지도록 centroid까지 연결
            pts = self.danger_regions.get(key, [])
            if pts:
                centroid = np.mean(np.array(pts, dtype=float), axis=0)
                for gy, gx in goals:
                    steps = max(2, int(np.hypot(centroid[0] - gy, centroid[1] - gx)))
                    for t in np.linspace(0.0, 1.0, steps):
                        cy = gy + (centroid[0] - gy) * t
                        cx = gx + (centroid[1] - gx) * t
                        self.danger_zone_map._stamp_disc(self.danger_zone_map.soft, cy, cx, 0.8, 0.6)
        # 생성 시점에 저장된 goal→danger 링크 재적용 (goal 변화와 무관)
        for line in self.danger_goal_links.values():
            for cy, cx in line:
                self.danger_zone_map._stamp_disc(self.danger_zone_map.soft, cy, cx, 0.4, 0.7)
        # 고아 목표 제거
        for key in orphans:
            self.danger_goal_points.pop(key, None)
            self.danger_goal_links.pop(key, None)

    def _cleanup_danger_regions(self):
        current = {id(obj): obj for obj in getattr(self, "dynamic_objs", [])}
        release_radius = float(getattr(self, "danger_release_radius", 3.0))
        release_steps = int(getattr(self, "danger_release_steps", 5))
        removed = False
        for key in list(self.danger_regions.keys()):
            obj = current.get(key)
            if obj is None:
                del self.danger_regions[key]
                self.danger_goal_links.pop(key, None)
                self.danger_goal_points.pop(key, None)
                removed = True
                self._danger_release_timers.pop(key, None)
                continue
            pts = np.array(self.danger_regions[key], dtype=float)
            if pts.size == 0:
                del self.danger_regions[key]
                self.danger_goal_links.pop(key, None)
                self.danger_goal_points.pop(key, None)
                removed = True
                self._danger_release_timers.pop(key, None)
                continue
            # 해제 판단: 해당 danger 클러스터(pts_all)와 객체 거리 기반
            pts_all = np.array(self.danger_regions[key], dtype=float)
            if pts_all.size == 0:
                del self.danger_regions[key]
                self.danger_goal_links.pop(key, None)
                self.danger_goal_points.pop(key, None)
                removed = True
                self._danger_release_timers.pop(key, None)
                continue
            dmin = float(np.min(np.linalg.norm(pts_all - np.array(obj.p, dtype=float), axis=1)))
            if dmin > release_radius:
                cnt = self._danger_release_timers.get(key, 0) + 1
                if cnt >= release_steps:
                    del self.danger_regions[key]
                    self.danger_goal_links.pop(key, None)
                    self.danger_goal_points.pop(key, None)
                    removed = True
                    self._danger_release_timers.pop(key, None)
                else:
                    self._danger_release_timers[key] = cnt
            else:
                self._danger_release_timers[key] = 0
        # danger_regions와 goal/링크 정합성 유지: orphan된 goal/링크 제거
        orphan_removed = False
        for key in list(self.danger_goal_points.keys()):
            if key not in self.danger_regions:
                self.danger_goal_points.pop(key, None)
                self.danger_goal_links.pop(key, None)
                orphan_removed = True
        # 동적 객체가 전혀 없는 경우에도 모든 danger를 정리
        if len(current) == 0 and (self.danger_regions or self.danger_goal_points or self.danger_goal_links):
            self.danger_regions.clear()
            self.danger_goal_points.clear()
            self.danger_goal_links.clear()
            orphan_removed = True

        if removed or orphan_removed:
            self._rebuild_danger_map()
            # danger 해제 시점에 항상 CPP를 새로 짜서 해금된 영역을 커버하게 함
            new_cpp = self._build_cpp_path(start_rc=self.agent_rc)
            if new_cpp:
                self._last_replan_reason = "danger_cleanup"
                self._apply_cpp_path(new_cpp)

    def _replan_after_danger_change(self):
        # use_escape_subpolicy가 켜져 있으면 ESCAPE 종료 시점에만 CPP를 재계획한다.
        if getattr(self, "use_escape_subpolicy", False):
            return
        path_cpp = self._build_cpp_path(start_rc=self.agent_rc)
        if not path_cpp:
            return
        self._last_replan_reason = "danger_cleanup"
        self._apply_cpp_path(path_cpp)

    def _handle_stuck(self, goal_xy_cells):
        if self.use_escape_subpolicy:
            # danger 영역만 갱신하고 sub-policy에 제어권을 넘김
            seen_ids = getattr(self, "avoid_seen_obj_ids", set())
            updated = False
            new_links = {}
            for obj in getattr(self, "dynamic_objs", []):
                if seen_ids and id(obj) not in seen_ids:
                    continue
                dr = float(np.linalg.norm(obj.p - self.agent_rc))
                if dr > getattr(self, "danger_stuck_radius", 8.0):
                    continue
                hist = self.obj_histories.get(id(obj))
                key = id(obj)
                if hist and len(hist) >= 2:
                    if key not in self.danger_regions:
                        pts = [(float(p[0]), float(p[1])) for p in hist]
                        self.danger_regions[key] = pts
                        updated = True
                    # 목표 스냅샷/링크는 stuck가 다시 발생해도 추가로 기록
                    if self.wp_idx < len(self.waypoints):
                        gx, gy = self.waypoints[self.wp_idx]
                        goals = self.danger_goal_points.get(key, [])
                        added_goal = False
                        if all(np.hypot(gy - g[0], gx - g[1]) > 0.25 for g in goals):
                            goals.append((gy, gx))
                            self.danger_goal_points[key] = goals
                            added_goal = True
                        if added_goal or key not in self.danger_goal_links:
                            pts = [(float(p[0]), float(p[1])) for p in hist] if key not in self.danger_regions else self.danger_regions[key]
                            if pts:
                                centroid = np.mean(np.array(pts, dtype=float), axis=0)
                                steps = max(2, int(np.hypot(centroid[0] - gy, centroid[1] - gx)))
                                line = []
                                for t in np.linspace(0.0, 1.0, steps):
                                    cy = gy + (centroid[0] - gy) * t
                                    cx = gx + (centroid[1] - gx) * t
                                    line.append((cy, cx))
                                new_links[key] = line
                    if key in self.danger_regions:
                        updated = True
            if new_links:
                self.danger_goal_links.update(new_links)
            if updated:
                self._rebuild_danger_map()
            # danger zone이 없어도 stuck이면 ESCAPE를 켜 주자
            hard = None
            if self.danger_zone_map is not None:
                hard = getattr(self.danger_zone_map, "hard", None)
            in_danger = self._agent_inside_danger()
            near_danger = False
            if hard is not None and np.any(hard):
                ay = int(round(self.agent_rc[0])); ax = int(round(self.agent_rc[1]))
                idx = np.argwhere(hard)
                if idx.size > 0:
                    d = np.min(np.linalg.norm(idx - np.array([ay, ax]), axis=1))
                    near_danger = d <= 2.0
            if in_danger or near_danger or updated:
                self.escape_active = True
                self._escape_release_counter = 0
            return

        seen_ids = getattr(self, "avoid_seen_obj_ids", set())
        updated = False
        new_links = {}
        for obj in getattr(self, "dynamic_objs", []):
            if seen_ids and id(obj) not in seen_ids:
                continue
            dr = float(np.linalg.norm(obj.p - self.agent_rc))
            if dr > getattr(self, "danger_stuck_radius", 8.0):
                continue
            hist = self.obj_histories.get(id(obj))
            key = id(obj)
            if hist and len(hist) >= 2:
                pts_new = [(float(p[0]), float(p[1])) for p in hist]
                if key in self.danger_regions:
                    self.danger_regions[key].extend(pts_new)
                else:
                    self.danger_regions[key] = pts_new
                updated = True
                if self.wp_idx < len(self.waypoints):
                    gx, gy = self.waypoints[self.wp_idx]
                    goals = self.danger_goal_points.get(key, [])
                    if all(np.hypot(gy - g[0], gx - g[1]) > 0.25 for g in goals):
                        goals.append((gy, gx))
                        self.danger_goal_points[key] = goals
                    pts_all = self.danger_regions.get(key, pts_new)
                    if pts_all:
                        centroid = np.mean(np.array(pts_all, dtype=float), axis=0)
                        steps = max(2, int(np.hypot(centroid[0] - gy, centroid[1] - gx)))
                        line = []
                        for t in np.linspace(0.0, 1.0, steps):
                            cy = gy + (centroid[0] - gy) * t
                            cx = gx + (centroid[1] - gx) * t
                            line.append((cy, cx))
                        new_links[key] = line

        if new_links:
            self.danger_goal_links.update(new_links)
        if updated:
            self._rebuild_danger_map()

        exit_path = self._find_exit_path()
        exit_queue = deque()
        start_for_cpp = self.agent_rc.copy()
        if exit_path and len(exit_path) > 1:
            exit_queue.extend(exit_path[1:])
            last = exit_path[-1]
            start_for_cpp = np.array([last[1], last[0]], dtype=float)

        bridge_path = self._find_path_to_unvisited(start_for_cpp)
        bridge_queue = deque()
        if bridge_path and len(bridge_path) > 1:
            bridge_queue.extend(bridge_path[1:])
            last = bridge_path[-1]
            start_for_cpp = np.array([last[1], last[0]], dtype=float)

        path_cpp = self._build_cpp_path(start_rc=start_for_cpp)
        if not path_cpp:
            mask = ((self.grid == 1) | self.covered_mask)
            if self.danger_zone_map is not None:
                mask |= self.danger_zone_map.hard
            fallback = self._plan_path_with_mask(mask, goal_xy_cells, start_rc=start_for_cpp)
            if fallback:
                path_cpp = fallback

        if path_cpp:
            assigned = self._apply_cpp_path(path_cpp, clear_override=False, set_override_target=True)
            if not assigned:
                path_cpp = None
            else:
                self._last_replan_reason = "stuck_replan"

        combined_override = deque()
        if exit_queue:
            combined_override.extend(exit_queue)
            self.override_path_full = list(exit_path)
        else:
            self.override_path_full = None
        if bridge_queue:
            combined_override.extend(bridge_queue)
        self.override_path = combined_override

    def _update_stuck_state(self, goal_xy_cells, info):
        metrics = {
            "block_level": float(info.get("block_avg", info.get("block_severity", 0.0))),
            "efficiency": float(info.get("efficiency", 0.0)),
            "prog_ratio": float(info.get("prog_ratio", 0.0)),
            "move_mean_m": float(info.get("move_mean_m", 0.0)),
        }
        self._stuck_hist.append(metrics)
        self._stuck_pos_hist.append(self.agent_rc.copy())

        if len(self._stuck_hist) < max(2, self.stuck_window // 2):
            self._stuck_state = False
            return

        block_avg = float(np.mean([h["block_level"] for h in self._stuck_hist]))
        mean_eff = float(np.mean([h["efficiency"] for h in self._stuck_hist]))
        mean_prog = float(np.mean([h["prog_ratio"] for h in self._stuck_hist]))
        mean_move = float(np.mean([h["move_mean_m"] for h in self._stuck_hist]))

        stagnating = False
        if len(self._stuck_pos_hist) >= 2:
            disp_cells = float(np.linalg.norm(self._stuck_pos_hist[-1] - self._stuck_pos_hist[0]))
            stagnating = disp_cells <= self.stuck_stagnation_radius_cells

        goal_dist_cells = float(np.linalg.norm(goal_xy_cells - self.agent_rc))
        far_from_goal = goal_dist_cells >= self.stuck_goal_dist_min_cells

        conds = {
            "block": block_avg >= self.stuck_block_min,
            "prog": mean_prog <= self.stuck_progress_min,
            "eff": mean_eff <= self.stuck_eff_min,
            "move": mean_move <= self.stuck_move_min_m,
        }
        satisfied = sum(1 for v in conds.values() if v)

        def _clip(v):
            return float(np.clip(v, 0.0, 1.0))

        block_score = _clip((block_avg - self.stuck_block_min) / max(1.0 - self.stuck_block_min, 1e-6))
        prog_score = _clip((self.stuck_progress_min - mean_prog) / max(self.stuck_progress_min, 1e-6))
        eff_score = _clip((self.stuck_eff_min - mean_eff) / max(self.stuck_eff_min, 1e-6))
        move_score = _clip((self.stuck_move_min_m - mean_move) / max(self.stuck_move_min_m, 1e-6))
        score_sum = block_score + prog_score + eff_score + move_score

        # 보수적 로직: 4개 중 3개 이상 + 점수 상향 + 정체 + 목표까지 거리 조건
        self._stuck_state = (
            satisfied >= 3 and
            score_sum >= (self.stuck_score_min + 0.3) and
            stagnating and
            far_from_goal
        )
        # 추가로 일정 스텝(기본 18) 목표 미도달 시 stuck 처리
        if self.goal_stagnation_timer >= getattr(self, "stuck_goal_miss_steps", 23):
            self._stuck_state = True

    def _motion_metrics(self, goal_xy_cells, zero_tol=1e-6, jitter_tol_cells=0.25):
        v = np.array(self.rb_hist_vec, dtype=float)
        k = len(v)
        if k == 0:
            return 0.0, 0.0, 0.0, 0.0, 0.0

        mags = np.linalg.norm(v, axis=1)
        seg_len = mags.sum()
        net_vec = v.sum(axis=0)
        eff = float(np.linalg.norm(net_vec) / (seg_len + 1e-6))

        back = 0
        for i in range(1, k):
            a, b = v[i-1], v[i]
            if np.linalg.norm(a) > zero_tol and np.linalg.norm(b) > zero_tol:
                if np.linalg.norm(a + b) < 1e-6:
                    back += 1
        backrate = float(back / max(1, k-1))

        ry, rx = self.agent_rc
        gx, gy = goal_xy_cells
        g = np.array([gy - ry, gx - rx], dtype=float)
        g_norm = np.linalg.norm(g)
        if g_norm < 1e-9 or seg_len < 1e-9:
            prog_ratio = 0.0
        else:
            g_hat = g / g_norm
            proj = float((v @ g_hat).sum())
            prog_ratio = float(proj / seg_len)

        idle_ratio   = float((mags <= zero_tol).mean())
        jitter_mask  = (mags > zero_tol) & (mags <= jitter_tol_cells)
        jitter_ratio = float(jitter_mask.mean())

        return eff, backrate, prog_ratio, idle_ratio, jitter_ratio

    # ----------------------- reset -----------------------
    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.dynamic_objs = self._default_spawn(self.grid, self.waypoints, self.rng, v_robot=1.0)

        self.wp_idx   = 0
        self.steps    = 0
        self.deviated_from_cpp = False
        self.prev_obj_dist_m   = None
        self.prev_agent_rc     = self.agent_rc.copy()
        self.avoiding          = False
        self.visited           = np.zeros(len(self.waypoints), dtype=bool)

        self.block_hist.clear()
        self.rb_hist_vec.clear()
        self._stuck_hist.clear()
        self._stuck_pos_hist.clear()
        self._stuck_state = False
        self._prev_stuck_state = False
        self._last_replan_reason = None

        self.danger_zone_map.clear()
        self.obj_histories.clear()
        self.override_path.clear()
        self.override_target = None
        self.override_path_full = None
        self.covered_mask.fill(False)
        self.covered_mask[int(self.agent_rc[0]), int(self.agent_rc[1])] = True
        self.visible_obj_ids.clear()
        self.avoid_seen_obj_ids.clear()
        self.danger_regions.clear()
        self.danger_goal_points.clear()
        self.danger_goal_links.clear()
        self._danger_release_timers.clear()
        self.goal_stagnation_timer = 0

        self._prev_goal_dist_cells = None

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
                    vmax_use = max(v_obj_range[1], 1.2)
                    if kind == "ou":
                        vmag = rng.uniform(0.3, 0.6); vmax_use = 0.8
                    theta = rng.uniform(0, 2*np.pi)
                    vy, vx = np.sin(theta)*vmag, np.cos(theta)*vmag
                    obj = MovingObj(pos=(float(sy), float(sx)),
                                    vel=np.array([vy, vx], float),
                                    vmax=vmax_use,
                                    kind=kind,
                                    seed=int(rng.integers(1e9)))
                    if kind == "ou":
                        obj.sigma = 0.5; obj.theta = 1.2
                    if kind == "patrol":
                        pts = []
                        if raw_kind == "circle":
                            r_cells = int(rng.integers(3, 6))
                            for ang in np.linspace(0, 2*np.pi, 8, endpoint=False):
                                py = int(round(sy + r_cells * np.sin(ang)))
                                px = int(round(sx + r_cells * np.cos(ang)))
                                if 0 <= py < H and 0 <= px < W and occ_grid[py, px] == 0:
                                    pts.append((py, px))
                        else:
                            amp = int(rng.integers(2, 4))
                            step = int(rng.integers(2, 4))
                            for k in range(6):
                                px = int(sx + k * step)
                                py = int(sy + (amp if (k % 2 == 0) else -amp))
                                if 0 <= py < H and 0 <= px < W and occ_grid[py, px] == 0:
                                    pts.append((py, px))
                        if len(pts) < 3:
                            cand = [(sy, sx), (sy, min(W-2, sx+3)), (min(H-2, sy+3), min(W-2, sx+3)), (min(H-2, sy+3), sx)]
                            pts = [(py, px) for (py, px) in cand if 0 <= py < H and 0 <= px < W and occ_grid[py, px] == 0]
                        if len(pts) >= 2:
                            obj.set_patrol([(float(py), float(px)) for (py, px) in pts])
                    objs.append(obj)
                    created = True
                    break
            if not created:
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
        self._update_obj_histories()
        self._cleanup_danger_regions()

        # 객체까지 거리(가림막 고려하여 모드 판단)
        dist_to_obj_m = self._distance_to_nearest_obj_m(visible_only=self.consider_occlusion_in_mode)
        dist_to_obj_cells = dist_to_obj_m / self.cell_size_m if np.isfinite(dist_to_obj_m) else float("inf")

        COLLISION = self.collision_cells
        DANGER    = self.danger_cells
        SAFE      = self.safe_cells

        mode = "AVOID" if dist_to_obj_cells < DANGER else "FOLLOW_CPP"
        if self.override_path:
            mode = "FOLLOW_CPP"
        if self.use_escape_subpolicy and getattr(self, "escape_active", False):
            mode = "ESCAPE"
        info["mode"] = mode
        info["override_active"] = bool(self.override_path)

        executed_action = 4  # 정지(로깅)

        # ---------------- FOLLOW_CPP ----------------
        if mode == "FOLLOW_CPP":
            used_override = False
            if self.override_path:
                action_override = self._follow_override_path()
                if action_override is not None:
                    executed_action = action_override
                    reward += self.follow_step_r
                    used_override = True

            if not used_override and getattr(self, "deviated_from_cpp", False):
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

            if not used_override and self.wp_idx < len(self.waypoints):
                executed_action = self._move_toward(self.waypoints[self.wp_idx])
                reward += self.follow_step_r  # 약간 상향

        # ---------------- AVOID ----------------
        else:
            self.avoiding = True
            moved_successfully = True
            if action_from_ppo is not None:
                executed_action = int(action_from_ppo)
                moved_successfully = self._move_agent(executed_action)
            if not moved_successfully:
                reward -= 0.25

            # 미래충돌 패널티 (강화)
            ry, rx = self.agent_rc
            future_pen = 0.0
            for obj in self.dynamic_objs:
                # 가려진 객체라도 미래 위치는 실제론 존재하지만, 관측 기반 회피를 원하면
                # 여기서도 occlusion 고려를 하고 싶으면 is_occluded_by_static 체크 추가 가능
                move_vec = self._obj_move_vec_cells(obj)
                future_pos = obj.p + move_vec * 5.0
                fdist_cells = float(np.linalg.norm(future_pos - np.array([ry, rx], dtype=float)))
                if fdist_cells < 7.5:
                    future_pen += (7.5 - fdist_cells) * self.future_pen_coef
            reward -= future_pen

            # 객체와의 거리 변화(칸 기준)
            prev_cells = (self.prev_obj_dist_m / self.cell_size_m) if (self.prev_obj_dist_m is not None and np.isfinite(self.prev_obj_dist_m)) else dist_to_obj_cells
            delta_cells = dist_to_obj_cells - prev_cells if np.isfinite(dist_to_obj_cells) else 0.0
            self.prev_obj_dist_m = dist_to_obj_m
            reward += self.delta_dist_coef * float(np.clip(delta_cells, -1.0, 1.0))

            # 충돌/위험/안전
            if np.isfinite(dist_to_obj_cells) and dist_to_obj_cells <= COLLISION:
                reward -= 2.0
                done = True
            elif np.isfinite(dist_to_obj_cells) and dist_to_obj_cells >= SAFE:
                reward += 0.3
                self.avoiding = False
            else:
                reward += self.avoid_base_pen  # 기본 AVOID 페널티(완화)
                if np.isfinite(dist_to_obj_cells):
                    closeness_norm = (DANGER - dist_to_obj_cells) / max(DANGER, 1e-6)
                    if closeness_norm > 0:
                        reward -= self.prox_pen_scale * (closeness_norm ** self.prox_pen_pow)
                    # 매우 근접(충돌 직전) 시 추가 강한 패널티
                    if dist_to_obj_cells < 1.5:
                        extra_close_pen = (1.5 - dist_to_obj_cells) * 0.6  # 최대 약 0.9 감점
                        reward -= extra_close_pen

            # 목표 진행도 보상(AVOID 중에도 전진 유도)
            gx, gy = self.waypoints[self.wp_idx] if self.wp_idx < len(self.waypoints) else self.waypoints[-1]
            goal_dist_cells = float(np.hypot(gy - ry, gx - rx))
            if self._prev_goal_dist_cells is None:
                self._prev_goal_dist_cells = goal_dist_cells
            prog_delta = self._prev_goal_dist_cells - goal_dist_cells
            clipped_delta = float(np.clip(prog_delta, -1.0, 1.0))
            reward += self.progress_coef * clipped_delta
            if prog_delta > 1e-6:
                reward += self.progress_bonus * min(1.0, prog_delta)
            self._prev_goal_dist_cells = goal_dist_cells

            # 거의 안 움직였으면 소폭 패널티
            move_dist = float(np.linalg.norm(self.agent_rc - self.prev_agent_rc))
            if move_dist < 0.1:
                reward -= 0.1

            # 대충 막히면 CPP 재동기화 플래그
            # if self.wp_idx < len(self.waypoints):
            #     wx, wy = self.waypoints[self.wp_idx]
            #     ay, ax = self.agent_rc
            #     cpp_dist_cells = np.hypot(wy - ay, wx - ax)
            #     if cpp_dist_cells > 3.0:
            #         self.deviated_from_cpp = True

        # === 이동 벡터 기록 ===
        dyx = self.agent_rc - self.prev_agent_rc
        self.rb_hist_vec.append(dyx.copy())
        self.prev_agent_rc = self.agent_rc.copy()

        # === 경로 방해/정체 지표 ===
        if self.wp_idx < len(self.waypoints):
            goal_xy_cells = self.waypoints[self.wp_idx]
        else:
            goal_xy_cells = self.waypoints[-1]

        block_metrics = self._compute_block_metrics(goal_xy_cells)
        self.block_hist.append(block_metrics["severity"])
        block_avg = float(np.mean(self.block_hist)) if len(self.block_hist) > 0 else 0.0

        eff, backrate, prog_ratio, idle_ratio, jitter_ratio = self._motion_metrics(goal_xy_cells)
        move_mean_m = 0.0
        if len(self.rb_hist_vec) > 0:
            move_mean_m = float(np.mean(np.linalg.norm(np.array(self.rb_hist_vec), axis=1))) * self.cell_size_m

        info["block_rate"] = block_avg
        info["block_future_rate"] = block_metrics["dyn_severity"]
        info["block_severity"] = block_metrics["severity"]
        info["block_dyn_severity"] = block_metrics["dyn_severity"]
        info["block_static_density"] = block_metrics["static_density"]
        info["block_avg"] = block_avg
        info["move_mean_m"] = move_mean_m
        goal_xy_m = self._goal_xy_m()
        if len(self.dynamic_objs) > 0:
            d_now = min(np.linalg.norm(self._obj_xy_m(o) - goal_xy_m) for o in self.dynamic_objs)
            info["goal_obj_dist_now_m"] = float(d_now)
        else:
            info["goal_obj_dist_now_m"] = None
        info["efficiency"] = eff
        info["backtrack_rate"] = backrate
        info["prog_ratio"] = prog_ratio
        info["last_replan_reason"] = self._last_replan_reason
        # 디버그용 danger 관측 로그(마지막 obs 기준)
        info["danger_here"] = float(self._last_danger_feats[0]) if self._last_danger_feats is not None else 0.0
        info["danger_near"] = float(self._last_danger_feats[1]) if self._last_danger_feats is not None else 0.0
        info["danger_lidar_max"] = float(np.max(self._last_danger_lidar)) if self._last_danger_lidar is not None else 0.0
        info["ray_min"] = float(np.min(self._last_rays)) if self._last_rays is not None else 0.0
        info["ray_mean"] = float(np.mean(self._last_rays)) if self._last_rays is not None else 0.0
        info["ray_max"] = float(np.max(self._last_rays)) if self._last_rays is not None else 0.0
        info["escape_active"] = bool(getattr(self, "escape_active", False))

        prev_stuck = bool(self._stuck_state)
        self._update_stuck_state(np.array(goal_xy_cells, dtype=float), info)
        if self._stuck_state and not prev_stuck:
            self._handle_stuck(goal_xy_cells)
        self._prev_stuck_state = bool(self._stuck_state)
        info["stuck_state"] = bool(self._stuck_state)

        if mode == "AVOID":
            self.avoid_seen_obj_ids = set(self.visible_obj_ids)
        elif not self.override_path:
            self.avoid_seen_obj_ids.clear()

        # ESCAPE 모드 해제 조건: 위험 구역을 벗어나 일정 스텝 유지
        if self.use_escape_subpolicy and getattr(self, "escape_active", False):
            if not self._agent_inside_soft_danger():
                self._escape_release_counter += 1
                if self._escape_release_counter >= getattr(self, "escape_release_steps", 3):
                    self.escape_active = False
                    self._escape_release_counter = 0
                    new_cpp = self._build_cpp_path(start_rc=self.agent_rc.copy())
                    if new_cpp:
                        self._last_replan_reason = "escape_complete"
                        self._apply_cpp_path(new_cpp)
            else:
                self._escape_release_counter = 0

        # 웨이포인트 처리
        if self._reached_waypoint():
            self.goal_stagnation_timer = 0
            self.visited[self.wp_idx] = True
            reward += 0.4
            if self.wp_idx >= len(self.waypoints) - 1:
                done = True
                reward += 2.0
            else:
                self.wp_idx += 1
        else:
            self.goal_stagnation_timer += 1
            if self.goal_stagnation_timer >= self.goal_timeout_steps:
                self.goal_stagnation_timer = 0
                mask = (self.grid == 1) | self.covered_mask
                if self.danger_zone_map is not None:
                    mask |= self.danger_zone_map.hard
                new_path = self._plan_path_with_mask(mask, self.waypoints[-1])
                if new_path:
                    self._apply_cpp_path(new_path)

        # Danger zone 침범 패널티 (ESCAPE 모드일 땐 종료시키지 않음)
        skip_danger_penalty = self.use_escape_subpolicy and getattr(self, "escape_active", False)
        if self.danger_zone_map is not None and not skip_danger_penalty:
            yy = int(round(self.agent_rc[0]))
            xx = int(round(self.agent_rc[1]))
            if 0 <= yy < self.H and 0 <= xx < self.W:
                sev = float(self.danger_zone_map.soft[yy, xx])
                if sev >= self.danger_block_threshold:
                    # 기존: reward -= 2.0; done = True
                    # 변경: 즉시 종료하지 않고 패널티만 부여(강도를 낮춰 동적 객체 회피를 우선)
                    reward -= 0.5
        # ESCAPE 모드일 때: danger에서 벗어나면 추가 보상, 안에 머무르면 소폭 패널티
        if self.use_escape_subpolicy and getattr(self, "escape_active", False):
            inside_soft = self._agent_inside_soft_danger()
            if not inside_soft:
                reward += 0.5
            else:
                reward -= 0.05

        # 보상 클리핑 폭 확대
        reward = float(np.clip(reward, -2.5, 2.5))
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
