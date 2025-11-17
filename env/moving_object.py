# eclass MovingObj:
    
import numpy as np

class MovingObj:
    def __init__(self, pos, vel, vmax, kind="cv", seed=0, **kwargs):
        self.p = np.array(pos, dtype=float)  # (y, x)
        self.v = np.array(vel, dtype=float)  # (vy, vx)
        self.vmax = vmax
        self.kind = kind
        self.rng = np.random.default_rng(seed)
        self.t = 0
        self.last_move = np.zeros(2, dtype=float)

        # kwargs로 받은 추가 파라미터를 인스턴스 속성으로 저장
        self.amplitude = kwargs.get('amplitude', 5.0)
        self.frequency = kwargs.get('frequency', 0.2)
        self.radius = kwargs.get('radius', 5.0)
        self.rot_speed = kwargs.get('rot_speed', 0.2)
        self.center = self.p.copy() # 'circle'을 위한 중심점

        # 초기 속도 방향 저장 (sin 모델용)
        self.v_initial_norm = self.v / (np.linalg.norm(self.v) + 1e-9)

        # OU (Ornstein-Uhlenbeck) 파라미터
        self.theta = 1.5
        self.mu = np.zeros(2)
        self.sigma = 0.8

        # 순찰 경로용
        self.patrol_points = None
        self.patrol_idx = 0
        self.stop_timer = 0.0

    def set_patrol(self, points):
        """순찰 경로 설정"""
        self.patrol_points = [np.array(p, dtype=float) for p in points]
        self.patrol_idx = 0

    def _clip_v(self):
        n = np.linalg.norm(self.v)
        if n > self.vmax:
            self.v *= self.vmax / (n + 1e-9)

    def move(self, occ_grid, dt=1.0):
        """kind에 따라 움직임 결정"""
        if self.kind == "cv":  # constant velocity
            self._move_cv(occ_grid, dt)
        elif self.kind == "ou":
            self._move_ou(occ_grid, dt)
        elif self.kind == "patrol":
            self._move_patrol(occ_grid, dt)
        elif self.kind == "stopgo":
            self._move_stopgo(occ_grid, dt)
        else:
            self._move_cv(occ_grid, dt)

    # ============== 움직임 타입별 구현 ==============
    def _move_cv(self, occ_grid, dt):
        """등속 직선"""
        self._clip_v()
        prev = self.p.copy()
        p_next = self.p + self.v * dt
        if not self._collide(occ_grid, p_next):
            self.p = p_next
        else:
            self.v = -self.v  # 충돌 시 반사
        self.last_move = self.p - prev

    def _move_ou(self, occ_grid, dt):
        """Ornstein-Uhlenbeck 노이즈 기반 랜덤 움직임"""
        noise = self.theta * (self.mu - self.v) * dt + self.sigma * np.sqrt(dt) * self.rng.normal(size=2)
        self.v += noise
        self._move_cv(occ_grid, dt)

    def _move_patrol(self, occ_grid, dt):
        """지정된 순찰 경로를 따라 왕복"""
        if self.patrol_points is None:
            self._move_cv(occ_grid, dt)
            return

        target = self.patrol_points[self.patrol_idx]
        vec = target - self.p
        dist = np.linalg.norm(vec)
        if dist < 0.5:
            self.patrol_idx = (self.patrol_idx + 1) % len(self.patrol_points)
            self.last_move = np.zeros_like(self.last_move)
        else:
            self.v = 0.6 * vec / (dist + 1e-9)
            self._move_cv(occ_grid, dt)

    def _move_stopgo(self, occ_grid, dt):
        """정지-이동 랜덤 패턴"""
        if self.stop_timer > 0:
            self.stop_timer -= dt
            self.last_move = np.zeros_like(self.last_move)
            return
        if self.rng.random() < 0.05:
            self.stop_timer = self.rng.uniform(1.0, 3.0)
            self.last_move = np.zeros_like(self.last_move)
        else:
            self.v += self.rng.normal(scale=0.3, size=2)
            self._move_cv(occ_grid, dt)

    # =================================================
    def _collide(self, occ_grid, p_next):
        """다음 위치가 벽이면 True"""
        y, x = p_next
        H, W = occ_grid.shape
        r, c = int(round(y)), int(round(x))
        if r < 0 or r >= H or c < 0 or c >= W:
            return True
        return occ_grid[r, c] == 1
