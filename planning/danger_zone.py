# planning/danger_zone.py
import numpy as np
from collections import defaultdict, deque

try:
    from scipy.ndimage import gaussian_filter
    _HAS_SCI = True
except Exception:
    _HAS_SCI = False

try:
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except Exception:
    _HAS_MPL = False

class DangerZoneMap:
    """
    soft[y,x] ∈ [0,1]: 위험 확률/강도
    hard = soft >= hard_thr : 장애물처럼 취급
    decay_step()로 시간에 따라 자연 소거
    """
    def __init__(self, shape, decay=0.93, hard_thr=0.65, max_val=1.0):
        self.H, self.W = int(shape[0]), int(shape[1])
        self.soft = np.zeros((self.H, self.W), dtype=np.float32)
        self.decay = float(decay)
        self.hard_thr = float(hard_thr)
        self.max_val = float(max_val)

    @property
    def hard(self):
        return self.soft >= self.hard_thr

    def clear(self):
        self.soft.fill(0.0)

    def decay_step(self):
        self.soft *= self.decay

    def _stamp_disc(self, buf, cy, cx, r_cells, val):
        y0 = max(0, int(np.floor(cy - r_cells - 1)))
        y1 = min(self.H, int(np.ceil (cy + r_cells + 2)))
        x0 = max(0, int(np.floor(cx - r_cells - 1)))
        x1 = min(self.W, int(np.ceil (cx + r_cells + 2)))
        yy, xx = np.ogrid[y0:y1, x0:x1]
        mask = (yy - cy)**2 + (xx - cx)**2 <= (r_cells * r_cells)
        buf[y0:y1, x0:x1][mask] = np.maximum(buf[y0:y1, x0:x1][mask], val)

    def ingest_tracks(self, tracks, horizon=8, radius_cells=2.0, base_val=0.6, blur_sigma=1.0):
        """
        tracks: iterable of dicts
          {'pos':(y,x), 'vel':(vy,vx), 'los':True/False}
        horizon: 예측 스텝 수
        radius_cells: 위험 튜브 반경(셀)
        """
        buf = np.zeros_like(self.soft)
        H, W = self.H, self.W

        for tr in tracks:
            if tr.get('los', True) is False:
                continue
            y, x = map(float, tr.get('pos', (0,0)))
            vy, vx = map(float, tr.get('vel', (0,0)))
            for t in range(int(horizon)+1):
                cy = y + vy * t
                cx = x + vx * t
                if 0 <= cy < H and 0 <= cx < W:
                    # t가 멀수록 조금씩 약하게 (선택)
                    val = base_val * (0.85 ** t)
                    self._stamp_disc(buf, cy, cx, radius_cells, val)

        if blur_sigma and _HAS_SCI:
            buf = gaussian_filter(buf, blur_sigma)

        self.soft = np.clip(np.maximum(self.soft, buf), 0.0, self.max_val)

    def stamp_polyline(self, points, radius_cells=2.0, val=0.6):
        """
        points: iterable of (y,x) in grid coordinates
        """
        pts = np.array(points, dtype=float)
        if len(pts) == 0:
            return
        seg_samples = max(2, int(np.ceil(np.linalg.norm(pts[-1] - pts[0]))))
        for i in range(len(pts) - 1):
            a, b = pts[i], pts[i+1]
            steps = max(2, int(np.ceil(np.linalg.norm(b - a))))
            for t in np.linspace(0.0, 1.0, steps):
                cy, cx = a + (b - a) * t
                self._stamp_disc(self.soft, cy, cx, radius_cells, val)


class BehavioralDangerPlanner:
    """
    - 각 동적 객체의 최근 궤적을 기록하고 행동 패턴에 따라 위험 구역을 예측
    - DangerZoneMap을 지속적으로 업데이트하여 CPP가 사용할 마스크를 생성
    """
    def __init__(
        self,
        grid_shape,
        decay=0.93,
        hard_thr=0.65,
        history=20,
        base_radius=2.0,
        max_horizon=10,
    ):
        self.map = DangerZoneMap(grid_shape, decay=decay, hard_thr=hard_thr)
        self.histories = defaultdict(lambda: deque(maxlen=history))
        self.base_radius = float(base_radius)
        self.max_horizon = int(max_horizon)

    def reset(self):
        self.map.clear()
        self.histories.clear()

    def observe_objects(self, objects, los_mask=None):
        """
        objects: iterable of MovingObj (env.moving_object)
        los_mask: optional grid for line-of-sight (1=wall)
        """
        self.map.decay_step()
        for obj in objects:
            key = id(obj)
            hist = self.histories[key]
            hist.append(np.array(obj.p, dtype=float))
            behavior = self._classify(hist)
            speed = np.linalg.norm(getattr(obj, "last_move", obj.v))
            horizon = self._horizon_for(behavior, speed)
            radius = self._radius_for(behavior, speed)
            if behavior in ("patrol", "ou"):
                self.map.stamp_polyline(hist, radius_cells=radius * 1.1, val=0.55)
            tracks = self._sample_tracks(obj, hist, behavior, horizon, los_mask)
            if tracks:
                self.map.ingest_tracks(
                    tracks,
                    horizon=horizon,
                    radius_cells=radius,
                    base_val=0.6 if behavior != "stationary" else 0.4,
                    blur_sigma=1.0,
                )

    def _classify(self, history):
        if len(history) < 3:
            return "cv"
        disp = np.diff(np.array(history), axis=0)
        speeds = np.linalg.norm(disp, axis=1)
        mean_speed = speeds.mean()
        if mean_speed < 0.05:
            return "stationary"
        headings = np.arctan2(disp[:, 0], disp[:, 1])
        heading_var = np.var(np.unwrap(headings))
        if heading_var < 0.01:
            return "cv"
        if heading_var < 0.08:
            return "patrol"
        return "ou"

    def _horizon_for(self, behavior, speed):
        if behavior == "stationary":
            return 2
        if behavior == "cv":
            return min(self.max_horizon, max(4, int(3 + speed * 5)))
        if behavior == "patrol":
            return min(self.max_horizon, 6)
        return self.max_horizon

    def _radius_for(self, behavior, speed):
        if behavior == "stationary":
            return self.base_radius * 0.8
        if behavior == "cv":
            return self.base_radius * (1.0 + 0.3 * np.clip(speed, 0.0, 2.0))
        if behavior == "patrol":
            return self.base_radius * 1.3
        return self.base_radius * 1.6  # OU/불규칙

    def _sample_tracks(self, obj, history, behavior, horizon, los_mask):
        tracks = []
        pos = tuple(map(float, obj.p))
        vel = getattr(obj, "last_move", obj.v)
        los_ok = True
        if los_mask is not None:
            los_ok = line_of_sight(los_mask, pos[0], pos[1], pos[0] + vel[0], pos[1] + vel[1])
        if behavior in ("cv", "stationary"):
            tracks.append({"pos": pos, "vel": tuple(map(float, vel)), "los": los_ok})
        elif behavior == "patrol":
            # 순찰: 최근 방향과 그 반대 방향 모두 튜브 생성
            tracks.append({"pos": pos, "vel": tuple(map(float, vel)), "los": los_ok})
            tracks.append({"pos": pos, "vel": tuple(map(float, -np.array(vel))), "los": los_ok})
        else:
            # OU/불규칙: 여러 수평선으로 부채꼴 샘플
            base_heading = np.arctan2(vel[0], vel[1])
            for delta in np.linspace(-0.6, 0.6, 5):
                ang = base_heading + delta
                speed = max(0.2, np.linalg.norm(vel))
                vy = speed * np.sin(ang)
                vx = speed * np.cos(ang)
                tracks.append({"pos": pos, "vel": (vy, vx), "los": los_ok})
        return tracks


# ---- 간단 LOS(장애물 가림) 체크: Bresenham ----
def line_of_sight(grid01, y0, x0, y1, x1):
    """
    grid01: 0=free, 1=obstacle
    시야가 막히면 False
    """
    y0 = int(round(y0)); x0 = int(round(x0))
    y1 = int(round(y1)); x1 = int(round(x1))
    H, W = grid01.shape

    dy = abs(y1 - y0); dx = abs(x1 - x0)
    sy = 1 if y0 < y1 else -1
    sx = 1 if x0 < x1 else -1
    err = dx - dy

    y, x = y0, x0
    while True:
        if not (0 <= y < H and 0 <= x < W):
            return False
        if grid01[y, x] == 1:
            # 시작 지점이 벽이면 바로 False, 목표 셀 포함 여부는 용도에 따라 조정 가능
            return False
        if y == y1 and x == x1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy; x += sx
        if e2 < dx:
            err += dx; y += sy
    return True


def demo_danger_zone(steps=120, shape=(40, 40), seed=0):
    """
    간단한 데모: 몇 가지 MovingObj 패턴을 돌리며 DangerZone 맵을 보여줌.
    """
    try:
        from env.moving_object import MovingObj
    except ImportError:
        import pathlib, sys
        root = pathlib.Path(__file__).resolve().parents[1]
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        from env.moving_object import MovingObj

    rng = np.random.default_rng(seed)
    grid = np.zeros(shape, dtype=int)
    planner = BehavioralDangerPlanner(grid_shape=shape, decay=0.95, hard_thr=0.6)

    objs = [
        MovingObj(pos=(10.0, 5.0), vel=(0.0, 1.0), vmax=1.2, kind="cv", seed=int(rng.integers(1e9))),
        MovingObj(pos=(25.0, 32.0), vel=(0.4, -0.3), vmax=1.2, kind="ou", seed=int(rng.integers(1e9))),
        MovingObj(pos=(30.0, 10.0), vel=(0.0, 0.8), vmax=1.0, kind="patrol", seed=int(rng.integers(1e9))),
    ]

    for step in range(int(steps)):
        for obj in objs:
            obj.move(grid)
        planner.observe_objects(objs, grid)

    soft = planner.map.soft.copy()
    print(f"[DEMO] danger soft max={soft.max():.3f}, mean={soft.mean():.3f}, hard cells={planner.map.hard.sum()}")

    if not _HAS_MPL:
        print("[DEMO] matplotlib을 찾을 수 없어 heatmap을 표시하지 않습니다.")
        return soft

    plt.figure(figsize=(5, 5))
    plt.imshow(soft, cmap="inferno", origin="upper")
    plt.colorbar(label="danger level")
    plt.contour(planner.map.hard, levels=[0.5], colors="cyan", linewidths=1.0)
    plt.title("Danger Zone Heatmap (cyan=hard mask)")
    plt.tight_layout()
    plt.show()
    return soft


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="DangerZoneMap demo runner")
    parser.add_argument("--steps", type=int, default=120, help="number of simulation steps")
    parser.add_argument("--height", type=int, default=40, help="grid height")
    parser.add_argument("--width", type=int, default=40, help="grid width")
    parser.add_argument("--seed", type=int, default=0, help="random seed for demo objects")
    args = parser.parse_args()

    demo_danger_zone(steps=args.steps, shape=(args.height, args.width), seed=args.seed)
