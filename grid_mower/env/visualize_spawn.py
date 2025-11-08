import numpy as np
import matplotlib.pyplot as plt

from env.dyn_env_one import DynAvoidOneObjEnv
from env.moving_object import MovingObj
from utils_timing import estimate_robot_timeline

def visualize_spawn():
    # ---------- 환경 생성 ----------
    grid = np.load("map_grid.npy")
    wps  = np.load("waypoints.npy")

    rng = np.random.default_rng(57)
    env = DynAvoidOneObjEnv(grid=grid, waypoints=wps, seed=42)

    # ---------- 로봇 타임라인 계산 ----------
    t_robot = estimate_robot_timeline(wps, v_robot=1.0)
    print(f"[INFO] total time ~ {t_robot[-1]:.2f}s")

    # ---------- 동적 객체 생성 ----------
    objs = env.spawn_collision_timed_objects(
        env.grid,                # 맵 정보
        env.waypoints,           # 로봇 경로 전달
        env.rng,                 # 난수 생성기 전달
        v_robot=1.0,
        v_obj_range=(0.8, 1.2)
    )
    
    # ---------- 시각화 ----------
    H, W = grid.shape
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.imshow(grid, cmap="Greys", origin="upper")

    # ① CPP 경로
    ax.plot(wps[:, 0], wps[:, 1], "g--", lw=2, label="CPP path")
    ax.scatter(wps[:, 0], wps[:, 1], c="lime", s=20)

    # ② 동적 객체 초기 위치
    for i, obj in enumerate(objs):
        ax.scatter(obj.p[1], obj.p[0], s=60, c="red", label="Object" if i == 0 else None)

        # 예측 궤적 (시간 흐름에 따른 위치)
        traj = [obj.p + obj.v * (t) for t in np.linspace(0, t_robot[-1], 30)]
        traj = np.array(traj)
        ax.plot(traj[:, 1], traj[:, 0], "r-", alpha=0.5)

    # ③ 로봇 경로 (시간 스케일 표시용)
    for i, (x, y) in enumerate(wps[::3]):
        ax.text(x + 0.3, y, f"t={t_robot[::3][i]:.1f}", fontsize=8, color="blue")

    ax.set_title("Trajectory-aligned Dynamic Object Spawning")
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.legend()
    plt.show()


if __name__ == "__main__":
    visualize_spawn()