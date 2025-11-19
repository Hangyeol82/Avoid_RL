# test_route_blocked_vis.py
import numpy as np
import torch
import matplotlib.pyplot as plt
import os, random

from env.dyn_env_one import DynAvoidOneObjEnv
from env.moving_object import MovingObj

# --------------------------------------------------
# 간단한 테스트 환경 생성: 목표 앞 통로를 순찰 객체가 막도록 구성
# --------------------------------------------------
def create_env(seed=0):
    rng = np.random.default_rng(seed)

    # 1) 기본 맵(원본 크기) 만들기: 20x28, 테두리는 env에서 +2로 패딩됨
    H, W = 20, 28
    base = np.zeros((H, W), dtype=int)

    # 테두리는 env에서 패딩으로 벽이 생기므로 여기선 내부만 구성
    # 통로: (x=14) 기둥 두 개로 위/아래 막고 가운데만 뚫림
    base[2:8, 14]  = 1
    base[12:18,14] = 1

    # 시작지점
    base[10, 3] = 2

    # 2) 웨이포인트(원본 좌표; env에서 자동 +1 시프트됨)
    #    목표(세 번째)가 바로 통로 너머에 위치 → 객체가 막으면 route_blocked 발생 쉬움
    wps = np.array([
        [3, 10],   # start 근처 (x,y)
        [10, 10],  # 통로 전
        [17, 10],  # 통로 통과 직후 (여기가 "목표 원" 중심)
        [24, 10]   # 오른쪽 끝 목표
    ], dtype=float)

    env = DynAvoidOneObjEnv(grid=base, waypoints=wps, seed=seed, cell_size_m=0.20)

    # 3) 목표 지점 앞을 상하로 순찰하는 객체 배치 (env는 이미 전체 좌표 +1된 상태!)
    #    통로 중앙 좌표를 (x=15, y=10) 근방으로 맞추어 순찰
    #    patrol 구간을 y=8~12로 왕복
    patrol_center_x = 15.0
    patrol_y1 = 8.0
    patrol_y2 = 12.0
    blocker = MovingObj(
        pos=(10.0, patrol_center_x),          # (y, x)  ← env 내부 좌표(이미 패딩 적용됨)
        vel=np.array([0.5, 0.0], dtype=float),# y방향 왕복
        vmax=0.6,
        kind="patrol",
        seed=int(rng.integers(1e9))
    )
    blocker.patrol_p1 = np.array([patrol_y1, patrol_center_x])
    blocker.patrol_p2 = np.array([patrol_y2, patrol_center_x])

    env.dynamic_objs = [blocker]

    # 4) 막힘 감지 민감도 조정
    env.block_dyn_radius_cells = 3.0
    env.block_route_threshold = 0.55
    env.block_route_avg_threshold = 0.45
    env.stuck_block_min = 0.45

    print("[ENV] test env ready.")
    return env

# --------------------------------------------------
# 시각화·검증 루프
# --------------------------------------------------
def run_episode(env, max_steps=400, render_interval=0.05, seed=0):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

    obs, _ = env.reset()

    # reset 이후 다시 동일 blocker를 재주입(내부 스폰이 바뀔 수 있으므로)
    # → create_env와 동일 설정 재적용
    env.dynamic_objs = []
    # 목표 앞 순찰 객체 재설치
    patrol_center_x = 15.0
    patrol_y1 = 8.0
    patrol_y2 = 12.0
    blocker = MovingObj(
        pos=(10.0, patrol_center_x),
        vel=np.array([0.5, 0.0], dtype=float),
        vmax=0.6,
        kind="patrol",
        seed=int(env.rng.integers(1e9))
    )
    blocker.patrol_p1 = np.array([patrol_y1, patrol_center_x])
    blocker.patrol_p2 = np.array([patrol_y2, patrol_center_x])
    env.dynamic_objs = [blocker]

    # 관측 캐시 업데이트
    obs = env._obs()

    H, W = env.grid.shape
    plt.ion()
    fig, ax = plt.subplots(figsize=(7.5, 6.5))

    def draw_circle(ax, cx, cy, r_cells, **kw):
        th = np.linspace(0, 2*np.pi, 200)
        xs = cx + r_cells * np.cos(th)
        ys = cy + r_cells * np.sin(th)
        ax.plot(xs, ys, **kw)

    step = 0
    done = False
    trunc = False
    while not (done or trunc) and step < max_steps:
        step += 1

        # 간단 정책:
        #  - FOLLOW_CPP: env가 내부에서 한 칸 전진하므로 action=None
        #  - AVOID: STAY(4)로 평균 이동거리 ↓ (route_blocked True가 빨리 뜸)
        action = None
        mode = "FOLLOW_CPP"
        # 관측 전 info는 없으니, 한 스텝 진행 후 info로 판단
        # 첫 스텝은 FOLLOW_CPP 가정, 이후 info 기반으로 처리
        obs, reward, done, trunc, info = env.step(action)

        mode = info.get("mode", "FOLLOW_CPP")
        if mode == "AVOID":
            # 다시 한 번, AVOID일 때 STAY로 진짜 route_blocked 조건을 쉽게 만족시키자
            obs, reward, done, trunc, info = env.step(4)  # 4=STAY

        # --- 그리기 ---
        ax.clear()
        ax.imshow(env.grid, cmap="Greys", origin="upper")

        # CPP
        if len(env.waypoints) > 0:
            ax.plot(env.waypoints[:, 0], env.waypoints[:, 1], "g--", alpha=0.6, label="CPP")

        # 목표별 표시 + 목표 원(반경)
        if env.wp_idx < len(env.waypoints):
            gx, gy = env.waypoints[env.wp_idx]
            ax.scatter(gx, gy, c="gold", s=120, marker="*", label="Goal")
            goal_R_cells = getattr(env, "block_dyn_radius_cells", 3.0)
            draw_circle(ax, gx, gy, goal_R_cells, color="gold", alpha=0.6)

        # 객체 표시 + 객체 원(반경)
        if hasattr(env, "dynamic_objs"):
            obj_R_cells = getattr(env, "block_dyn_radius_cells", 3.0)
            for i, obj in enumerate(env.dynamic_objs):
                ax.scatter(obj.p[1], obj.p[0], c="crimson", s=60, label="Obj" if i == 0 else None)
                draw_circle(ax, obj.p[1], obj.p[0], obj_R_cells, color="crimson", alpha=0.5)

        # 로봇
        ax.scatter(env.agent_rc[1], env.agent_rc[0], c="royalblue", s=70, label="Robot")

        # 진단값 표시
        severity = float(info.get("block_severity", 0.0))
        block_avg = float(info.get("block_avg", 0.0))
        dyn_sev = float(info.get("block_dyn_severity", 0.0))
        static_density = float(info.get("block_static_density", 0.0))
        move_mean = float(info.get("move_mean_m", 0.0))
        stuck_state = bool(info.get("stuck_state", False))

        t1 = (f"step={step}  mode={info.get('mode','-')}  "
              f"stuck={int(stuck_state)}  sev={severity:.2f} avg={block_avg:.2f}")
        t2 = (f"dyn={dyn_sev:.2f} static={static_density:.2f} move={move_mean:.3f}m")
        ax.set_title(t1 + "\n" + t2, loc="left", fontsize=10)

        ax.set_xlim(0, W); ax.set_ylim(H, 0)
        ax.legend(loc="upper right")
        plt.pause(render_interval)

    plt.ioff(); plt.show()


def main():
    seed = 7
    env = create_env(seed=seed)

    run_episode(env, max_steps=600, render_interval=0.05, seed=seed)


if __name__ == "__main__":
    main()
