#!/usr/bin/env python3
"""
Online CPP + RL Avoidance Runner

개요
- OnlineMapperAgent로 커버리지 경로를 온라인으로 생성/갱신(지그재그 + A*)
- DynAvoidOneObjEnv(하이브리드 FOLLOW_CPP/AVOID)를 사용해 로봇/동적객체 시뮬레이션
- FOLLOW_CPP 구간: CPP 경로를 따라 이동
- AVOID 구간: RL 정책으로 회피 행동 수행
- AVOID 이후 안전해지면 현재 위치/방문 셀 기준으로 CPP 리플랜, 경로 갱신

좌표계
- OnlineMapperAgent: (y, x)
- Env(DynAvoidOneObjEnv): waypoints는 (x, y). 내부에서 +1 패딩 오프셋 적용

주의
- RL 체크포인트의 obs_dim은 현재 환경의 관측 차원과 일치해야 함. 불일치 시 랜덤 가중치로 진행.
"""

import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt

from update_map.online_mapper_cpp import OnlineMapperAgent, load_map_grid
from env.dyn_env_one import DynAvoidOneObjEnv
from rl.network import ActorCritic


def path_to_waypoints_xy(path_yx):
    """[(y,x), ...] -> np.ndarray[[x,y], ...] (float32)"""
    if not path_yx:
        return np.zeros((0, 2), dtype=np.float32)
    wps = np.array([(x, y) for (y, x) in path_yx], dtype=np.float32)
    return wps


def build_env_with_agent(grid: np.ndarray, agent: OnlineMapperAgent, seed: int, cell_size_m: float = 0.20) -> DynAvoidOneObjEnv:
    wps = path_to_waypoints_xy(agent.current_path)
    env = DynAvoidOneObjEnv(grid=grid, waypoints=wps, seed=seed, cell_size_m=cell_size_m)
    return env


def load_policy(obs_dim: int, act_dim: int, hidden=(128, 128), feat_dim=128, ckpt: str | None = None, device: str = "cpu"):
    model = ActorCritic(obs_dim=obs_dim, act_dim=act_dim, hidden_sizes=hidden, feat_dim=feat_dim).to(device)
    if ckpt and os.path.exists(ckpt):
        try:
            sd = torch.load(ckpt, map_location=device)
            model.load_state_dict(sd, strict=False)
            print(f"[POLICY] Loaded checkpoint: {ckpt}")
        except Exception as e:
            print(f"[POLICY] Failed to load checkpoint ({e}). Using random weights.")
    else:
        if ckpt:
            print(f"[POLICY] Checkpoint not found: {ckpt}. Using random weights.")
    model.eval()
    return model


def main():
    ap = argparse.ArgumentParser(description="Online CPP + RL avoidance runner")
    ap.add_argument("--map", type=str, default="map_grid.npy")
    ap.add_argument("--ckpt", type=str, default="checkpoints_dyn/saved/best_ever.pt")
    ap.add_argument("--max-steps", type=int, default=2000)
    ap.add_argument("--render-interval", type=float, default=0.03)
    ap.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--unknown-blocked", action="store_true", help="Online planner treats unknown as blocked")
    ap.add_argument("--num-rays", type=int, default=16)
    ap.add_argument("--range", type=int, default=16)
    ap.add_argument("--replan-every", type=int, default=25, help="Force CPP replan every N steps when FOLLOW mode")
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # 1) 맵 및 온라인 플래너 준비
    base_grid = load_map_grid(args.map)
    agent = OnlineMapperAgent(
        base_grid=base_grid,
        sense_num_rays=args.num_rays,
        sense_fov_deg=360.0,
        sense_max_range_cells=args.range,
        unknown_is_blocked=args.unknown_blocked,
        robot_radius_cells=0,
        replan_timeout=200,
        draw_rays=False,
    )

    # 2) 초기 경로로 환경 구성
    env_seed = int(rng.integers(1, 1_000_000_000))
    env = build_env_with_agent(base_grid, agent, seed=env_seed)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    # 3) 정책 로드
    device = args.device
    policy = load_policy(obs_dim, act_dim, ckpt=args.ckpt, device=device)

    # 4) 시각화 준비
    plt.ion()
    fig, ax = plt.subplots(figsize=(7, 7))

    # 루프 상태
    steps = 0
    obs, _ = env.reset()
    prev_mode = "FOLLOW_CPP"
    last_replan_step = -99999

    while steps < args.max_steps:
        steps += 1

        # 정책/행동 결정
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            logits, _ = policy(obs_t)
            action = int(torch.argmax(logits, dim=-1).item())

        # 환경 스텝 (FOLLOW면 내부적으로 CPP 추종, AVOID면 action 사용)
        obs, reward, done, trunc, info = env.step(action)
        mode = info.get("mode", "-")

        # 온라인 플래너와 동기화/리플랜
        # - FOLLOW_CPP에서는 주기적으로 또는 경로 거의 소진 시 리플랜
        # - AVOID 종료 후 FOLLOW로 돌아오면 즉시 리플랜
        need_replan = False
        reason = None
        if mode == "FOLLOW_CPP":
            if prev_mode == "AVOID":
                need_replan, reason = True, "after-avoid"
            elif (steps - last_replan_step) >= args.replan_every:
                need_replan, reason = True, "periodic"
            elif getattr(env, "wp_idx", 0) >= max(0, len(env.waypoints) - 3):
                need_replan, reason = True, "path-near-end"
        else:
            # AVOID 중에는 CPP 경로 업데이트만 지연
            pass

        if need_replan:
            # 현재 위치를 에이전트에 반영하고 센싱/리플랜
            yx = (int(env.agent_rc[0]), int(env.agent_rc[1]))
            agent.pos = yx
            agent.sense_update()
            agent.replan(reason or "manual")
            # 환경 웨이포인트를 최신 경로로 교체
            env.waypoints = path_to_waypoints_xy(agent.current_path).astype(np.float32)
            # 가장 가까운 웨이포인트로 인덱스 맞추기
            if len(env.waypoints) > 0:
                ry, rx = env.agent_rc
                curr_xy = np.array([rx, ry], dtype=float)
                d = np.linalg.norm(env.waypoints - curr_xy[None, :], axis=1)
                env.wp_idx = int(np.argmin(d))
            last_replan_step = steps

        prev_mode = mode

        # 렌더
        ax.clear()
        ax.imshow(env.grid, cmap="Greys", origin="upper")
        if len(env.waypoints) > 0:
            ax.plot(env.waypoints[:, 0], env.waypoints[:, 1], "g--", alpha=0.6, label="CPP")
        ax.scatter(env.agent_rc[1], env.agent_rc[0], c=("orange" if mode == "AVOID" else "blue"), s=80, label="Robot")
        for i, obj in enumerate(getattr(env, "dynamic_objs", [])):
            ax.scatter(obj.p[1], obj.p[0], c="red", s=50, label=("Obj" if i == 0 else None))
        ax.set_xlim(0, env.W); ax.set_ylim(env.H, 0)
        ax.set_title(f"step {steps} | mode={mode} | R={reward:.2f}")
        ax.legend(loc="upper right")
        plt.pause(args.render_interval)

        if done or trunc:
            break

    plt.ioff(); plt.show()


if __name__ == "__main__":
    main()
