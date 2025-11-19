# ppo_train_dyn.py
import os
import argparse
import numpy as np
import torch
from typing import Optional, Dict, Tuple, List

from env.dyn_env_one import DynAvoidOneObjEnv
from env.env import make_map_30

from rl.ppo import PPOConfig, PPOTrainer

# =============================================================================
#  전이학습 설정: 불러올 모델 경로 (없으면 None)
# =============================================================================
PRETRAINED_MODEL_PATH: Optional[str] = "checkpoints_dyn/ppo_dyn_iter500.pt"
# =============================================================================


# ------------------------------ 커리큘럼 ------------------------------ #
def curriculum(iter_num: int, total_iters: int) -> Tuple[int, Dict[str, float], float]:
    """
    iter_num(1-base), total_iters에 따라 (객체수 K, 타입확률, easy_mix)을 반환.
      - type_probs: {"cv": p1, "patrol": p2, "ou": p3}
      - easy_mix: 과거 쉬운 분포(cv-only)로 강제 샘플링할 확률
    기본 설계(20k 가정)를 비율로 환산해 total_iters에 맞게 스케일함.
      0~3k (15%) : 1개, cv90%
      3k~6k      : 1개, cv60% patrol35%
      6k~9k      : 1개, ou50%
      9k~12k     : 1개, cv+patrol+ou 균등 + easy_mix 0.2
      12k~20k    : 2개 위주(80%는 K=2), 전 타입 균등 + easy_mix 0.15
    """
    base_total = 20000.0
    base_bounds = np.array([3000, 6000, 9000, 12000], dtype=float) / base_total
    scaled = [max(1, int(round(fr * total_iters))) for fr in base_bounds]
    # 경계가 같은 값으로 모이지 않게 단조 증가 보정
    t1 = scaled[0]
    t2 = max(scaled[1], t1 + 1)
    t3 = max(scaled[2], t2 + 1)
    t4 = max(scaled[3], t3 + 1)

    # 기본값
    obj_k = 1
    type_probs = {"cv": 1.0, "patrol": 0.0, "ou": 0.0}
    easy_mix = 0.0

    if iter_num <= t1:
        # 초반: 등속 위주
        type_probs = {"cv": 0.90, "patrol": 0.05, "ou": 0.05}
        obj_k = 1
        easy_mix = 0.0
    elif iter_num <= t2:
        # 순찰(사인/원형) 도입
        type_probs = {"cv": 0.60, "patrol": 0.35, "ou": 0.05}
        obj_k = 1
        easy_mix = 0.0
    elif iter_num <= t3:
        # 랜덤/OU 본격
        type_probs = {"cv": 0.30, "patrol": 0.20, "ou": 0.50}
        obj_k = 1
        easy_mix = 0.0
    elif iter_num <= t4:
        # 단일 혼합 + 망각 방지
        type_probs = {"cv": 0.34, "patrol": 0.33, "ou": 0.33}
        obj_k = 1
        easy_mix = 0.20
    else:
        # 12k ~ total_iters: 2개 객체 학습(경우의 수 많음)
        type_probs = {"cv": 0.34, "patrol": 0.33, "ou": 0.33}
        # 80% 확률로 2개, 20%는 1개 에피소드 섞어서 안정성
        obj_k = 2 if np.random.rand() < 0.8 else 1
        easy_mix = 0.15

    # easy_mix 발생 시: 과거 쉬운 분포로 강제 다운샘플
    if np.random.rand() < easy_mix:
        obj_k = 1
        type_probs = {"cv": 1.0, "patrol": 0.0, "ou": 0.0}

    return obj_k, type_probs, easy_mix


# --------------------- 스폰 함수(커리큘럼 기반 몽키패치) --------------------- #
def make_spawn_fn(obj_k: int, type_probs: Dict[str, float]):
    """
    DynAvoidOneObjEnv._default_spawn 대체 함수 생성.
    - obj_k: 이번 에피소드 목표 동적 객체 수(정수)
    - type_probs: {"cv","patrol","ou"}의 확률 분포(합=1)
    """
    # import는 내부에서 (환경 의존)
    from env.moving_object import MovingObj
    from utils_timing import estimate_robot_timeline

    # 확률 누적
    keys = ["cv", "patrol", "ou"]
    probs = np.array([type_probs.get(k, 0.0) for k in keys], dtype=float)
    probs = probs / max(probs.sum(), 1e-9)
    cdf = np.cumsum(probs)

    def sample_kind(rng: np.random.Generator) -> str:
        u = rng.random()
        for i, k in enumerate(keys):
            if u <= cdf[i]:
                return k
        return keys[-1]

    def _spawn(occ_grid, waypoints, rng, v_robot=1.0,
               v_obj_range=(0.6, 1.2), k_min=1, k_max=2, max_retry=50):
        """
        기존 env._default_spawn와 동일한 시그니처를 유지하되, obj_k / type_probs를 강제 반영.
        - patrol은 내부에서 'circle' 또는 'sin' 패턴 중 하나를 랜덤하게 구성.
        """
        H, W = occ_grid.shape
        objs: List[MovingObj] = []

        # 로봇 예상 타임라인 (cv 스폰에 쓰임)
        t_robot = estimate_robot_timeline(waypoints, v_robot_cells_per_step=v_robot)

        # 이번 에피소드 객체 수
        K = int(max(1, obj_k))

        def create_cv():
            # 로봇 타임라인 상 특정 지점에 맞춰 충돌궤적 생성(기존 로직 준용)
            for _ in range(max_retry):
                if len(waypoints) >= 6:
                    idx = int(rng.integers(3, max(4, len(waypoints) - 3)))
                else:
                    idx = int(rng.integers(0, len(waypoints)))
                wp_xy = np.array(waypoints[idx], float)  # (x,y)
                target_rc = np.array([wp_xy[1], wp_xy[0]], float)  # (y,x)
                t = float(t_robot[idx]) if len(t_robot) > idx else float(rng.uniform(8.0, 20.0))
                vmag = rng.uniform(*v_obj_range)
                theta = rng.uniform(0, 2 * np.pi)
                vy, vx = np.sin(theta) * vmag, np.cos(theta) * vmag
                start_rc = target_rc - np.array([vy, vx]) * t
                sy, sx = start_rc
                if 1 < sy < H - 2 and 1 < sx < W - 2 and occ_grid[int(sy), int(sx)] == 0:
                    return MovingObj(
                        pos=(float(sy), float(sx)),
                        vel=np.array([vy, vx], float),
                        vmax=max(v_obj_range[1], 1.2),
                        kind="cv",
                        seed=int(rng.integers(1e9)),
                    )
            return None

        def create_ou():
            for _ in range(max_retry):
                sy = int(rng.integers(1, H - 1))
                sx = int(rng.integers(1, W - 1))
                if occ_grid[sy, sx] != 0:
                    continue
                vmag = rng.uniform(0.3, 0.6)  # OU는 더 느리게
                theta = rng.uniform(0, 2 * np.pi)
                vy, vx = np.sin(theta) * vmag, np.cos(theta) * vmag
                obj = MovingObj(
                    pos=(float(sy), float(sx)),
                    vel=np.array([vy, vx], float),
                    vmax=0.8,
                    kind="ou",
                    seed=int(rng.integers(1e9)),
                )
                # OU 파라미터 완만
                obj.sigma = 0.5
                obj.theta = 1.2
                return obj
            return None

        def create_patrol():
            # circle 또는 sin 패턴으로 순찰 포인트 생성
            for _ in range(max_retry):
                sy = int(rng.integers(1, H - 1))
                sx = int(rng.integers(1, W - 1))
                if occ_grid[sy, sx] != 0:
                    continue

                vmag = rng.uniform(*v_obj_range)
                theta = rng.uniform(0, 2 * np.pi)
                vy, vx = np.sin(theta) * vmag, np.cos(theta) * vmag

                obj = MovingObj(
                    pos=(float(sy), float(sx)),
                    vel=np.array([vy, vx], float),
                    vmax=max(v_obj_range[1], 1.2),
                    kind="patrol",
                    seed=int(rng.integers(1e9)),
                )

                pts = []
                if rng.random() < 0.5:
                    # circle
                    r_cells = int(rng.integers(3, 6))
                    for ang in np.linspace(0, 2 * np.pi, 8, endpoint=False):
                        py = int(round(sy + r_cells * np.sin(ang)))
                        px = int(round(sx + r_cells * np.cos(ang)))
                        if 0 <= py < H and 0 <= px < W and occ_grid[py, px] == 0:
                            pts.append((py, px))
                else:
                    # sin/zigzag
                    amp = int(rng.integers(2, 4))
                    step = int(rng.integers(2, 4))
                    for k in range(6):
                        px = int(sx + k * step)
                        py = int(sy + (amp if (k % 2 == 0) else -amp))
                        if 0 <= py < H and 0 <= px < W and occ_grid[py, px] == 0:
                            pts.append((py, px))

                # 부족하면 사각 루프 백업
                if len(pts) < 3:
                    cand = [
                        (sy, sx),
                        (sy, min(W - 2, sx + 3)),
                        (min(H - 2, sy + 3), min(W - 2, sx + 3)),
                        (min(H - 2, sy + 3), sx),
                    ]
                    pts = [(py, px) for (py, px) in cand if 0 <= py < H and 0 <= px < W and occ_grid[py, px] == 0]

                if len(pts) >= 2:
                    obj.set_patrol([(float(py), float(px)) for (py, px) in pts])
                    return obj
            return None

        creators = {"cv": create_cv, "ou": create_ou, "patrol": create_patrol}

        for _ in range(K):
            kind = sample_kind(rng)
            obj = creators[kind]()
            if obj is None:
                # 최후 백업: cv 시도
                obj = create_cv()
            if obj is None:
                # 그래도 실패하면 OU 시도
                obj = create_ou()
            if obj is None:
                # 최종 백업: 무조건 한 칸 비어있는 곳에 느린 cv
                while True:
                    sy = int(rng.integers(1, H - 1))
                    sx = int(rng.integers(1, W - 1))
                    if occ_grid[sy, sx] == 0:
                        break
                vmag = rng.uniform(0.4, 0.8)
                theta = rng.uniform(0, 2 * np.pi)
                vy, vx = np.sin(theta) * vmag, np.cos(theta) * vmag
                obj = MovingObj(
                    pos=(float(sy), float(sx)),
                    vel=np.array([vy, vx], float),
                    vmax=1.0,
                    kind="cv",
                    seed=int(rng.integers(1e9)),
                )
            objs.append(obj)

        return objs

    return _spawn


# ---------------------------------------- #
"""
예시:
python3 ppo_train_dyn_fast.py \
  --total-iters 500 \
  --regen-every 10 \
  --map-size 30 \
  --overwrite-base \
  --save-maps
"""
def main():
    parser = argparse.ArgumentParser(description="Train PPO with curriculum + periodic random map regeneration")
    parser.add_argument("--total-iters", type=int, default=500, help="Total training iterations (fast mode default)")
    parser.add_argument("--regen-every", type=int, default=10, help="Regenerate map every N iters")
    parser.add_argument("--map-size", type=int, default=30, help="Map size (size x size) when generating maps")
    parser.add_argument("--save-maps", action="store_true", help="Save generated maps/waypoints at regeneration")
    parser.add_argument("--maps-dir", type=str, default="generated_maps", help="Directory to save generated maps when --save-maps")
    parser.add_argument("--overwrite-base", action="store_true", help="Overwrite map_grid.npy & waypoints.npy on each regeneration (use with caution)")
    args = parser.parse_args()

    grid_path = "map_grid.npy"
    wps_path  = "waypoints.npy"

    rng = np.random.default_rng()

    # 초기 환경 준비(파일 있으면 로드, 없으면 생성)
    if os.path.exists(grid_path) and os.path.exists(wps_path):
        grid = np.load(grid_path)
        wps  = np.load(wps_path)
        seed = int(rng.integers(1, 1_000_000_000))
        env = DynAvoidOneObjEnv(grid=grid, waypoints=wps, seed=seed, cell_size_m=0.20)
        print(f"[ENV] Random training seed: {seed}")
        print("[ENV] Loaded grid/waypoints from files.")
    else:
        map_seed = int(rng.integers(1, 1_000_000_000))
        grid, wps, _ = make_map_30(seed=map_seed, size=args.map_size)
        env_seed = int(rng.integers(1, 1_000_000_000))
        env = DynAvoidOneObjEnv(grid=grid, waypoints=wps, seed=env_seed, cell_size_m=0.20)
        print(f"[ENV] Initialized by generating first map (seed={map_seed}, env_seed={env_seed})")
        if args.save_maps:
            os.makedirs(args.maps_dir, exist_ok=True)
            np.save(os.path.join(args.maps_dir, f"map_iter0000_seed{map_seed}.npy"), grid)
            np.save(os.path.join(args.maps_dir, f"wps_iter0000_seed{map_seed}.npy"), wps)

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    print(f"[ENV] obs_dim={obs_dim}, act_dim={act_dim}")

    cfg = PPOConfig(
        obs_dim=obs_dim,
        act_dim=act_dim,
        rollout_steps=2048,
        lr=3e-4,
        epochs=10,
        batch_size=256,
        clip_eps=0.2,
        vf_coef=0.5,
        ent_coef=0.01,
        max_grad_norm=0.5,
        gamma=0.99,
        gae_lambda=0.95,
        device="cpu",
        seed=0,
        hidden_sizes=(256,256,256),
        feat_dim=256
    )

    trainer = PPOTrainer(env, cfg)

    # 전이학습 로드
    if PRETRAINED_MODEL_PATH and os.path.exists(PRETRAINED_MODEL_PATH):
        try:
            map_location = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            trainer.model.load_state_dict(torch.load(PRETRAINED_MODEL_PATH, map_location=map_location))
            print(f"\n✅ 모델을 성공적으로 불러왔습니다: {PRETRAINED_MODEL_PATH}")
        except Exception as e:
            print(f"\n⚠️ 모델 불러오기 실패: {e}. 새 학습을 시작합니다.")
    else:
        if PRETRAINED_MODEL_PATH:
            print(f"\n⚠️ 지정된 모델 파일을 찾을 수 없습니다: {PRETRAINED_MODEL_PATH}. 새 학습을 시작합니다.")
        else:
            print("\n✨ 새로운 학습을 시작합니다.")

    total_iters = int(args.total_iters)

    eval_every  = 10
    save_every  = 50
    save_dir    = "checkpoints_dyn"
    os.makedirs(save_dir, exist_ok=True)

    regen_every = int(args.regen_every)

    # 초기 스폰 정책(커리큘럼 1 iter 적용)
    init_k, init_probs, _ = curriculum(1, total_iters)
    env._default_spawn = make_spawn_fn(init_k, init_probs)

    for it in range(1, total_iters + 1):
        # === 커리큘럼 적용: iter마다 스폰 정책 갱신 ===
        obj_k, type_probs, _easy = curriculum(it, total_iters)
        env._default_spawn = make_spawn_fn(obj_k, type_probs)

        # 주기적으로 맵/웨이포인트 재생성
        if it % regen_every == 0:
            try:
                map_seed = int(rng.integers(1, 1_000_000_000))
                grid_new, wps_new, _ = make_map_30(seed=map_seed, size=args.map_size)
                env_seed = int(rng.integers(1, 1_000_000_000))
                new_env = DynAvoidOneObjEnv(grid=grid_new, waypoints=wps_new, seed=env_seed, cell_size_m=0.20)

                # 관측/행동 공간 호환성 체크
                assert new_env.observation_space.shape[0] == cfg.obs_dim, \
                    f"obs_dim changed: {new_env.observation_space.shape[0]} != {cfg.obs_dim}"
                assert new_env.action_space.n == cfg.act_dim, \
                    f"act_dim changed: {new_env.action_space.n} != {cfg.act_dim}"

                # 새 환경에도 커리큘럼 스폰 정책 주입
                new_env._default_spawn = make_spawn_fn(obj_k, type_probs)

                trainer.env = new_env
                if hasattr(trainer, "_curr_obs"):
                    delattr(trainer, "_curr_obs")
                if hasattr(trainer, "_last_mode"):
                    trainer._last_mode = "FOLLOW_CPP"

                print(f"[ENV] Iter {it}: regenerated map with seed={map_seed}, env_seed={env_seed}, size={args.map_size}")

                # 기본 파일 덮어쓰기 옵션 처리
                if args.overwrite_base:
                    np.save(grid_path, grid_new)
                    np.save(wps_path, wps_new)
                    print(f"[ENV] Base files overwritten: {grid_path}, {wps_path}")
                else:
                    if it == regen_every:
                        print("[INFO] --overwrite-base 미사용: 디스크의 map_grid.npy / waypoints.npy 는 초기 로드 상태 유지.")

                if args.save_maps:
                    os.makedirs(args.maps_dir, exist_ok=True)
                    np.save(os.path.join(args.maps_dir, f"map_iter{it:04d}_seed{map_seed}.npy"), grid_new)
                    np.save(os.path.join(args.maps_dir, f"wps_iter{it:04d}_seed{map_seed}.npy"), wps_new)

            except Exception as e:
                print(f"[ENV] Iter {it}: map regeneration failed: {e}")

        steps = trainer.collect_rollout()
        logs  = trainer.update()

        if it % eval_every == 0:
            print(f"[Iter {it:05d}] steps={steps} pi={logs.get('policy_loss',0):.4f} "
                  f"vf={logs.get('value_loss',0):.4f} ent={logs.get('entropy',0):.3f}")

        if it % save_every == 0:
            path = os.path.join(save_dir, f"ppo_dyn_iter{it}.pt")
            torch.save(trainer.model.state_dict(), path)
            print(f"[SAVE] {path}")

    print("[DONE] PPO Training finished.")


if __name__ == "__main__":
    main()
