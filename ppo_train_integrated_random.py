import os
import argparse
import numpy as np
import torch
from typing import Dict, Tuple, List
from torch.distributions import Categorical

from env.dyn_env_one import DynAvoidOneObjEnv
from env.env import make_map_30
from rl.ppo import PPOConfig, PPOTrainer

"""
단순 랜덤 커리큘럼 통합 학습:
- 객체 수: 1개 75%, 2개 25% (3:1)
- 객체 타입: cv/patrol/ou/stopgo 균등(4종 25%씩)

예시:
python3 ppo_train_integrated_random.py \
  --random-map --map-size 30 --regen-map-interval 10 \
  --escape-updates 500 --main-every 1 --main-updates-per-escape 1 \
  --rollout-steps 4096 --batch-size 512 --lr 2e-4 --device cpu
"""

# ------------------------------ 커리큘럼 ------------------------------ #
def curriculum(iter_num: int, total_iters: int) -> Tuple[int, Dict[str, float], float]:
    obj_k = 1 if np.random.rand() < 0.75 else 2
    type_probs = {"cv": 0.25, "patrol": 0.25, "ou": 0.25, "stopgo": 0.25}
    easy_mix = 0.0
    return obj_k, type_probs, easy_mix


def make_spawn_fn(obj_k: int, type_probs: Dict[str, float]):
    from env.moving_object import MovingObj
    from utils_timing import estimate_robot_timeline

    keys = ["cv", "patrol", "ou", "stopgo"]
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
        H, W = occ_grid.shape
        objs: List[MovingObj] = []

        _ = estimate_robot_timeline(waypoints, v_robot_cells_per_step=v_robot)
        K = int(max(1, obj_k))
        for _ in range(K):
            for _ in range(max_retry):
                y = int(rng.integers(H))
                x = int(rng.integers(W))
                if occ_grid[y, x] != 0:
                    continue
                kind = sample_kind(rng)
                if kind == "cv":
                    theta = float(rng.random() * 2 * np.pi)
                    vy = float(np.sin(theta) * rng.uniform(*v_obj_range))
                    vx = float(np.cos(theta) * rng.uniform(*v_obj_range))
                    obj = MovingObj(
                        pos=np.array([y, x], float),
                        vel=np.array([vy, vx], float),
                        vmax=1.0,
                        kind="cv",
                        seed=int(rng.integers(1e9)),
                    )
                elif kind == "patrol":
                    kind_internal = "circle" if rng.random() < 0.5 else "sin"
                    obj = MovingObj(
                        pos=np.array([y, x], float),
                        vel=np.array([0.0, 0.0], float),
                        vmax=1.0,
                        kind="patrol",
                        seed=int(rng.integers(1e9)),
                        patrol_type=kind_internal,
                        patrol_center=np.array([y, x], float),
                        patrol_radius=float(rng.uniform(3.0, 6.0)),
                        patrol_phase=float(rng.uniform(0, 2 * np.pi)),
                    )
                elif kind == "stopgo":
                    obj = MovingObj(
                        pos=np.array([y, x], float),
                        vel=np.array([0.0, 0.0], float),
                        vmax=1.0,
                        kind="stopgo",
                        seed=int(rng.integers(1e9)),
                    )
                else:  # "ou"
                    obj = MovingObj(
                        pos=np.array([y, x], float),
                        vel=np.array([0.0, 0.0], float),
                        vmax=1.0,
                        kind="ou",
                        seed=int(rng.integers(1e9)),
                        ou_theta=0.15 + 0.05 * rng.random(),
                        ou_sigma=0.4 + 0.1 * rng.random(),
                    )
                objs.append(obj)
                break
        return objs

    return _spawn


def parse_args():
    p = argparse.ArgumentParser(description="Main + Escape 통합 학습 (랜덤 커리큘럼)")
    p.add_argument("--random-map", action="store_true")
    p.add_argument("--map-size", type=int, default=30)
    p.add_argument("--regen-map-interval", type=int, default=10, help="이 주기(escape iter 기준)마다 새 맵 생성 (0이면 고정)")
    p.add_argument("--grid-path", default="map_grid.npy")
    p.add_argument("--waypoints-path", default="waypoints.npy")
    p.add_argument("--escape-updates", type=int, default=300)
    p.add_argument("--main-every", type=int, default=1, help="escape iter마다 main을 학습할 간격(0이면 main 학습 안함)")
    p.add_argument("--main-updates-per-escape", type=int, default=1, help="main 업데이트를 한 번에 몇 회 돌릴지")
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=1234)
    # PPO 공통
    p.add_argument("--rollout-steps", type=int, default=2048)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--main-hidden-sizes", type=int, nargs="+", default=[256, 256, 128])
    p.add_argument("--escape-hidden-sizes", type=int, nargs="+", default=[256, 256, 128])
    p.add_argument("--main-feat-dim", type=int, default=256)
    p.add_argument("--escape-feat-dim", type=int, default=256)
    # 경로/저장
    p.add_argument("--out-dir", default="checkpoints_integrated_random")
    p.add_argument("--save-interval", type=int, default=50)
    p.add_argument("--pretrained-main", default=None)
    p.add_argument("--pretrained-escape", default=None)
    # 콜랩 드라이브 연동
    p.add_argument("--mount-drive", action="store_true", help="Colab에서 Google Drive 마운트 시도")
    p.add_argument("--drive-out-dir", default=None, help="지정 시 out-dir 대신 이 경로에 저장 (예: /content/drive/MyDrive/grid_ckpt)")
    return p.parse_args()


def load_array(path):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return np.load(path)


def detect_device(arg_device: str):
    if arg_device != "auto":
        return arg_device
    return "cuda" if torch.cuda.is_available() else "cpu"


def build_map(args, seed):
    if args.random_map:
        grid, wps, _ = make_map_30(seed=seed, size=args.map_size)
    else:
        grid = load_array(args.grid_path)
        wps = load_array(args.waypoints_path)
    return grid, wps


def build_env(grid, wps, seed, use_escape=False, cell_size=0.20):
    return DynAvoidOneObjEnv(
        grid=grid,
        waypoints=wps,
        seed=seed,
        cell_size_m=cell_size,
        use_escape_subpolicy=use_escape,
    )


def switch_env(trainer: PPOTrainer, env, device):
    trainer.env = env
    obs, _ = env.reset()
    trainer._curr_obs = torch.as_tensor(obs, dtype=torch.float32, device=device)


def collect_escape_segments(trainer: PPOTrainer, cfg, pre_steps=12, escape_release_steps=3):
    env = trainer.env
    model = trainer.model
    buffer = trainer.buffer
    device = trainer.device
    buffer.clear()

    max_cap = buffer.cfg.max_size
    if not hasattr(trainer, "_curr_obs"):
        obs, _ = env.reset()
        trainer._curr_obs = torch.as_tensor(obs, dtype=torch.float32, device=device)
    curr_obs = trainer._curr_obs
    prev_info = {}

    from collections import deque
    pre_buf = deque(maxlen=pre_steps)
    recording = False
    escape_out_counter = 0

    collected = 0
    while collected < cfg.rollout_steps and buffer.ptr < max_cap:
        obs_t = curr_obs.unsqueeze(0)
        with torch.no_grad():
            logits, value = model(obs_t)
            value = value.squeeze(0).squeeze(-1)
            dist = Categorical(logits=logits.squeeze(0))
            action = dist.sample()
            logprob = dist.log_prob(action)

        next_obs, reward, done, trunc, info = env.step(int(action.item()))
        done_flag = bool(done or trunc)

        is_stuck = bool(info.get("stuck_state", False))
        was_stuck = bool(prev_info.get("stuck_state", False))
        mode = info.get("mode", "")

        pre_buf.append((curr_obs, action, logprob, float(reward), done_flag, value))

        if not recording and is_stuck and not was_stuck:
            for (o, a, lp, r, d, v) in pre_buf:
                if buffer.ptr >= max_cap or collected >= cfg.rollout_steps:
                    break
                buffer.store(obs=o, action=a, logprob=lp, reward=r, done=d, value=v, mask=True)
                collected += 1
            recording = True

        if recording:
            if buffer.ptr >= max_cap or collected >= cfg.rollout_steps:
                break
            buffer.store(obs=curr_obs, action=action, logprob=logprob, reward=float(reward), done=done_flag, value=value, mask=True)
            collected += 1

        if recording:
            if not is_stuck and mode != "ESCAPE":
                escape_out_counter += 1
                if escape_out_counter >= escape_release_steps:
                    recording = False
                    escape_out_counter = 0
                    with torch.no_grad():
                        _, v_boot = model(torch.as_tensor(next_obs, dtype=torch.float32, device=device).unsqueeze(0))
                        v_boot = v_boot.squeeze(0).squeeze(-1)
                    buffer.finish_path(last_value=v_boot)
                    pre_buf.clear()
            else:
                escape_out_counter = 0

        if done_flag:
            if recording:
                buffer.finish_path(last_value=torch.zeros((), device=device))
                recording = False
                escape_out_counter = 0
                pre_buf.clear()
            next_obs, _ = env.reset()
            curr_obs = torch.as_tensor(next_obs, dtype=torch.float32, device=device)
            prev_info = {}
            continue

        curr_obs = torch.as_tensor(next_obs, dtype=torch.float32, device=device)
        prev_info = info

    if recording:
        with torch.no_grad():
            _, v_boot = model(curr_obs.unsqueeze(0))
            v_boot = v_boot.squeeze(0).squeeze(-1)
        buffer.finish_path(last_value=v_boot)

    trainer._curr_obs = curr_obs
    return collected


def main():
    args = parse_args()
    device = detect_device(args.device)
    map_seed = args.seed

    out_dir = args.out_dir
    if args.drive_out_dir:
        out_dir = args.drive_out_dir
    if args.mount_drive or (out_dir and "/content/drive" in out_dir):
        try:
            from google.colab import drive  # type: ignore
            drive.mount("/content/drive")
            print("[INFO] Mounted Google Drive at /content/drive")
        except Exception as e:
            print(f"[WARN] Drive mount failed or not in Colab: {e}")
    os.makedirs(out_dir, exist_ok=True)

    rng = np.random.default_rng()

    grid, wps = build_map(args, map_seed)
    env_main = build_env(grid, wps, seed=args.seed, use_escape=False)
    env_escape = build_env(grid, wps, seed=args.seed + 1, use_escape=True)

    cfg_main = PPOConfig(
        obs_dim=env_main.observation_space.shape[0],
        act_dim=env_main.action_space.n,
        hidden_sizes=tuple(args.main_hidden_sizes),
        feat_dim=args.main_feat_dim,
        rollout_steps=args.rollout_steps,
        lr=args.lr,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=device,
        seed=args.seed,
    )
    cfg_escape = PPOConfig(
        obs_dim=env_escape.observation_space.shape[0],
        act_dim=env_escape.action_space.n,
        hidden_sizes=tuple(args.escape_hidden_sizes),
        feat_dim=args.escape_feat_dim,
        rollout_steps=args.rollout_steps,
        lr=args.lr,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=device,
        seed=args.seed + 999,
    )

    trainer_main = PPOTrainer(env_main, cfg_main)
    trainer_escape = PPOTrainer(env_escape, cfg_escape)

    if args.pretrained_main and os.path.exists(args.pretrained_main):
        trainer_main.model.load_state_dict(torch.load(args.pretrained_main, map_location=device), strict=False)
        print(f"[INFO] loaded main pretrained: {args.pretrained_main}")
    if args.pretrained_escape and os.path.exists(args.pretrained_escape):
        trainer_escape.model.load_state_dict(torch.load(args.pretrained_escape, map_location=device), strict=False)
        print(f"[INFO] loaded escape pretrained: {args.pretrained_escape}")

    print(f"[INFO] device={device}, escape_updates={args.escape_updates}, main_every={args.main_every}")

    def apply_curriculum(it_num: int):
        obj_k, type_probs, easy_mix = curriculum(it_num, args.escape_updates)
        spawn_fn = make_spawn_fn(obj_k, type_probs)
        env_main._default_spawn = spawn_fn
        env_escape._default_spawn = spawn_fn
        return obj_k, type_probs, easy_mix

    apply_curriculum(1)

    for it in range(1, args.escape_updates + 1):
        if args.regen_map_interval > 0 and it % args.regen_map_interval == 0:
            map_seed += 1
            grid, wps = build_map(args, map_seed)
            switch_env(trainer_main, build_env(grid, wps, seed=map_seed, use_escape=False), device)
            switch_env(trainer_escape, build_env(grid, wps, seed=map_seed + 1, use_escape=True), device)
            env_main = trainer_main.env
            env_escape = trainer_escape.env
            print(f"[INFO] regenerated map at escape iter {it} (seed={map_seed})")

        apply_curriculum(it)

        steps_escape = collect_escape_segments(trainer_escape, cfg_escape, pre_steps=12, escape_release_steps=3)
        logs_escape = trainer_escape.update()

        logs_main = {}
        if args.main_every > 0 and (it % args.main_every == 0):
            for _ in range(args.main_updates_per_escape):
                steps_main = trainer_main.collect_rollout()
                logs_main = trainer_main.update()
                print("[MAIN]", {"iter": it, "steps": steps_main, **{k: f"{v:.4f}" for k, v in (logs_main or {}).items()}})

        print("[ESC ]", {"iter": it, "steps": steps_escape, **{k: f"{v:.4f}" for k, v in (logs_escape or {}).items()}})

        if it % args.save_interval == 0:
            torch.save(trainer_escape.model.state_dict(), os.path.join(out_dir, f"escape_iter{it}.pt"))
            torch.save(trainer_main.model.state_dict(), os.path.join(out_dir, f"main_iter{it}.pt"))

    torch.save(trainer_escape.model.state_dict(), os.path.join(out_dir, f"escape_iter{args.escape_updates}.pt"))
    torch.save(trainer_main.model.state_dict(), os.path.join(out_dir, f"main_iter{args.escape_updates}.pt"))
    print(f"[DONE] saved final models to {out_dir}")


if __name__ == "__main__":
    main()
