import os
import argparse
import numpy as np
import torch
from torch.distributions import Categorical

from env.dyn_env_one import DynAvoidOneObjEnv
from env.env import make_map_30
from rl.ppo import PPOConfig, PPOTrainer

"""
python3 ppo_train_escape_from_dyn.py \
  --random-map --map-size 30 \
  --regen-map-interval 10 \
  --updates 300 \
  --device cpu
"""

def parse_args():
    p = argparse.ArgumentParser(description="Escape sub-policy training using DynAvoidOneObjEnv segments (stuck 직전/탈출 구간만)")
    p.add_argument("--grid-path", default="map_grid.npy")
    p.add_argument("--waypoints-path", default="waypoints.npy")
    p.add_argument("--random-map", action="store_true")
    p.add_argument("--map-size", type=int, default=30)
    p.add_argument("--regen-map-interval", type=int, default=10, help="이 주기(업데이트 기준)마다 새 맵 생성(0이면 고정)")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--device", default="cpu")
    p.add_argument("--updates", type=int, default=300)
    p.add_argument("--rollout-steps", type=int, default=2048)
    p.add_argument("--save-interval", type=int, default=50)
    p.add_argument("--out-dir", default="checkpoints_escape_from_dyn")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--hidden-sizes", type=int, nargs="+", default=[256, 256, 128])
    p.add_argument("--feat-dim", type=int, default=256)
    p.add_argument("--pretrained", default=None)
    p.add_argument("--cell-size", type=float, default=0.20)
    p.add_argument("--pre-stuck-steps", type=int, default=12, help="stuck 직전 포함할 스텝 수")
    p.add_argument("--escape-release-steps", type=int, default=3, help="danger를 벗어난 뒤 기록 종료까지 유지할 스텝")
    # Colab Drive 연동
    p.add_argument("--mount-drive", action="store_true", help="Colab에서 Google Drive 마운트 시도")
    p.add_argument("--drive-out-dir", default=None, help="지정 시 out-dir 대신 이 경로에 저장 (예: /content/drive/MyDrive/escape_ckpt)")
    return p.parse_args()


def load_array(path):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return np.load(path)


def build_env(args, seed, grid=None, wps=None):
    if grid is None or wps is None:
        if args.random_map:
            grid, wps, _ = make_map_30(seed=seed, size=args.map_size)
        else:
            grid = load_array(args.grid_path)
            wps = load_array(args.waypoints_path)
    env = DynAvoidOneObjEnv(
        grid=grid,
        waypoints=wps,
        seed=seed,
        cell_size_m=args.cell_size,
    )
    return env


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

        # 일반 단계에서는 pre_buf에만 저장
        pre_buf.append((curr_obs, action, logprob, float(reward), done_flag, value))

        # stuck 진입 시점: pre_buf를 buffer에 플러시하고 recording 시작
        if not recording and is_stuck and not was_stuck:
            for (o, a, lp, r, d, v) in pre_buf:
                if buffer.ptr >= max_cap or collected >= cfg.rollout_steps:
                    break
                buffer.store(obs=o, action=a, logprob=lp, reward=r, done=d, value=v, mask=True)
                collected += 1
            recording = True

        # recording 중이면 현재 스텝도 저장
        if recording:
            if buffer.ptr >= max_cap or collected >= cfg.rollout_steps:
                break
            buffer.store(obs=curr_obs, action=action, logprob=logprob, reward=float(reward), done=done_flag, value=value, mask=True)
            collected += 1

        # recording 종료 조건: stuck 해제되고 ESCAPE 모드가 아니며 danger 밖 상태를 일정 스텝 유지
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

        # 에피소드 종료 처리
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

    # rollout 루프 종료 후 열려 있는 traj 처리
    if recording:
        with torch.no_grad():
            _, v_boot = model(curr_obs.unsqueeze(0))
            v_boot = v_boot.squeeze(0).squeeze(-1)
        buffer.finish_path(last_value=v_boot)

    trainer._curr_obs = curr_obs
    return collected


def main():
    args = parse_args()
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

    env = build_env(args, seed=map_seed)

    cfg = PPOConfig(
        obs_dim=env.observation_space.shape[0],
        act_dim=env.action_space.n,
        hidden_sizes=tuple(args.hidden_sizes),
        feat_dim=args.feat_dim,
        rollout_steps=args.rollout_steps,
        lr=args.lr,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=args.device,
        seed=args.seed,
    )
    trainer = PPOTrainer(env, cfg)
    if args.pretrained:
        if os.path.exists(args.pretrained):
            state = torch.load(args.pretrained, map_location=args.device)
            trainer.model.load_state_dict(state, strict=False)
            print(f"[INFO] loaded pretrained: {args.pretrained}")
        else:
            print(f"[WARN] pretrained not found: {args.pretrained}")

    print(f"[INFO] obs_dim={cfg.obs_dim}, act_dim={cfg.act_dim}, device={args.device}")
    print(f"[INFO] escape data only: pre_stuck_steps={args.pre_stuck_steps}")

    for it in range(1, args.updates + 1):
        if args.regen_map_interval > 0 and it % args.regen_map_interval == 0:
            map_seed += 1
            new_env = build_env(args, seed=map_seed)
            switch_env(trainer, new_env, trainer.device)
            print(f"[INFO] regenerated map at iter {it} (seed={map_seed})")

        steps = collect_escape_segments(
            trainer,
            cfg,
            pre_steps=args.pre_stuck_steps,
            escape_release_steps=args.escape_release_steps,
        )
        logs = trainer.update()
        info = trainer.last_info or {}
        summary = {
            "iter": it,
            "steps": steps,
            **{k: f"{v:.4f}" for k, v in (logs or {}).items()},
        }
        print("[TRAIN]", summary)

        if it % args.save_interval == 0:
            os.makedirs(out_dir, exist_ok=True)
            path = os.path.join(out_dir, f"escape_from_dyn_iter{it}.pt")
            torch.save(trainer.model.state_dict(), path)
            print(f"[CKPT] saved => {path}")

    os.makedirs(out_dir, exist_ok=True)
    final_path = os.path.join(out_dir, f"escape_from_dyn_iter{args.updates}.pt")
    torch.save(trainer.model.state_dict(), final_path)
    print(f"[DONE] saved => {final_path}")


if __name__ == "__main__":
    main()
