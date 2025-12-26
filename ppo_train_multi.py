import os
import argparse
import numpy as np
import torch
from typing import Dict, Tuple, List, Callable
from torch.distributions import Categorical
from collections import deque

from env.dyn_env_one import DynAvoidOneObjEnv
from env.env import make_map_30
from rl.ppo import PPOConfig
from rl.ppo_multi import PPOTrainerMulti
from rl.vec_env import SubprocVecEnv

"""
# Windows PowerShell Execution Command (CUDA, 6 Envs)
python ppo_train_multi.py `
  --random-map --map-size 30 --regen-map-interval 10 `
  --escape-updates 300 --main-every 1 --main-updates-per-escape 1 `
  --rollout-steps 4096 --batch-size 256 --lr 2e-4 --device cuda `
  --num-envs 6
"""

# Re-use curriculum logic
from ppo_train_integrated_random import curriculum, make_spawn_fn, load_array, detect_device, build_map

def parse_args():
    p = argparse.ArgumentParser(description="PPO Multi-Env Trainer")
    p.add_argument("--random-map", action="store_true")
    p.add_argument("--map-size", type=int, default=30)
    p.add_argument("--regen-map-interval", type=int, default=10, help="Regen map interval")
    p.add_argument("--grid-path", default="map_grid.npy")
    p.add_argument("--waypoints-path", default="waypoints.npy")
    p.add_argument("--escape-updates", type=int, default=300)
    p.add_argument("--main-every", type=int, default=1, help="Main update interval")
    p.add_argument("--main-updates-per-escape", type=int, default=1)
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=1234)
    # PPO Common
    p.add_argument("--rollout-steps", type=int, default=2048)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--main-hidden-sizes", type=int, nargs="+", default=[256, 256, 128])
    p.add_argument("--escape-hidden-sizes", type=int, nargs="+", default=[256, 256, 128])
    p.add_argument("--main-feat-dim", type=int, default=256)
    p.add_argument("--escape-feat-dim", type=int, default=256)
    # Path/Save
    p.add_argument("--out-dir", default="checkpoints_integrated_random")
    p.add_argument("--save-interval", type=int, default=50)
    p.add_argument("--pretrained-main", default="checkpoints_integrated_random/main_iter300.pt")
    p.add_argument("--pretrained-escape", default="checkpoints_integrated_random/escape_iter300.pt")
    # Drive
    p.add_argument("--mount-drive", action="store_true")
    p.add_argument("--drive-out-dir", default=None)
    
    # [Added]
    p.add_argument("--num-envs", type=int, default=6, help="Number of parallel environments")
    
    return p.parse_args()

def make_env_fn(grid, wps, seed, use_escape=False, cell_size=0.20):
    def _thunk():
        env = DynAvoidOneObjEnv(
            grid=grid,
            waypoints=wps,
            seed=seed,
            cell_size_m=cell_size,
            use_escape_subpolicy=use_escape,
        )
        return env
    return _thunk

def collect_escape_segments_multi(trainer: PPOTrainerMulti, cfg, pre_steps=12, escape_release_steps=3):
    env = trainer.env
    model = trainer.model
    buffers = trainer.buffers
    device = trainer.device
    num_envs = trainer.num_envs
    
    # Clear all buffers
    for buf in buffers:
        buf.clear()

    # Per-env state
    pre_bufs = [deque(maxlen=pre_steps) for _ in range(num_envs)]
    recording = [False] * num_envs
    escape_out_counters = [0] * num_envs
    prev_infos = [{} for _ in range(num_envs)]
    
    # We need to collect enough data. 
    # Since we filter data, we might need to run longer than rollout_steps.
    # But to avoid infinite loops, we'll limit by total steps or collected samples.
    
    total_collected = 0
    steps_run = 0
    # Limit the maximum iterations to prevent blocking the main training for too long.
    # 3000 iterations * 6 envs = 18,000 steps. 
    # If stuck rate is low, we might get a small batch, but that's better than hanging.
    max_steps_run = 3000 
    
    # Ensure initial obs
    if not hasattr(trainer, "_curr_obs"):
        obs = env.reset()
        trainer._curr_obs = torch.as_tensor(obs, dtype=torch.float32, device=device)
    
    curr_obs = trainer._curr_obs # (N, obs_dim)

    # Calculate effective target based on buffer capacity
    target_steps = trainer.steps_per_env * num_envs

    while total_collected < target_steps and steps_run < max_steps_run:
        steps_run += 1
        if steps_run % 500 == 0:
            print(f"[DEBUG] Collecting escape segments: steps_run={steps_run}, collected={total_collected}/{target_steps}")
        
        # 1. Inference
        with torch.no_grad():
            logits, values = model(curr_obs)
            values = values.squeeze(-1)
            dist = Categorical(logits=logits)
            actions = dist.sample()
            logprobs = dist.log_prob(actions)
            
        # 2. Step
        next_obs, rewards, dones, infos = env.step(actions.cpu().numpy())
        
        # 3. Process per env
        for i in range(num_envs):
            # If buffer full, skip
            if buffers[i].full:
                continue
                
            done_flag = dones[i]
            info = infos[i]
            prev_info = prev_infos[i]
            
            is_stuck = bool(info.get("stuck_state", False))
            was_stuck = bool(prev_info.get("stuck_state", False))
            mode = info.get("mode", "")
            
            # Store in pre-buf
            pre_bufs[i].append((
                curr_obs[i], 
                actions[i], 
                logprobs[i], 
                float(rewards[i]), 
                done_flag, 
                values[i]
            ))
            
            # Trigger recording
            if not recording[i] and is_stuck and not was_stuck:
                # Flush pre-buf
                for (o, a, lp, r, d, v) in pre_bufs[i]:
                    if buffers[i].full: break
                    buffers[i].store(obs=o, action=a, logprob=lp, reward=r, done=d, value=v, mask=True)
                    total_collected += 1
                recording[i] = True
                
            # If recording, store current step
            if recording[i]:
                if not buffers[i].full:
                    buffers[i].store(
                        obs=curr_obs[i], 
                        action=actions[i], 
                        logprob=logprobs[i], 
                        reward=float(rewards[i]), 
                        done=done_flag, 
                        value=values[i], 
                        mask=True
                    )
                    total_collected += 1
            
            # Check stop recording condition
            if recording[i]:
                if not is_stuck and mode != "ESCAPE":
                    escape_out_counters[i] += 1
                    if escape_out_counters[i] >= escape_release_steps:
                        recording[i] = False
                        escape_out_counters[i] = 0
                        # Finish path with bootstrap value
                        # We need value of next_obs[i]
                        with torch.no_grad():
                            _, v_boot = model(torch.as_tensor(next_obs[i], dtype=torch.float32, device=device).unsqueeze(0))
                            v_boot = v_boot.squeeze().item()
                        buffers[i].finish_path(last_value=torch.tensor(v_boot, device=device))
                        pre_bufs[i].clear()
                else:
                    escape_out_counters[i] = 0
            
            # Handle done
            if done_flag:
                if recording[i]:
                    buffers[i].finish_path(last_value=torch.zeros((), device=device))
                    recording[i] = False
                    escape_out_counters[i] = 0
                    pre_bufs[i].clear()
                # Reset handled by SubprocVecEnv, next_obs[i] is new obs
                prev_infos[i] = {}
            else:
                prev_infos[i] = info

        curr_obs = torch.as_tensor(next_obs, dtype=torch.float32, device=device)
        trainer._curr_obs = curr_obs

    # Finish any open paths in buffers
    with torch.no_grad():
        logits, v_boots = model(curr_obs)
        v_boots = v_boots.squeeze(-1)
        
    for i in range(num_envs):
        if recording[i]:
             buffers[i].finish_path(last_value=v_boots[i])

    return total_collected

def main():
    args = parse_args()
    
    device = detect_device(args.device)
    map_seed = args.seed

    out_dir = args.out_dir
    if args.drive_out_dir:
        out_dir = args.drive_out_dir
    os.makedirs(out_dir, exist_ok=True)

    grid, wps = build_map(args, map_seed)
    
    # Create Vector Envs
    env_fns_main = [make_env_fn(grid, wps, seed=args.seed + i, use_escape=False) for i in range(args.num_envs)]
    env_fns_escape = [make_env_fn(grid, wps, seed=args.seed + 100 + i, use_escape=True) for i in range(args.num_envs)]
    
    vec_env_main = SubprocVecEnv(env_fns_main)
    vec_env_escape = SubprocVecEnv(env_fns_escape)

    # Get spaces from first env
    obs_dim = vec_env_main.observation_space.shape[0]
    act_dim = vec_env_main.action_space.n

    cfg_main = PPOConfig(
        obs_dim=obs_dim,
        act_dim=act_dim,
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
        obs_dim=obs_dim,
        act_dim=act_dim,
        hidden_sizes=tuple(args.escape_hidden_sizes),
        feat_dim=args.escape_feat_dim,
        rollout_steps=max(args.rollout_steps // 4, 256), # Reduce requirement for escape policy (sparse data)
        lr=args.lr,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=device,
        seed=args.seed + 999,
    )

    trainer_main = PPOTrainerMulti(vec_env_main, cfg_main)
    trainer_escape = PPOTrainerMulti(vec_env_escape, cfg_escape)

    if args.pretrained_main and os.path.exists(args.pretrained_main):
        trainer_main.model.load_state_dict(torch.load(args.pretrained_main, map_location=device), strict=False)
        print(f"[INFO] loaded main pretrained: {args.pretrained_main}")
    if args.pretrained_escape and os.path.exists(args.pretrained_escape):
        trainer_escape.model.load_state_dict(torch.load(args.pretrained_escape, map_location=device), strict=False)
        print(f"[INFO] loaded escape pretrained: {args.pretrained_escape}")

    print(f"[INFO] device={device}, escape_updates={args.escape_updates}, main_every={args.main_every}, num_envs={args.num_envs}")

    def apply_curriculum(it_num: int):
        print(f"[DEBUG] Applying curriculum for iter {it_num}...")
        obj_k, type_probs, easy_mix = curriculum(it_num, args.escape_updates)
        spawn_fn = make_spawn_fn(obj_k, type_probs)
        
        # Update spawn_fn in all remote envs
        # We use set_attr. But spawn_fn is a closure.
        # SubprocVecEnv uses cloudpickle, so it should work.
        print("[DEBUG] Sending spawn_fn to main envs...")
        vec_env_main.set_attr('_default_spawn', spawn_fn)
        print("[DEBUG] Sending spawn_fn to escape envs...")
        vec_env_escape.set_attr('_default_spawn', spawn_fn)
        print("[DEBUG] Curriculum applied.")
        
        return obj_k, type_probs, easy_mix

    apply_curriculum(1)

    for it in range(1, args.escape_updates + 1):
        if args.regen_map_interval > 0 and it % args.regen_map_interval == 0:
            map_seed += 1
            grid, wps = build_map(args, map_seed)
            
            # Re-create envs? Or just update grid/wps?
            # Updating grid/wps in existing envs is hard because they are initialized in __init__.
            # Easier to close and recreate vec envs.
            vec_env_main.close()
            vec_env_escape.close()
            
            env_fns_main = [make_env_fn(grid, wps, seed=map_seed + i, use_escape=False) for i in range(args.num_envs)]
            env_fns_escape = [make_env_fn(grid, wps, seed=map_seed + 100 + i, use_escape=True) for i in range(args.num_envs)]
            
            vec_env_main = SubprocVecEnv(env_fns_main)
            vec_env_escape = SubprocVecEnv(env_fns_escape)
            
            trainer_main.env = vec_env_main
            trainer_escape.env = vec_env_escape
            
            # Reset obs
            obs_main = vec_env_main.reset()
            trainer_main._curr_obs = torch.as_tensor(obs_main, dtype=torch.float32, device=device)
            
            obs_escape = vec_env_escape.reset()
            trainer_escape._curr_obs = torch.as_tensor(obs_escape, dtype=torch.float32, device=device)
            
            # Re-apply curriculum (spawn_fn) to new envs
            apply_curriculum(it)
            
            print(f"[INFO] regenerated map at escape iter {it} (seed={map_seed})")

        apply_curriculum(it)

        steps_escape = collect_escape_segments_multi(trainer_escape, cfg_escape, pre_steps=12, escape_release_steps=3)
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
    
    vec_env_main.close()
    vec_env_escape.close()

if __name__ == "__main__":
    # Support multiprocessing on macOS/Windows
    import multiprocessing
    multiprocessing.set_start_method("spawn", force=True)
    main()
