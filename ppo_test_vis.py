# ppo_test_vis.py
import os, random
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt

from env.dyn_env_one import DynAvoidOneObjEnv
from rl.network import ActorCritic

# 모델 아키텍처 기본값 (통합 학습 스크립트와 동일)
MAIN_HIDDEN = (512, 512, 256)
MAIN_FEAT   = 384
ESC_HIDDEN  = (512, 512, 256)
ESC_FEAT    = 384


def load_model(model, path, device="cpu"):
    if os.path.exists(path):
        print(f"[INFO] Loading trained PPO weights from {path}")
        sd = torch.load(path, map_location=device)
        model.load_state_dict(sd, strict=False)
        model.eval()
    else:
        raise FileNotFoundError(f"Checkpoint not found: {path}")


def summarize_block(info):
    stuck  = bool(info.get("stuck_state", False))
    override = bool(info.get("override_active", False))
    severity = float(info.get("block_severity", 0.0))
    block_avg = float(info.get("block_avg", 0.0))
    dyn_sev = float(info.get("block_dyn_severity", 0.0))
    static_density = float(info.get("block_static_density", 0.0))
    mm     = float(info.get("move_mean_m", 0.0))
    dnow   = info.get("goal_obj_dist_now_m", None)

    eff    = float(info.get("efficiency", 0.0))
    bkr    = float(info.get("backtrack_rate", 0.0))
    prg    = float(info.get("prog_ratio", 0.0))

    dn_txt = "-" if dnow is None else f"{dnow:.2f}m"
    line1 = f"Stuck={int(stuck)} | Override={int(override)} | Sev={severity:.2f} | Avg={block_avg:.2f}"
    line2 = f"DynSev={dyn_sev:.2f} | StaticDense={static_density:.2f} | MoveMean={mm:.3f}m/step | Goal-Obj dist={dn_txt}"
    line3 = f"Eff={eff:.2f} | Backtrack={bkr:.2f} | ProgRatio={prg:.2f}"
    return line1 + "\n" + line2 + "\n" + line3


def visualize_episode(env, model, escape_model=None, device="cpu", render_interval=0.05, max_steps=1000, seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    obs, _ = env.reset()
    done = False; trunc = False; step = 0

    grid = env.grid.copy(); H, W = grid.shape
    plt.ion(); fig, ax = plt.subplots(figsize=(6,6))
    action_map = {0: "UP", 1: "LEFT", 2: "DOWN", 3: "RIGHT", 4: "STAY"}

    while not (done or trunc) and step < max_steps:
        step += 1
        use_escape_policy = escape_model is not None and getattr(env, "escape_active", False)
        active_model = escape_model if use_escape_policy else model
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            logits, _ = active_model(obs_t)
            action = torch.argmax(logits, dim=-1).item()
        obs, reward, done, trunc, info = env.step(action)

        ax.clear()
        ax.imshow(grid, cmap="Greys", origin="upper")

        dz = getattr(env, "danger_zone_map", None)
        if dz is not None:
            soft = getattr(dz, "soft", None)
            if soft is not None and np.max(soft) > 1e-3:
                ax.imshow(soft, cmap="Oranges", origin="upper", alpha=0.35, vmin=0.0, vmax=1.0)

        # 새 CPP는 env.waypoints 위에 그대로 표시되므로 별도 선을 덧그리지 않음
        if len(env.waypoints) > 0:
            ax.plot(env.waypoints[:,0], env.waypoints[:,1], "g--", alpha=0.6, label="CPP")

        stuck_state = bool(info.get("stuck_state", False))
        if stuck_state:
            robot_color = "#FF3355"
        else:
            robot_color = "#1f77b4"
        ax.scatter(env.agent_rc[1], env.agent_rc[0], c=robot_color, s=80, label="Robot")

        if env.wp_idx < len(env.waypoints):
            gx, gy = env.waypoints[env.wp_idx]
            ax.scatter(gx, gy, c="gold", s=100, marker="*", label="Goal")

        for i, obj in enumerate(getattr(env, "dynamic_objs", [])):
            ax.scatter(obj.p[1], obj.p[0], c="red", s=50, label="Obj" if i==0 else None)

        mode = info.get('mode', '-')
        title_str = f"Step {step} | Mode={mode} | Policy={'ESC' if use_escape_policy else 'MAIN'} | R={reward:.2f}"
        if mode == "AVOID":
            title_str += f" | Policy: {action_map.get(action,'?')}"
        if stuck_state:
            title_str += " | STUCK"
        ax.set_title(title_str, loc="left", fontsize=10)

        crit_text = summarize_block(info)
        last_replan = info.get("last_replan_reason", None)
        if last_replan:
            crit_text += f"\nReplan: {last_replan}"
        ax.text(
            0.01, 0.01, crit_text,
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment="bottom",
            bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.5)
        )

        ax.set_xlim(0, W); ax.set_ylim(H, 0)
        ax.legend(loc="upper right")
        plt.pause(render_interval)

    plt.ioff(); plt.show()

    
def main():
    parser = argparse.ArgumentParser(description="메인/ESC 정책 시각화")
    parser.add_argument("--ckpt", default="checkpoints_integrated/main_iter500.pt")
    parser.add_argument("--escape-ckpt", default="checkpoints_integrated/escape_iter500.pt", help="ESC 서브 정책 checkpoint")
    parser.add_argument("--grid-path", default="map_grid.npy")
    parser.add_argument("--waypoints-path", default="waypoints.npy")
    parser.add_argument("--seed", type=int, default=47)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--render-interval", type=float, default=0.01)
    parser.add_argument("--max-steps", type=int, default=1500)
    args = parser.parse_args()

    seed = args.seed
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

    if not (os.path.exists(args.grid_path) and os.path.exists(args.waypoints_path)):
        raise FileNotFoundError("map_grid.npy / waypoints.npy 가 필요합니다.")
    grid = np.load(args.grid_path); wps = np.load(args.waypoints_path)

    use_escape = args.escape_ckpt is not None
    env = DynAvoidOneObjEnv(grid=grid, waypoints=wps, seed=seed, cell_size_m=0.20, use_escape_subpolicy=use_escape)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    device = args.device
    model = ActorCritic(obs_dim=obs_dim, act_dim=act_dim, hidden_sizes=MAIN_HIDDEN, feat_dim=MAIN_FEAT).to(device)
    load_model(model, args.ckpt, device)

    escape_model = None
    if use_escape:
        escape_model = ActorCritic(obs_dim=obs_dim, act_dim=act_dim, hidden_sizes=ESC_HIDDEN, feat_dim=ESC_FEAT).to(device)
        load_model(escape_model, args.escape_ckpt, device)

    visualize_episode(env, model, escape_model=escape_model, device=device, render_interval=args.render_interval, max_steps=args.max_steps, seed=seed)


if __name__ == "__main__":
    main()
