# test_dyn_policy.py
import os
import time
import numpy as np
import torch

from env.dyn_env_one import DynAvoidOneObjEnv
from rl.network import ActorCritic

def load_env():
    grid_path = "map_grid.npy"
    wps_path  = "waypoints.npy"
    if os.path.exists(grid_path) and os.path.exists(wps_path):
        grid = np.load(grid_path)
        wps  = np.load(wps_path)
        print("[ENV] Loaded grid/waypoints from files.")
        return DynAvoidOneObjEnv(grid=grid, waypoints=wps, seed=42)
    print("[ENV] Using internal random map.")
    return DynAvoidOneObjEnv(seed=42)

def load_model(env, ckpt_path, device="cpu"):
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    model = ActorCritic(obs_dim=obs_dim, act_dim=act_dim,
                        hidden_sizes=(128,128), feat_dim=128).to(device)
    sd = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(sd, strict=False)
    model.eval()
    print(f"[MODEL] Loaded: {ckpt_path}")
    return model

@torch.no_grad()
def run_eval(env, model, episodes=5, render=False, deterministic=True, device="cpu"):
    total_return = 0.0
    successes, collisions, timeouts = 0, 0, 0

    for ep in range(episodes):
        obs, _ = env.reset()
        ep_ret = 0.0
        done = False; trunc = False
        step = 0

        while not (done or trunc):
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            logits, _ = model(obs_t)

            if deterministic:
                action = torch.argmax(logits, dim=-1).item()
            else:
                # 확률적으로 샘플 (탐색 성능 확인용)
                probs = torch.softmax(logits, dim=-1)
                action = torch.distributions.Categorical(probs).sample().item()

            obs, r, done, trunc, info = env.step(action)
            ep_ret += float(r)
            step += 1

            if render:
                env.render()
                time.sleep(0.02)

        total_return += ep_ret
        # 간단한 성공/실패 판정 규칙(환경의 info 키를 따름)
        if info.get("success", False):
            successes += 1
        elif info.get("collision", False):
            collisions += 1
        else:
            timeouts += 1

        print(f"[EP {ep+1}/{episodes}] return={ep_ret:.2f}, "
              f"success={info.get('success', False)}, "
              f"collision={info.get('collision', False)}, "
              f"steps={step}")

    avg_return = total_return / episodes
    print("\n==== EVAL SUMMARY ====")
    print(f"Episodes     : {episodes}")
    print(f"Avg Return   : {avg_return:.2f}")
    print(f"Success Rate : {successes/episodes:.2%} ({successes})")
    print(f"Collisions   : {collisions}")
    print(f"Timeouts     : {timeouts}")
    return avg_return

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=False,
                        default="checkpoints_dyn/ppo_dyn_iter300.pt",
                        help="학습된 모델 가중치 경로")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--deterministic", action="store_true",
                        help="argmax 행동 (미지정 시 확률 샘플)")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    env = load_env()
    model = load_model(env, ckpt_path=args.ckpt, device=args.device)

    # 평가 실행
    run_eval(env, model,
             episodes=args.episodes,
             render=args.render,
             deterministic=args.deterministic,
             device=args.device)

if __name__ == "__main__":
    main()