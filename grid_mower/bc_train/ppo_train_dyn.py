# ppo_train_dyn.py
import os
import torch
import numpy as np

from env.dyn_env_one import DynAvoidOneObjEnv
from rl.network import ActorCritic
from rl.ppo import PPOConfig, PPOTrainer  # 네가 만든 버전 그대로 사용
# from rl.network import ActorCritic  # PPOTrainer 내부에서 import했다면 불필요
def maybe_load_pretrained(model: torch.nn.Module, path: str):
    if path and os.path.exists(path):
        print(f"[INFO] Loading pretrained weights from: {path}")
        sd = torch.load(path, map_location="cpu")
        # 공유 백본 + 액터/크리틱 헤드 키가 정확히 같다면 strict=True 사용해도 OK
        model.load_state_dict(sd, strict=False)
    else:
        print(f"[WARN] Pretrained path not found: {path}. Start from scratch.")

def evaluate_once(env, model, episodes=1, render=False):
    model.eval()
    device = next(model.parameters()).device
    total = 0.0
    with torch.no_grad():
        for _ in range(episodes):
            obs, _ = env.reset()
            ep_ret = 0.0
            done = False
            trunc = False
            while not (done or trunc):
                obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                logits, _ = model(obs_t)
                action = torch.argmax(logits, dim=-1).item()
                obs, r, done, trunc, _ = env.step(action)
                ep_ret += float(r)
                if render:
                    env.render()
            total += ep_ret
    model.train()
    return total / episodes

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
grid_path = os.path.join(BASE_DIR, "map_grid.npy")
wps_path = os.path.join(BASE_DIR, "waypoints.npy")
PRETRAIN_PATH = os.path.join(BASE_DIR, "bc_actorcritic.pth")

print("[DEBUG] cwd:", os.getcwd())
print("[DEBUG] map exists?", os.path.exists("map_grid.npy"))
print("[DEBUG] wps exists?", os.path.exists("waypoints.npy"))

def main():
    # ---------- 0) (선택) 저장된 맵/웨이포인트 사용 ----------
    grid_path = "map_grid.npy"
    wps_path  = "waypoints.npy"
    if os.path.exists(grid_path) and os.path.exists(wps_path):
        grid = np.load(grid_path)
        wps  = np.load(wps_path)
        env = DynAvoidOneObjEnv(grid=grid, waypoints=wps, seed=0)
        print("[ENV] Loaded grid/waypoints from files.")
    else:
        # 파일이 없다면 내부에서 랜덤 맵/CPP로 세팅하는 환경이어야 함
        env = DynAvoidOneObjEnv(seed=0)
        print("[ENV] Using internal random map (no saved files).")

    # ---------- 1) 관측/행동 차원 ----------
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    print(f"[ENV] obs_dim={obs_dim}, act_dim={act_dim}")

    # ---------- 2) PPO 설정 ----------
    cfg = PPOConfig(
        obs_dim=obs_dim,
        act_dim=act_dim,
        rollout_steps=4096,
        lr=3e-4,
        epochs=10,
        batch_size=256,
        clip_eps=0.2,
        vf_coef=0.5,
        ent_coef=0.01,
        max_grad_norm=0.5,
        gamma=0.99,
        gae_lambda=0.95,
        seed=0,
        device="cpu",  # 가능하면 "cuda"
    )

    # ---------- 3) 트레이너 ----------
    trainer = PPOTrainer(env, cfg)

    # (선택) BC 사전학습 가중치
    PRETRAIN_PATH = "bc_actorcritic.pth"
    maybe_load_pretrained(trainer.model, PRETRAIN_PATH)

    # ---------- 4) 학습 루프 ----------
    total_iters = 300
    eval_every  = 10
    save_every  = 50
    save_dir    = "checkpoints_dyn"
    os.makedirs(save_dir, exist_ok=True)

    for it in range(1, total_iters + 1):
        steps = trainer.collect_rollout()   # cfg.rollout_steps 만큼 샘플 수집
        logs  = trainer.update()            # PPO 업데이트(여러 epoch/미니배치)

        if it % eval_every == 0:
            print(
                f"[Iter {it:04d}] steps={steps} "
                f"loss={logs.get('loss', 0):.4f} "
                f"pi={logs.get('policy_loss', 0):.4f} "
                f"vf={logs.get('value_loss', 0):.4f} "
                f"ent={logs.get('entropy', 0):.3f} "
                f"kl={logs.get('approx_kl', 0):.4f} "
                f"clipfrac={logs.get('clipfrac', 0):.3f}"
            )

        if it % save_every == 0:
            path = os.path.join(save_dir, f"ppo_dyn_iter{it}.pt")
            torch.save(trainer.model.state_dict(), path)
            print(f"[SAVE] {path}")

    print("[DONE] Training finished.")


if __name__ == "__main__":
    main()