# ppo_train_dyn.py
import os
import numpy as np
import torch
from typing import Optional

from env.dyn_env_one import DynAvoidOneObjEnv
from rl.ppo import PPOConfig, PPOTrainer

# =============================================================================
#  (수정) 전이학습 설정: 불러올 모델 경로 (없으면 None)
# =============================================================================
# 예시: "checkpoints_dyn/ppo_dyn_best.pt" 또는 "checkpoints_dyn/ppo_dyn_iter500.pt"
PRETRAINED_MODEL_PATH: Optional[str] = "checkpoints_dyn/200_i.pt"
# =============================================================================

def main():
    grid_path = "map_grid.npy"
    wps_path  = "waypoints.npy"

    if os.path.exists(grid_path) and os.path.exists(wps_path):
        grid = np.load(grid_path)
        wps  = np.load(wps_path)
        seed = int(np.random.randint(1, 1e9))
        env = DynAvoidOneObjEnv(grid=grid, waypoints=wps, seed=seed, cell_size_m=0.20)
        print(f"[ENV] Random training seed: {seed}")
        print("[ENV] Loaded grid/waypoints from files.")
    else:
        raise FileNotFoundError("map_grid.npy / waypoints.npy 가 필요합니다.")

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
        hidden_sizes=(128,128),
        feat_dim=128
    )

    trainer = PPOTrainer(env, cfg)

    # =========================================================================
    #  (수정) 전이학습 모델 불러오기
    # =========================================================================
    if PRETRAINED_MODEL_PATH and os.path.exists(PRETRAINED_MODEL_PATH):
        try:
            # 디바이스 설정에 맞게 불러오기
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

    total_iters = 2000
    eval_every  = 10
    save_every  = 50
    save_dir    = "checkpoints_dyn"
    os.makedirs(save_dir, exist_ok=True)

    for it in range(1, total_iters+1):
        steps = trainer.collect_rollout()
        logs  = trainer.update()

        if it % eval_every == 0:
            print(f"[Iter {it:04d}] steps={steps} pi={logs.get('policy_loss',0):.4f} "
                  f"vf={logs.get('value_loss',0):.4f} ent={logs.get('entropy',0):.3f}")

        if it % save_every == 0:
            path = os.path.join(save_dir, f"ppo_dyn_iter{it}.pt")
            torch.save(trainer.model.state_dict(), path)
            print(f"[SAVE] {path}")

    print("[DONE] PPO Training finished.")

if __name__ == "__main__":
    main()