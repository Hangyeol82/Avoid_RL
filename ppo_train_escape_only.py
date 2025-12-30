import os
import argparse
import numpy as np
import torch
from env.escape_env import EscapeTrainingEnv
from env.env import make_map_30
from rl.ppo import PPOConfig
from rl.ppo_multi import PPOTrainerMulti
from rl.vec_env import SubprocVecEnv

"""
[Escape Policy 집중 훈련 스크립트 (Multi-Env)]
python3 ppo_train_escape_only.py --device cpu --rollout-steps 2048 --epochs 10 --num-envs 4
"""

def parse_args():
    p = argparse.ArgumentParser(description="Escape Policy 집중 훈련 (Multi-Env)")
    p.add_argument("--random-map", action="store_true")
    p.add_argument("--map-size", type=int, default=30)
    p.add_argument("--grid-path", default="map_grid.npy")
    p.add_argument("--waypoints-path", default="waypoints.npy")
    
    p.add_argument("--total-iters", type=int, default=500, help="총 학습 반복 횟수")
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=42)
    
    # PPO 하이퍼파라미터
    p.add_argument("--rollout-steps", type=int, default=2048)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--hidden-sizes", type=int, nargs="+", default=[256, 256, 128])
    p.add_argument("--feat-dim", type=int, default=256)
    
    # 저장/로드
    p.add_argument("--out-dir", default="checkpoints_escape_only")
    p.add_argument("--save-interval", type=int, default=50)
    p.add_argument("--pretrained", default=None, help="이전 학습된 모델 경로 (선택)")
    
    # 병렬 처리
    p.add_argument("--num-envs", type=int, default=4, help="병렬 환경 개수")
    
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

def make_env_fn(grid, wps, seed, cell_size=0.20):
    def _thunk():
        env = EscapeTrainingEnv(
            grid=grid,
            waypoints=wps,
            seed=seed,
            cell_size_m=cell_size,
        )
        return env
    return _thunk

def main():
    args = parse_args()
    device = detect_device(args.device)
    os.makedirs(args.out_dir, exist_ok=True)
    
    print(f"[INFO] Escape Training Started on {device} with {args.num_envs} envs")
    
    # 1. 맵 및 환경 생성
    grid, wps = build_map(args, args.seed)
    
    # Multi-Env 생성
    env_fns = [make_env_fn(grid, wps, seed=args.seed + i) for i in range(args.num_envs)]
    vec_env = SubprocVecEnv(env_fns)
    
    # 2. PPO 설정
    # SubprocVecEnv의 observation_space는 첫 번째 환경의 것을 가져옴
    obs_dim = vec_env.observation_space.shape[0]
    act_dim = vec_env.action_space.n
    
    cfg = PPOConfig(
        obs_dim=obs_dim,
        act_dim=act_dim,
        hidden_sizes=tuple(args.hidden_sizes),
        feat_dim=args.feat_dim,
        rollout_steps=args.rollout_steps,
        lr=args.lr,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=device,
        seed=args.seed,
    )
    
    trainer = PPOTrainerMulti(vec_env, cfg)
    
    # 3. 모델 로드 (있으면)
    if args.pretrained and os.path.exists(args.pretrained):
        trainer.model.load_state_dict(torch.load(args.pretrained, map_location=device))
        print(f"[INFO] Loaded pretrained model: {args.pretrained}")
        
    # 4. 학습 루프
    for it in range(1, args.total_iters + 1):
        # 데이터 수집 (모든 에피소드가 탈출 시나리오)
        steps = trainer.collect_rollout()
        
        # 업데이트
        logs = trainer.update()
        
        # 로그 출력
        # PPOTrainerMulti는 last_info에 ep_return을 저장하지 않으므로 loss만 출력
        print(f"[Iter {it:4d}] Steps: {steps}, "
              f"Loss: {logs.get('loss', 0):.4f}, "
              f"PolLoss: {logs.get('policy_loss', 0):.4f}, "
              f"ValLoss: {logs.get('value_loss', 0):.4f}")
        
        # 저장
        if it % args.save_interval == 0:
            path = os.path.join(args.out_dir, f"escape_iter{it}.pt")
            torch.save(trainer.model.state_dict(), path)
            print(f"[SAVE] Saved to {path}")
            
    # 최종 저장
    final_path = os.path.join(args.out_dir, "escape_final.pt")
    torch.save(trainer.model.state_dict(), final_path)
    print(f"[DONE] Training finished. Saved to {final_path}")
    
    vec_env.close()

if __name__ == "__main__":
    # Support multiprocessing on macOS/Windows
    import multiprocessing
    multiprocessing.set_start_method("spawn", force=True)
    main()
