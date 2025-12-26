import os
import numpy as np
import torch
import argparse
from collections import defaultdict
import pandas as pd
from tqdm import tqdm
import multiprocessing as mp
import time

from env.dyn_env_one import DynAvoidOneObjEnv
from rl.network import ActorCritic
from planning.cpp import CoveragePlanner, HeuristicType
from env.moving_object import MovingObj
from utils_timing import estimate_robot_timeline

# 모델 설정
MAIN_HIDDEN = (256, 256, 128)
MAIN_FEAT   = 256
ESC_HIDDEN  = (256, 256, 128)
ESC_FEAT    = 256

# --- Global Variables for Worker Processes ---
MAIN_POLICY = None
ESCAPE_POLICY = None
DEVICE = None

def make_custom_map(seed, size=30, num_obstacles=3):
    """장애물 개수를 지정하여 맵 생성"""
    rng = np.random.default_rng(seed)
    g = np.zeros((size, size), dtype=int)

    # 장애물 생성
    for _ in range(num_obstacles):
        h = rng.integers(3, 6)
        w = rng.integers(3, 6)
        r0 = rng.integers(2, size - h - 2)
        c0 = rng.integers(2, size - w - 2)
        g[r0:r0 + h, c0:c0 + w] = 1

    # 시작 위치
    start = None
    for r in range(size):
        for c in range(size):
            if g[r, c] == 0:
                g[r, c] = 2
                start = (r, c)
                break
        if start:
            break

    # CPP 경로 생성
    cp = CoveragePlanner(g)
    cp.start(initial_orientation=0, cp_heuristic=HeuristicType.VERTICAL)
    cp.compute()
    _, _, _, traj, xy = cp.result()
    waypoints = np.array([(t[2], t[1]) for t in traj], dtype=np.int32)

    return g, waypoints

class FixedSpawner:
    """Pickle 가능한 스포너 클래스"""
    def __init__(self, num_objects):
        self.num_objects = num_objects
        
    def __call__(self, occ_grid, waypoints, rng, v_robot=1.0, v_obj_range=(0.3, 1.1), **kwargs):
        H, W = occ_grid.shape
        objs = []
        t_robot = estimate_robot_timeline(waypoints, v_robot_cells_per_step=v_robot)
        candidates = ["cv", "random_walk", "sin", "circle"]
        alias = {"random_walk":"ou", "sin":"patrol", "circle":"patrol"}

        for _ in range(self.num_objects):
            raw_kind = rng.choice(candidates)
            kind = alias.get(raw_kind, raw_kind)
            for _ in range(50):
                sy = int(rng.integers(1, H-1))
                sx = int(rng.integers(1, W-1))
                if occ_grid[sy, sx] == 0:
                    vmag = rng.uniform(*v_obj_range)
                    theta = rng.uniform(0, 2*np.pi)
                    vy, vx = np.sin(theta)*vmag, np.cos(theta)*vmag
                    obj = MovingObj(
                        pos=(float(sy), float(sx)),
                        vel=np.array([vy, vx], float),
                        vmax=max(v_obj_range[1], 1.2),
                        kind=kind,
                        seed=int(rng.integers(1e9))
                    )
                    if kind == "patrol":
                        pts = [(sy, sx), (sy, min(W-2, sx+5))]
                        obj.set_patrol([(float(py), float(px)) for (py, px) in pts])
                    objs.append(obj)
                    break
        return objs

def init_worker(ckpt, escape_ckpt, device):
    """워커 프로세스 초기화: 모델 로드"""
    global MAIN_POLICY, ESCAPE_POLICY, DEVICE
    DEVICE = device
    
    # Obs dim 하드코딩 (환경 생성 없이)
    # MLP(106) + Map(225) = 331
    obs_dim = 331
    act_dim = 5
    
    MAIN_POLICY = ActorCritic(obs_dim, act_dim, MAIN_HIDDEN, MAIN_FEAT).to(device)
    MAIN_POLICY.load_state_dict(torch.load(ckpt, map_location=device), strict=False)
    MAIN_POLICY.eval()
    
    if escape_ckpt:
        ESCAPE_POLICY = ActorCritic(obs_dim, act_dim, ESC_HIDDEN, ESC_FEAT).to(device)
        ESCAPE_POLICY.load_state_dict(torch.load(escape_ckpt, map_location=device), strict=False)
        ESCAPE_POLICY.eval()

def run_episode(task_args):
    """단일 에피소드 실행 함수"""
    sc_name, n_obs, n_obj, seed, max_steps = task_args
    global MAIN_POLICY, ESCAPE_POLICY, DEVICE
    
    # 맵 생성
    grid, wps = make_custom_map(seed, size=30, num_obstacles=n_obs)
    if len(wps) == 0:
        return None # CPP 실패

    env = DynAvoidOneObjEnv(
        grid=grid, 
        waypoints=wps, 
        seed=seed, 
        use_escape_subpolicy=(ESCAPE_POLICY is not None),
        local_map_size=15
    )
    env._default_spawn = FixedSpawner(n_obj)
    
    obs, _ = env.reset()
    done = False
    ep_steps = 0
    final_reward = 0.0
    
    while not done and ep_steps < max_steps:
        ep_steps += 1
        use_escape = (ESCAPE_POLICY is not None) and getattr(env, "escape_active", False)
        active_model = ESCAPE_POLICY if use_escape else MAIN_POLICY
        
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        with torch.no_grad():
            logits, _ = active_model(obs_t)
            action = torch.argmax(logits, dim=-1).item()
        
        obs, reward, done, _, info = env.step(action)
        if done:
            final_reward = reward

    # --- 결과 집계 ---
    res = {
        "scenario": sc_name,
        "steps": ep_steps,
        "success": 0,
        "coll_dyn": 0,
        "coll_static": 0,
        "timeout": 0,
        "coverage": 0.0
    }

    # 커버리지 계산
    visited_count = np.sum(env.visited)
    total_wps = len(env.waypoints)
    
    # 유효 커버리지 (위험 지역 제외)
    unvisited_indices = np.where(~env.visited)[0]
    blocked_wps = 0
    if getattr(env, "danger_zone_map", None) is not None:
        soft = getattr(env.danger_zone_map, "soft", None)
        if soft is not None:
            for idx in unvisited_indices:
                wx, wy = env.waypoints[idx]
                r, c = int(wy), int(wx)
                if 0 <= r < env.H and 0 <= c < env.W:
                    if soft[r, c] >= 0.4:
                        blocked_wps += 1
    
    effective_total = total_wps - blocked_wps
    if effective_total > 0:
        coverage_ratio = visited_count / effective_total
    else:
        coverage_ratio = 1.0
    
    res["coverage"] = coverage_ratio * 100.0

    if done:
        if final_reward >= 1.0:
            res["success"] = 1
        elif final_reward <= -1.5:
            res["coll_dyn"] = 1
        else:
            res["coll_static"] = 1
    else:
        # Soft Success: 타임아웃이어도 95% 이상이면 성공
        if coverage_ratio >= 0.95:
            res["success"] = 1
        else:
            res["timeout"] = 1
            
    return res

def evaluate(args):
    # Multiprocessing 설정 (macOS/Linux spawn/fork 이슈 대응)
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    print(f"[INFO] Evaluating on {args.device} with {args.cores} cores")
    
    scenarios = [
        {"name": "Easy (Obs=3, Obj=1)",   "n_obs": 3, "n_obj": 1},
        {"name": "Medium (Obs=5, Obj=1)", "n_obs": 5, "n_obj": 1},
        {"name": "Hard (Obs=5, Obj=2)",   "n_obs": 5, "n_obj": 2},
    ]

    # 작업 목록 생성
    tasks = []
    for sc in scenarios:
        for i in range(args.episodes):
            # (시나리오이름, 장애물수, 객체수, 시드, 최대스텝)
            tasks.append((sc["name"], sc["n_obs"], sc["n_obj"], args.seed + i, args.max_steps))
    
    print(f"[INFO] Total tasks: {len(tasks)} (Episodes per scenario: {args.episodes})")
    
    # 병렬 처리 시작
    results_raw = []
    with mp.Pool(processes=args.cores, initializer=init_worker, initargs=(args.ckpt, args.escape_ckpt, args.device)) as pool:
        # tqdm으로 진행상황 표시
        for res in tqdm(pool.imap_unordered(run_episode, tasks), total=len(tasks)):
            if res is not None:
                results_raw.append(res)
    
    # 결과 정리
    grouped = defaultdict(lambda: defaultdict(list))
    for r in results_raw:
        sc = r["scenario"]
        grouped[sc]["success"].append(r["success"])
        grouped[sc]["coll_dyn"].append(r["coll_dyn"])
        grouped[sc]["coll_static"].append(r["coll_static"])
        grouped[sc]["timeout"].append(r["timeout"])
        grouped[sc]["steps"].append(r["steps"])
        grouped[sc]["coverage"].append(r["coverage"])

    # 표 생성
    summary = []
    # 시나리오 순서 보장
    sc_order = [s["name"] for s in scenarios]
    
    for sc_name in sc_order:
        if sc_name not in grouped:
            continue
        data = grouped[sc_name]
        total = len(data["success"])
        
        row = {
            "Scenario": sc_name,
            "Success Rate": f"{sum(data['success'])/total*100:.1f}%",
            "Coll (Dynamic)": f"{sum(data['coll_dyn'])/total*100:.1f}%",
            "Coll (Static)": f"{sum(data['coll_static'])/total*100:.1f}%",
            "Timeout": f"{sum(data['timeout'])/total*100:.1f}%",
            "Avg Steps": f"{np.mean(data['steps']):.1f}",
            "Avg Coverage": f"{np.mean(data['coverage']):.1f}%"
        }
        summary.append(row)

    df = pd.DataFrame(summary)
    print("\n" + "="*80)
    print("Evaluation Results (Parallel Execution)")
    print("="*80)
    print(df.to_string(index=False))
    print("="*80)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="checkpoints_integrated_random/main_iter300.pt")
    parser.add_argument("--escape-ckpt", default="checkpoints_integrated_random/escape_iter300.pt")
    parser.add_argument("--episodes", type=int, default=50, help="Episodes per scenario")
    parser.add_argument("--max-steps", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--cores", type=int, default=6, help="Number of parallel cores")
    
    args = parser.parse_args()
    evaluate(args)
