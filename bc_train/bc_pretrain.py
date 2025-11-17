# rl/bc_pretrain.py
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from env.env import StaticFieldEnv, FieldConfig
from planning.cpp import CoveragePlanner, HeuristicType
from rl.network import ActorCritic


# --------------------- CPP 유틸 ---------------------
def make_cpp_grid_from_field(occ_or_inflated: np.ndarray, start_rc=(0, 0)) -> np.ndarray:
    """점유맵 => CPP용 격자(0/1/2)로 변환"""
    g = occ_or_inflated.copy().astype(int)
    g[g != 0] = 1
    r0, c0 = start_rc
    if g[r0, c0] == 1:
        free = np.argwhere(g == 0)
        if len(free) == 0:
            raise RuntimeError("No free cell for start")
        r0, c0 = free[0]
    g[r0, c0] = 2
    return g


def compute_waypoints(grid: np.ndarray) -> np.ndarray:
    """CoveragePlanner로 전경로 웨이포인트(x=col,y=row) 생성"""
    cp = CoveragePlanner(grid)
    cp.start(
        initial_orientation=0,
        cp_heuristic=HeuristicType.VERTICAL,
        a_star_heuristic=HeuristicType.MANHATTAN,
    )
    cp.compute()
    _, _, _, traj, _ = cp.result()
    wps = np.array([[t[2], t[1]] for t in traj], dtype=np.float32)  # (N,2) x,y
    return wps


# --------------------- 라벨/관측 생성 ---------------------
MOVE2ACTION = {
    (-1, 0): 0,  # 상
    (0, -1): 1,  # 좌
    (1, 0): 2,   # 하
    (0, 1): 3,   # 우
}


def build_bc_dataset(grid: np.ndarray, waypoints_xy: np.ndarray):
    """
    CPP 경로를 그대로 따라가며 (obs, action) 샘플 생성
    obs = [dx_wp, dy_wp, obj_dx=0, obj_dy=0, obj_speed=0, obj_dist=0]
    action = 셀 간 이동방향 (0/1/2/3), 마지막은 정지(4)
    """
    samples_obs = []
    samples_act = []

    # 시작 위치 = 첫 웨이포인트의 (y,x) 셀
    pos = np.array([waypoints_xy[0][1], waypoints_xy[0][0]], dtype=int)  # r,c

    for i in range(len(waypoints_xy) - 1):
        curr_xy = waypoints_xy[i]
        next_xy = waypoints_xy[i + 1]

        # CPP는 인접 셀로 움직임. (dr,dc)
        dr = int(round(next_xy[1] - curr_xy[1]))
        dc = int(round(next_xy[0] - curr_xy[0]))
        dr = np.clip(dr, -1, 1)
        dc = np.clip(dc, -1, 1)

        if (dr, dc) == (0, 0):
            # 같은 셀 반복이면 스킵
            continue

        if (dr, dc) not in MOVE2ACTION:
            # 대각선 등 예외는 Manhattan 보정: row 우선 이동 후 col 이동으로 쪼개기
            steps = []
            if dr != 0:
                steps.append((np.sign(dr), 0))
            if dc != 0:
                steps.append((0, np.sign(dc)))
        else:
            steps = [(dr, dc)]

        # 스텝마다 샘플 추가
        for sdr, sdc in steps:
            # 관측: 다음 웨이포인트까지 상대벡터 (x,y) 기준
            wp_xy = next_xy
            dx_wp = float(wp_xy[0] - pos[1])
            dy_wp = float(wp_xy[1] - pos[0])

            obs = np.array([dx_wp, dy_wp, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
            act = MOVE2ACTION[(sdr, sdc)]

            samples_obs.append(obs)
            samples_act.append(act)

            # 위치 업데이트
            pos = pos + np.array([sdr, sdc], dtype=int)

    # 경로 마지막엔 정지 샘플 1개 추가(선택)
    last_wp = waypoints_xy[-1]
    dx_last = float(last_wp[0] - pos[1])
    dy_last = float(last_wp[1] - pos[0])
    samples_obs.append(np.array([dx_last, dy_last, 0, 0, 0, 0], dtype=np.float32))
    samples_act.append(4)  # 정지

    obs_arr = np.stack(samples_obs, axis=0)          # (N,6)
    act_arr = np.array(samples_act, dtype=np.int64)  # (N,)
    return obs_arr, act_arr


# --------------------- 학습 루프 ---------------------
def train_bc(
    obs_dim=6,
    act_dim=5,
    hidden_sizes=(128, 128),
    feat_dim=128,
    lr=3e-4,
    batch_size=256,
    epochs=20,
    device="cpu",
    seed=0,
    save_path="bc_actorcritic.pth",
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = torch.device(device)

    # 1) 필드 생성 & CPP 웨이포인트
    fcfg = FieldConfig(width_m=40.0, height_m=30.0, cell_m=0.1, seed=seed)
    artifacts = StaticFieldEnv(fcfg).reset()
    grid = make_cpp_grid_from_field(artifacts.inflated, start_rc=(0, 0))
    wps = compute_waypoints(grid)

    # 2) (obs, action) 데이터셋 구성
    obs_np, act_np = build_bc_dataset(grid, wps)
    N = len(obs_np)
    print(f"[BC] dataset size: {N}")

    # 저장: 이후 PPO에서 재사용
    np.save("map_grid.npy", grid)
    np.save("waypoints.npy", wps)
    print("[BC] saved map_grid.npy & waypoints.npy")

    # 3) 네트워크/옵티마이저
    model = ActorCritic(obs_dim=obs_dim, act_dim=act_dim,
                        hidden_sizes=hidden_sizes, feat_dim=feat_dim).to(dev)
    optimizer = optim.Adam(model.actor.parameters(), lr=lr)  # actor만 업데이트(BC)

    criterion = nn.CrossEntropyLoss()

    # 4) 학습
    obs_tensor = torch.from_numpy(obs_np).to(dev)
    act_tensor = torch.from_numpy(act_np).to(dev)

    num_batches = int(np.ceil(N / batch_size))
    for ep in range(1, epochs + 1):
        # 셔플
        idx = np.random.permutation(N)
        obs_shuf = obs_tensor[idx]
        act_shuf = act_tensor[idx]

        epoch_loss = 0.0
        epoch_acc = 0.0

        model.train()
        for b in range(num_batches):
            s = b * batch_size
            e = min(N, s + batch_size)
            ob = obs_shuf[s:e]
            y  = act_shuf[s:e]

            logits, _ = model(ob)       # (B, act_dim)
            loss = criterion(logits, y) # CE

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                preds = logits.argmax(dim=-1)
                acc = (preds == y).float().mean().item()

            epoch_loss += loss.item() * (e - s)
            epoch_acc  += acc * (e - s)

        epoch_loss /= N
        epoch_acc  /= N
        print(f"[BC] epoch {ep:02d}/{epochs}  loss={epoch_loss:.4f}  acc={epoch_acc:.3f}")

    # 5) 저장(ActorCritic 전체 저장: 이후 PPO warm-start 가능)
    torch.save(model.state_dict(), save_path)
    print(f"[BC] saved weights -> {save_path}")


# --------------------- main ---------------------
if __name__ == "__main__":
    os.makedirs(".", exist_ok=True)
    train_bc(
        obs_dim=6,        # DynAvoidOneObjEnv 관측과 일치
        act_dim=5,        # 상/좌/하/우/정지
        hidden_sizes=(128, 128),
        feat_dim=128,
        lr=3e-4,
        batch_size=256,
        epochs=20,
        device="cpu",     # "cuda" 가능
        seed=0,
        save_path="bc_actorcritic.pth",
    )