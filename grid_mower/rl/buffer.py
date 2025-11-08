"""
1.	RolloutBuffer 먼저 작성
2.	PPOTrainer.__init__ 세팅
3.	collect_rollout()
4.	update()
5.	learn() 루프
"""
from dataclasses import dataclass
import torch
import numpy as np
from typing import Optional, Iterator, Dict


"""
 BufferConfig
  환경 정의 + 하이퍼 파라미터

  gamma, gae_lambda, max_size -> 하이퍼 파라미터
  obs_dim, device, dtype -> 환경 정의
"""
@dataclass
class BufferConfig:
    obs_dim: int        # 관측 벡터 크기
    max_size: int       # 저장할 수 있는 최대 step 크기
    gamma: float = 0.99 # 할인율 discout factor
    gae_lambda: float = 0.95 # GAE의 람다 값
    """
     gae_lambda가 의미하는 것 (아직 잘 이해 안됨)
      λ = 0
      -> 1 step TD
      λ = 1
      -> 끝까지 다 합산한 값 리턴
    """
    device: str = "cpu" # 학습 할 때 cpu/cuda 선택
    dtype: torch.dtype = torch.float32 # 텐서 자료형

class RolloutBuffer:
    """
    PPO용 trajectory 저장/전처리 버퍼
    - store: 매 step에서 (obs, action, logprob, reward, done, value) 저장
    - finish_path: GAE Advantage/Return 계산
    - get: 미니배치 iterator 반환
    - clear: 비우기
    """
    def __init__(self, cfg: BufferConfig):
        self.cfg = cfg
        n = cfg.max_size
        d = cfg.obs_dim
        dev = torch.device(cfg.device) # cpu / cuda 어디서 연산할지

        # storage (torch tensor로 관리)
        self.obs = torch.zeros((n, d), dtype=cfg.dtype, device=dev)
        self.actions = torch.zeros(n, dtype=torch.long, device=dev)   # discrete
        self.logprobs = torch.zeros(n, dtype=cfg.dtype, device=dev)
        self.rewards = torch.zeros(n, dtype=cfg.dtype, device=dev)
        self.dones = torch.zeros(n, dtype=torch.bool, device=dev)
        self.values = torch.zeros(n, dtype=cfg.dtype, device=dev)
        self.masks = torch.zeros(n, dtype=torch.bool, device=dev)  # ✅ bool 타입으로 변경

        self.advantages = torch.zeros(n, dtype=cfg.dtype, device=dev) # Q - V 즉 advantage 함수 결과값
        self.returns = torch.zeros(n, dtype=cfg.dtype, device=dev) # critic의 mse (정답 - 현재) ^ 2 에서 정답을 의미함

        self.ptr = 0 # 현재 어디까지 기록했는지 가르키는 포인터
        self.path_start = 0 # 하나의 trajectroy가 어디서 시작했는지 가르킴
        self.full = False # 버퍼가 다 찼는지 여부

    # ---------- write ----------
    # 한 스텝씩 경험을 기록하는 함수
    @torch.no_grad()
    def store(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        logprob: torch.Tensor,
        reward: float,
        done: bool,
        value: torch.Tensor,
        mask: torch.Tensor
    ):
        """
        obs:  (obs_dim,)  or (1, obs_dim)
        action: scalar tensor (discrete) 또는 long
        logprob: scalar tensor
        value: scalar tensor (critic output)
        """
        # 예외 처리
        if self.ptr >= self.cfg.max_size:
            raise RuntimeError("RolloutBuffer overflow: increase max_size or call finish_path/get earlier.")

        # shape 정리
        obs = obs.detach().flatten() # 순수한 값만 빼고 크기 맞춰 주기
        #예외 처리
        if obs.numel() != self.cfg.obs_dim:
            raise ValueError(f"obs dim mismatch: {obs.shape} vs {self.cfg.obs_dim}")

        self.obs[self.ptr] = obs
        self.actions[self.ptr] = action.long().view(())
        self.logprobs[self.ptr] = logprob.view(())
        self.rewards[self.ptr] = torch.as_tensor(reward, dtype=self.cfg.dtype, device=self.obs.device).view(())
        self.dones[self.ptr] = bool(done)
        self.values[self.ptr] = value.view(())
        self.masks[self.ptr] = bool(mask)  # ✅ bool 타입으로 저장

        self.ptr += 1
        if self.ptr == self.cfg.max_size:
            self.full = True

    # ---------- post-process (GAE) ----------
    """
     Function: 
     Parameter: 
      last_value
       -> trajectory가 끝날 때 마지막 상태의 value (Critic 예측값)
       -> 에피소드가 끝난 경우 (done=True) last_value = 0
    """
    @torch.no_grad()
    def finish_path(self, last_value: Optional[torch.Tensor] = None):
        """
        현재 path_start ~ ptr-1 구간에 대해 GAE/Return 계산
        - last_value: trajectory가 시간 제한 등으로 끊긴 경우 bootstrap value
                      에피소드가 완전히 끝난(done=True) 상태라면 0 또는 None
        """
        gamma = self.cfg.gamma
        lam = self.cfg.gae_lambda

        if last_value is None:
            last_val = torch.zeros((), dtype=self.cfg.dtype, device=self.obs.device)
        else:
            last_val = last_value.view(())

        # indices for current trajectory segment
        end = self.ptr
        rewards = self.rewards[self.path_start:end] # 보상
        values = self.values[self.path_start:end]  # critic이 예측한 V(s)
        dones = self.dones[self.path_start:end]  # 종료 여부

        # δ_t = r_t + γ V_{t+1} - V_t (TD 오차 구하기)
        # V_{t+1}: 다음 값, 마지막은 last_val 사용
        next_values = torch.zeros_like(values)
        if len(values) > 0:
            next_values[:-1] = values[1:]
            next_values[-1] = last_val

        deltas = rewards + gamma * next_values * (~dones).float() - values

        # GAE: A_t = δ_t + (γλ) δ_{t+1} + ...
        # GAE 는 Advantage 추정하는 기술
        adv = torch.zeros_like(deltas)
        gae = torch.zeros((), dtype=self.cfg.dtype, device=self.obs.device)

        for t in reversed(range(len(deltas))):
            mask = (0.0 if dones[t] else 1.0)
            gae = deltas[t] + gamma * lam * mask * gae
            adv[t] = gae

        self.advantages[self.path_start:end] = adv  # actor 학습에 사용 (Ratio * advantage = 손실함수)
        self.returns[self.path_start:end] = adv + values # critic 학습에 사용 (정답 부분에 해당)

        # 다음 path 시작 인덱스 갱신
        self.path_start = self.ptr

    # ---------- read ----------
    """
     Function: 배치단위로 학습 데이터를 묶어서 리턴해주는 기능
     Parameter:
       batch_size: 미니배치 크기, 따로 지정 안하면 전부 학습
       shuffle: 배치 꺼낼 때 섞을지 말지
    """
    def get(self, batch_size: Optional[int] = None, shuffle: bool = True) -> Iterator[Dict[str, torch.Tensor]]:
        """
        버퍼 전체(0..ptr-1)에서 미니배치 iterator 반환
        - advantages는 표준화해서 제공
        """
        n = self.ptr
        if n == 0:
            return iter(())

        # advantage 정규화 (학습 안정화)
        adv = self.advantages[:n]
        adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)
        self.advantages[:n] = adv

        idx = np.arange(n) # 인덱스 섞기
        if shuffle:
            np.random.shuffle(idx)

        if batch_size is None or batch_size >= n:
            batch = {
                "obs":         self.obs[:n],
                "actions":     self.actions[:n],
                "logprobs":    self.logprobs[:n],
                "returns":     self.returns[:n],
                "advantages":  self.advantages[:n],
                "values":      self.values[:n],
                "masks":       self.masks[:n].float(),  # ✅ float() 타입으로 변환
            }
            yield batch
            return

        # ✅ 미니배치
        for start in range(0, n, batch_size):
            j = idx[start:start+batch_size]
            yield {
                "obs":         self.obs[j],
                "actions":     self.actions[j],
                "logprobs":    self.logprobs[j],
                "returns":     self.returns[j],
                "advantages":  self.advantages[j],
                "values":      self.values[j],
                "masks":       self.masks[j].float(),   # ✅ float() 타입으로 변환
            }
    
    def apply_mask_retroactively(self, num_steps: int):
        """
        현재 경로 내에서 마지막 `num_steps` 만큼의 데이터에 마스크를 적용합니다.
        """
        if self.ptr == 0:
            return
        
        # 에피소드 경계(path_start)를 넘지 않도록 시작 인덱스 계산
        start_idx = max(self.path_start, self.ptr - num_steps)
        end_idx = self.ptr
        
        self.masks[start_idx:end_idx] = True

    # 버퍼 초기화 -> 새로운 Rollout를 저장하게 하는 함수
    def clear(self):
        self.ptr = 0
        self.path_start = 0
        self.advantages.zero_()
        self.returns.zero_()
        self.full = False
        self.masks.zero_()   # ✅ 마스크도 초기화

