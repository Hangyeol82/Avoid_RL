# rl/ppo.py
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
import torch
import torch.optim as optim
from torch.distributions import Categorical

from rl.network import ActorCritic          # 네가 만든 네트워크
from rl.buffer import RolloutBuffer, BufferConfig


@dataclass
class PPOConfig:  # 하이퍼파라미터, 환경 정의
    obs_dim: int  # 관측 벡터의 크기
    act_dim: int  # 행동 공간의 크기

    # (추가) 모델 구조
    hidden_sizes: Tuple[int, ...] = (128, 128)  # Actor-Critic 공유 백본의 히든 레이어 크기
    feat_dim: int = 128                         # 백본에서 뽑을 특징 차원

    # rollout
    rollout_steps: int = 4096  # rolloutbuffer에 모을 최대 스텝

    # optimization
    lr: float = 3e-4           # Adam 옵티마이저 학습률
    epochs: int = 10           # rollout 데이터로 몇 번 학습할 지
    batch_size: int = 256      # 미니배치 크기

    # PPO 손실 계수
    clip_eps: float = 0.2      # Ratio 클리핑 할 범위
    vf_coef: float = 0.5       # Actor, Critic의 중요도를 설정 (값이 크면 value, 작으면 policy)
    ent_coef: float = 0.05     # 엔트로피(탐험) 강도
    max_grad_norm: float = 0.5 # 역전파의 기울기 상한선 (ppo는 기울기가 튈 수 있어 제한)

    # GAE/감가율
    gamma: float = 0.99        # 할인율
    gae_lambda: float = 0.95   # 0 -> 1-step TD, 1 -> MC (0~1 사이로 조정)

    # 기타
    seed: Optional[int] = 0
    device: str = "cpu"        # "cuda" 가능


class PPOTrainer:
    # Parameter:
    #
    def __init__(self, env, cfg: PPOConfig):
        """
        env: Gymnasium 스타일 환경 (reset, step, action_space.n, observation_space.shape)
        """
        self.env = env
        self.cfg = cfg  # PPOConfig 파라미터 모두 모아둔 거

        # ----- 디바이스 & 시드 -----
        self.device = torch.device(cfg.device)
        if cfg.seed is not None:
            np.random.seed(cfg.seed)
            torch.manual_seed(cfg.seed)

        # ----- 네트워크 -----
        self.model = ActorCritic(
            obs_dim=cfg.obs_dim,
            act_dim=cfg.act_dim,
            hidden_sizes=cfg.hidden_sizes,  # (수정) Config 반영
            feat_dim=cfg.feat_dim,          # (수정) Config 반영
        ).to(self.device)

        # ----- 옵티마이저 -----
        self.optimizer = optim.Adam(self.model.parameters(), lr=cfg.lr)  # 아담으로 최적화 진행

        # ----- 버퍼 -----
        self.buffer = RolloutBuffer(
            BufferConfig(
                obs_dim=cfg.obs_dim,
                max_size=cfg.rollout_steps,
                gamma=cfg.gamma,
                gae_lambda=cfg.gae_lambda,
                device=cfg.device,
                dtype=torch.float32,
            )
        )

        # 로그용
        self.global_step = 0
        self.last_info = {}
        self._last_mode = "FOLLOW_CPP" # 이전 스텝의 모드를 추적하기 위한 변수
        self.mask_retroactive_steps = 4 # AVOID 모드 진입 시 소급 적용할 스텝 수

    def collect_rollout(self):
        """
        - cfg.rollout_steps 만큼 (s, a, r, done, v, logπ) 수집
        - 에피소드가 종료되면 finish_path(last_value=0)
        - 시간제한/중도중단으로 끊기면 bootstrap value로 finish_path(last_value=V(s_T))
        - 다음 업데이트에서 쓸 수 있게 버퍼가 꽉 찬 상태로 반환
        - 반환값: 수집한 스텝 수 (int) → print에서 steps로 사용
        """
        self.model.eval()  # 수집 단계에서는 학습 아님 (드롭아웃/정규화 추론 모드)
        steps_to_collect = self.cfg.rollout_steps  # 얼만큼 모을지

        # 첫 호출 시 초기 obs 준비
        if not hasattr(self, "_curr_obs"):
            obs, _ = self.env.reset()
            self._curr_obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)

        ep_ret, ep_len = 0.0, 0
        collected = 0

        with torch.no_grad():
            while collected < steps_to_collect:
                # 1) 정책/가치 추론
                logits, value = self.model(self._curr_obs.unsqueeze(0))   # (1, obs_dim) -> (1, A), (1, 1)
                value = value.squeeze(0).squeeze(-1)                      # scalar
                dist = Categorical(logits=logits.squeeze(0))              # 이산 행동분포
                action = dist.sample()
                logprob = dist.log_prob(action)

                # 2) 환경 한 스텝
                next_obs, reward, terminated, truncated, info = self.env.step(action.item())
                done = bool(terminated or truncated)
                
                # (수정) info에서 "mode"를 가져와 마스크 생성
                current_mode = info.get("mode", "FOLLOW_CPP")
                mask = (current_mode == "AVOID")

                # (추가) AVOID 모드로 처음 전환되는 시점을 감지
                if current_mode == "AVOID" and self._last_mode == "FOLLOW_CPP":
                    # 버퍼에 저장된 직전 N개 데이터의 마스크를 True로 덮어쓰기
                    self.buffer.apply_mask_retroactively(self.mask_retroactive_steps)

                # 3) 버퍼에 기록
                self.buffer.store(
                    obs=self._curr_obs,
                    action=action,
                    logprob=logprob,
                    reward=float(reward),
                    done=done,
                    value=value,
                    mask=mask,
                )

                # (추가) 현재 모드를 _last_mode에 저장
                self._last_mode = current_mode

                ep_ret += float(reward)
                ep_len += 1
                self.global_step += 1
                collected += 1

                # 4) 경로 종료 처리 (GAE 마감)
                if done:
                    # 완전 종료면 bootstrap 없음 (last_value=0)
                    self.buffer.finish_path(last_value=torch.zeros((), device=self.device))
                    # 리셋
                    next_obs, _ = self.env.reset()
                    # 로그 저장(선택)
                    self.last_info = {"ep_return": ep_ret, "ep_length": ep_len}
                    ep_ret, ep_len = 0.0, 0
                    self._last_mode = "FOLLOW_CPP" # 에피소드 리셋 시 모드 초기화

                # 5) 다음 obs 갱신
                self._curr_obs = torch.as_tensor(next_obs, dtype=torch.float32, device=self.device)

            # 루프가 steps_to_collect에서 끊긴 경우:
            # 에피소드가 진행 중이라 경로가 열려 있으면 bootstrap value로 마감
            logits, v_boot = self.model(self._curr_obs.unsqueeze(0))
            v_boot = v_boot.squeeze(0).squeeze(-1)
            self.buffer.finish_path(last_value=v_boot)

        self.model.train()
        return collected  # ← 정수 스텝 수를 반환 (print에서 steps 사용)

    # PPOTrainer 내부
    def update(self):
        self.model.train()

        clip_eps = self.cfg.clip_eps       # 클리핑 범위
        vf_coef = self.cfg.vf_coef         # critic 학습 비중 조절
        ent_coef = self.cfg.ent_coef       # 엔트로피 탐색성 유지
        max_norm = self.cfg.max_grad_norm  # 기울기 한계치 (기울기 폭주 방지)

        # 로깅 누적용
        log_policy, log_value, log_entropy = 0.0, 0.0, 0.0  # 정책 손실 합, 가치 손실 합, 엔트로피 평균
        log_loss, log_kl, log_clipfrac = 0.0, 0.0, 0.0      # 전체 손실 합, KL, clip된 비율
        n_batches = 0

        for _ in range(self.cfg.epochs):
            for batch in self.buffer.get(batch_size=self.cfg.batch_size, shuffle=True):
                # batch는 딕셔너리
                obs        = batch["obs"]
                actions    = batch["actions"]
                old_logp   = batch["logprobs"]
                returns    = batch["returns"]
                advantages = batch["advantages"]
                old_values = batch["values"].squeeze(-1)

                # === AVOID 학습 마스크 ===
                masks      = batch["masks"]                   # ✅ 이름 복수형으로 변경
                mask_sum = torch.clamp(masks.sum(), min=1.0) # 0으로 나눔 방지 (모든 샘플이 FOLLOW면 0이 될 수 있음)

                # 현재 정책/가치 예측
                logits, values = self.model(obs)
                values = values.squeeze(-1)

                dist   = torch.distributions.Categorical(logits=logits)
                logp   = dist.log_prob(actions)

                # 엔트로피도 마스크 평균으로 (AVOID 구간에서만 탐험성 유지)
                entropy = (dist.entropy() * masks).sum() / mask_sum

                # ratio = π_new / π_old
                ratio = torch.exp(logp - old_logp)

                # 정책 손실 (clipped surrogate) — 마스크 적용
                pg_loss1 = -advantages * ratio * masks
                pg_loss2 = -advantages * torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * masks
                policy_loss = torch.max(pg_loss1, pg_loss2).sum() / mask_sum  # 보수적인 surrogate (마스크 평균)

                # 가치 손실 (value clipping 포함) — 마스크 적용
                v_pred_clipped   = old_values + (values - old_values).clamp(-clip_eps, clip_eps)
                v_loss_unclipped = ((values - returns) ** 2) * masks
                v_loss_clipped   = ((v_pred_clipped - returns) ** 2) * masks
                value_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).sum() / mask_sum

                # 최종 손실
                loss = policy_loss + vf_coef * value_loss - ent_coef * entropy

                # 최적화
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm)
                self.optimizer.step()

                # 로깅 통계 (마스크 평균으로)
                approx_kl = ((old_logp - logp) * masks).sum().detach().item() / mask_sum.item()
                clipfrac  = ((torch.abs(ratio - 1.0) > clip_eps).float() * masks).sum().detach().item() / mask_sum.item()

                log_policy   += policy_loss.detach().item()
                log_value    += value_loss.detach().item()
                log_entropy  += entropy.detach().item()
                log_loss     += loss.detach().item()
                log_kl       += approx_kl
                log_clipfrac += clipfrac
                n_batches    += 1

        # 한 번의 update 평균 로그 반환
        if n_batches > 0:
            logs = dict(
                loss=log_loss / n_batches,
                policy_loss=log_policy / n_batches,
                value_loss=log_value / n_batches,
                entropy=log_entropy / n_batches,
                approx_kl=log_kl / n_batches,
                clipfrac=log_clipfrac / n_batches,  # ← 키 이름 통일
            )
        else:
            logs = {}

        # 다음 rollout을 위해 버퍼 비우기
        self.buffer.clear()
        return logs

    def learn(self, total_steps: int):
        """
        total_steps까지:
        1) rollout 수집
        2) 업데이트
        3) 로그 갱신
        """
        while self.global_step < total_steps:
            # 1. rollout 수집 (환경 돌리기)
            steps = self.collect_rollout()

            # 2. 네트워크 업데이트 (actor + critic 학습)
            update_info = self.update()

            # 3. 로그/스텝 갱신
            self.global_step += steps
            self.last_info = {"steps": steps, **update_info}

        return self.last_info