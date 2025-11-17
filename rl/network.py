# rl/network.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

"""
벡터 관측용 공유 특징 추출기.
obs_dim -> hidden -> hidden -> feat_dim
- LayerNorm + Tanh로 안정성/수렴성 확보
- Orthogonal init로 초깃값 품질 개선
"""
class Backbone(nn.Module):
    """
     Parameter:
      obs_dim: 특징 추출 관측값 벡터의 크기
      hidden_sizes: 은닉층의 크기 기본 (128, 128)
      feat_dim: 특징 벡터의 크기 128 (64, 256로 할지 실험하면서 결정)
    """
    def __init__(self, obs_dim: int, hidden_sizes=(128, 128), feat_dim=128):
        super().__init__()
        if len(hidden_sizes) == 0:
            raise ValueError("hidden_sizes must contain at least one layer size.")

        # 입력→여러 은닉층(각각 LayerNorm 포함)
        self.hidden_layers = nn.ModuleList()
        in_dim = obs_dim
        for h in hidden_sizes:
            layer = nn.Linear(in_dim, h)
            ln = nn.LayerNorm(h)
            self.hidden_layers.append(nn.ModuleDict({"linear": layer, "ln": ln}))
            in_dim = h

        self.fc3 = nn.Linear(in_dim, feat_dim) # 마지막 은닉 → feature
        self._init_weights()


    def _init_weights(self): # 각 층의 weight와 bias들이 너무 크거나 작은 값이 아니게 초기화 하는 메소드
        # Orthogonal init (ReLU/Tanh 호환), 작은 gain로 폭주 방지
        for layer in self.hidden_layers:
            lin = layer["linear"]
            nn.init.orthogonal_(lin.weight, gain=1.0)
            nn.init.zeros_(lin.bias)
        nn.init.orthogonal_(self.fc3.weight, gain=1.0)
        nn.init.zeros_(self.fc3.bias)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x = obs
        for layer in self.hidden_layers:
            x = layer["linear"](x)
            x = layer["ln"](x)
            x = torch.tanh(x)

        feat = self.fc3(x)          # 마지막은 보통 비선형 없이 feature로 사용
        feat = torch.tanh(feat)      # 원하면 주석 처리 가능
        return feat


"""
 액터 크리틱 클래스
"""
class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes=(128,128), feat_dim=128):
        super().__init__()
        self.backbone = Backbone(obs_dim, hidden_sizes, feat_dim)

        # Actor head → 행동 분포를 출력
        self.actor = nn.Linear(feat_dim, act_dim)

        # Critic head → 상태의 value (스칼라) 출력
        self.critic = nn.Linear(feat_dim, 1)

    def forward(self, obs):
        feat = self.backbone(obs)       # 특징 추출
        logits = self.actor(feat)       # 행동 로짓 (softmax 전 단계)
        value  = self.critic(feat)      # 상태 가치
        return logits, value
