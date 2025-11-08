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
        h1, h2 = hidden_sizes

        """
         Function: 입력 크기와 출력 크기를 정하고 정해진 크기에 맞는 층을 생성
         Interface: nn.Linaer(in_features, out_features, bias=True)
         Parameter: 
           in_features 입력 크기
           out_features 출력 크기
           bias 편향 벡터 포함 여부
        """
        self.fc1 = nn.Linear(obs_dim, h1) # h1 은닉층에 관측값 입력
        self.ln1 = nn.LayerNorm(h1) # 정규화
        """
         정규화의 역할
          은닉 벡터 h1의 평균과 분산을 계산하고 평균 0, 분산 1로 스케일링  
         왜 정규화를 사용하는가?
          안정된 학습: 입력 분포가 다름 (환경이 계속 변함) -> 분포를 안정시킴
          기울기 폭주 예방: 은닉층이 값이 커지거나 작아지는걸 방지
          빠른 수렴: 크기가 안정적 -> 최적화가 잘됨

          결론: 어떤 에피소드에서는 100, 50, -200이 나오고 어던 에피소드는 0.1, 3, -0.5 가 나올수 있음
               이 값을 그대로 사용하면 학습이 불안정해기 때문에 정규화 진행
        """
        self.fc2 = nn.Linear(h1, h2) # h1 값을 h2 은닉층에 입력
        self.ln2 = nn.LayerNorm(h2) # 정규화

        self.fc3 = nn.Linear(h2, feat_dim) # h2를 feat_dim으로 압축

        self._init_weights()


    def _init_weights(self): # 각 층의 weight와 bias들이 너무 크거나 작은 값이 아니게 초기화 하는 메소드
        # Orthogonal init (ReLU/Tanh 호환), 작은 gain로 폭주 방지
        for m in [self.fc1, self.fc2, self.fc3]:
            nn.init.orthogonal_(m.weight, gain=1.0)
            nn.init.zeros_(m.bias)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x = self.fc1(obs)
        x = self.ln1(x)
        x = torch.tanh(x)

        x = self.fc2(x)
        x = self.ln2(x)
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