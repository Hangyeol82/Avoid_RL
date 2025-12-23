# rl/network.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
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


class HybridBackbone(nn.Module):
    """
    Splits input into (MLP features) + (Map features).
    - MLP part: Processed by FC layers
    - Map part: Processed by CNN layers
    """
    def __init__(self, obs_dim: int, map_size=15, hidden_sizes=(128, 128), feat_dim=256):
        super().__init__()
        
        self.map_size = map_size
        self.map_dim = map_size * map_size
        self.mlp_in_dim = obs_dim - self.map_dim
        
        if self.mlp_in_dim <= 0:
            raise ValueError(f"obs_dim({obs_dim}) must be larger than map_dim({self.map_dim})")

        # --- MLP Part (Sensor Data) ---
        self.mlp_layers = nn.ModuleList()
        in_dim = self.mlp_in_dim
        # Reduce hidden size slightly since we have CNN features too
        mlp_hiddens = [h // 2 for h in hidden_sizes] 
        
        for h in mlp_hiddens:
            layer = nn.Linear(in_dim, h)
            ln = nn.LayerNorm(h)
            self.mlp_layers.append(nn.ModuleDict({"linear": layer, "ln": ln}))
            in_dim = h
        self.mlp_out_dim = in_dim

        # --- CNN Part (Local Map) ---
        # Input: (N, 1, 15, 15)
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1), # -> (16, 15, 15)
            nn.ReLU(),
            nn.MaxPool2d(2), # -> (16, 7, 7)
            
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1), # -> (32, 7, 7)
            nn.ReLU(),
            nn.MaxPool2d(2), # -> (32, 3, 3)
            
            nn.Flatten(), # -> 32 * 3 * 3 = 288
        )
        
        # Calculate CNN output size dynamically
        with torch.no_grad():
            dummy = torch.zeros(1, 1, map_size, map_size)
            cnn_out = self.cnn(dummy)
            self.cnn_out_dim = cnn_out.shape[1]

        # --- Fusion ---
        fusion_in = self.mlp_out_dim + self.cnn_out_dim
        self.fusion = nn.Linear(fusion_in, feat_dim)
        
        self._init_weights()

    def _init_weights(self):
        # MLP init
        for layer in self.mlp_layers:
            lin = layer["linear"]
            nn.init.orthogonal_(lin.weight, gain=np.sqrt(2))
            nn.init.zeros_(lin.bias)
            
        # CNN init
        for m in self.cnn.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
                    
        # Fusion init
        nn.init.orthogonal_(self.fusion.weight, gain=1.0)
        nn.init.zeros_(self.fusion.bias)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # Split observation
        # obs: (N, obs_dim)
        mlp_in = obs[:, :self.mlp_in_dim]
        map_in = obs[:, self.mlp_in_dim:] # (N, 225)
        
        # 1. MLP Forward
        x_mlp = mlp_in
        for layer in self.mlp_layers:
            x_mlp = layer["linear"](x_mlp)
            x_mlp = layer["ln"](x_mlp)
            x_mlp = torch.tanh(x_mlp)
            
        # 2. CNN Forward
        # Reshape map: (N, 225) -> (N, 1, 15, 15)
        x_map = map_in.view(-1, 1, self.map_size, self.map_size)
        x_cnn = self.cnn(x_map)
        
        # 3. Fusion
        combined = torch.cat([x_mlp, x_cnn], dim=1)
        feat = self.fusion(combined)
        feat = torch.tanh(feat)
        
        return feat


"""
 액터 크리틱 클래스
"""
class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes=(128,128), feat_dim=128):
        super().__init__()
        # Check if we should use Hybrid or Standard Backbone
        # Heuristic: if obs_dim is large (> 300) and likely contains map, use Hybrid
        # Or we can pass a flag. For now, let's assume if obs_dim > 300 it's hybrid.
        # Current obs_dim with map is ~100 + 225 = 325.
        
        if obs_dim > 300:
            self.backbone = HybridBackbone(obs_dim, map_size=15, hidden_sizes=hidden_sizes, feat_dim=feat_dim)
        else:
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
