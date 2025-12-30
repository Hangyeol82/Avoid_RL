import numpy as np
from env.dyn_env_one import DynAvoidOneObjEnv
from env.moving_object import MovingObj

class EscapeTrainingEnv(DynAvoidOneObjEnv):
    """
    [Escape Policy 집중 훈련 환경]
    - Reset 시: 맵의 랜덤한 위치에 '인공적인 위험 구역'을 생성하고 에이전트를 그 안에 배치함.
    - 목표: 위험 구역을 탈출하여 'escape_active' 상태를 해제하는 것.
    - 종료 조건: 탈출 성공(Done=True, Reward++), 충돌(Done=True, Reward--), 시간 초과.
    """
    def __init__(self, *args, **kwargs):
        # Escape 모드 강제 활성화
        kwargs["use_escape_subpolicy"] = True
        super().__init__(*args, **kwargs)
        
    def reset(self, seed=None, options=None):
        # 1. 기본 초기화 (맵 로드, 동적 객체 스폰 등)
        obs, info = super().reset(seed=seed, options=options)
        
        # 2. [납치 시나리오] 위험 구역 생성 및 에이전트 배치
        self._setup_escape_scenario()
        
        # 3. 관측 업데이트 (위험 구역 반영)
        return self._obs(), info

    def _setup_escape_scenario(self):
        H, W = self.grid.shape
        
        # (1) 위험 구역을 생성할 중심점 찾기 (빈 공간)
        cy, cx = 0, 0
        found = False
        for _ in range(100):
            cy = self.rng.integers(5, H - 5)
            cx = self.rng.integers(5, W - 5)
            if self.grid[cy, cx] == 0:
                found = True
                break
        
        if not found:
            cy, cx = H // 2, W // 2

        # (2) 동적 객체 확보 및 배치
        if not self.dynamic_objs:
            from env.moving_object import MovingObj
            self.dynamic_objs.append(MovingObj(np.array([0,0]), np.array([0,0]), 1.0, "cv", 123))
        
        target_obj = self.dynamic_objs[0]
        
        # (3) 인공 위험 구역 생성 (객체 ID와 연동)
        # 사용자의 요청대로 위험 구역 크기를 더 키움 (7.0 ~ 11.0)
        radius = self.rng.uniform(7.0, 11.0)
        danger_pts = []
        num_pts = 200  # 영역이 매우 커졌으므로 포인트 개수 대폭 증가 (80 -> 200)
        for _ in range(num_pts):
            r = self.rng.uniform(0, radius)
            th = self.rng.uniform(0, 2 * np.pi)
            py = cy + r * np.sin(th)
            px = cx + r * np.cos(th)
            if 0 <= py < H and 0 <= px < W:
                danger_pts.append((py, px))
        
        self.danger_regions[id(target_obj)] = danger_pts
        self._rebuild_danger_map()
        
        # (4) 에이전트를 위험 구역 내부(중심)로 이동
        self.agent_rc = np.array([float(cy), float(cx)], dtype=float)
        
        # (5) 동적 객체 배치
        # 사용자의 요청대로 에이전트와 3칸(3.0) 떨어진 위치에 랜덤하게 배치
        offset_angle = self.rng.uniform(0, 2 * np.pi)
        offset_dist = 3.0
        oy = cy + offset_dist * np.sin(offset_angle)
        ox = cx + offset_dist * np.cos(offset_angle)
        
        target_obj.p = np.array([oy, ox], dtype=float)
        
        # 객체가 제자리에서 조금씩 움직이게 하여(배회) 계속 위협을 줌
        target_obj.v = np.array([0.0, 0.0], dtype=float)
        target_obj.kind = "patrol" # 제자리 배회 모드 등으로 변경 가능

        # (6) Escape 모드 강제 활성화
        self.escape_active = True
        self._escape_release_counter = 0

    def _cleanup_danger_regions(self):
        # [중요] 훈련 도중 위험 구역이 사라지면 안 됨
        pass 

    def step(self, action):
        # 부모 클래스의 step 실행 (보상 계산 등은 그대로 활용)
        obs, reward, done, trunc, info = super().step(action)
        
        # [종료 조건 추가]
        # 원래 환경에서는 escape가 끝나면 다시 FOLLOW 모드로 가지만,
        # 여기서는 '탈출 성공'이 곧 에피소드 클리어임.
        if not self.escape_active and not done:
            done = True
            reward += 2.0  # 추가 성공 보상 (기존 1.3 + 2.0 = 3.3)
            info["finish_reason"] = "escape_training_success"
            
        return obs, reward, done, trunc, info

    def _stamp_danger_pts(self, pts):
        """
        [최적화] 부모 클래스의 _stamp_danger_pts는 점 하나당 5x5=25번의 _stamp_disc를 호출하여
        매우 느림 (특히 점이 200개일 때). 이를 단순화하여 속도를 개선함.
        """
        if self.danger_zone_map is None or not pts:
            return
        
        # 1. Polyline 연결 (기존 유지 - 선형 보간)
        self.danger_zone_map.stamp_polyline(pts, radius_cells=1.2, val=0.9)
        
        # 2. 점 찍기 (단순화: 루프 제거)
        # 기존에는 거리별로 3단계 Gradient를 주었으나, 여기서는 단일 원으로 처리
        for y, x in pts:
            self.danger_zone_map._stamp_disc(
                self.danger_zone_map.soft,
                y, x,
                r_cells=1.8,  # 충분히 큰 반경
                val=0.95      # 높은 위험도
            )
