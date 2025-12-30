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
        self._cached_trap = None  # [최적화] 위험 구역 맵 캐시

    def reset(self, seed=None, options=None):
        self._cached_trap = None  # 리셋 시 캐시 초기화
        # 1. 기본 초기화 (맵 로드, 동적 객체 스폰 등)
        obs, info = super().reset(seed=seed, options=options)
        
        # 2. [납치 시나리오] 위험 구역 생성 및 에이전트 배치
        self._setup_escape_scenario()
        
        # 3. 관측 업데이트 (위험 구역 반영)
        return self._obs(), info

    def _rebuild_danger_map(self):
        """
        [최적화] 매 스텝 200개의 점을 다시 그리는 것은 낭비이므로,
        첫 번째 프레임에서 그린 결과를 캐싱해두고 재사용함.
        """
        # 캐시가 있으면 그것만 복구하고 끝 (매우 빠름)
        if self._cached_trap is not None:
            np.copyto(self.danger_zone_map.soft, self._cached_trap)
            return

        # 캐시가 없으면(첫 실행) 부모 로직으로 그림
        super()._rebuild_danger_map()
        
        # 그린 결과를 캐시에 저장
        if self.danger_zone_map is not None:
            self._cached_trap = self.danger_zone_map.soft.copy()

    def _setup_escape_scenario(self):
        self._cached_trap = None  # 리셋 시 캐시 초기화
        H, W = self.grid.shape
        
        # (1) 에이전트와 동적 객체가 모두 안전한(벽이 아닌) 위치 찾기
        cy, cx = H // 2, W // 2
        oy, ox = cy + 3.0, cx
        
        found_valid_setup = False
        
        # 최대 100번 시도하여 적절한 위치 쌍을 찾음
        for _ in range(100):
            # 1. 에이전트 위치 (위험 구역 중심) 랜덤 선정
            t_cy = self.rng.integers(5, H - 5)
            t_cx = self.rng.integers(5, W - 5)
            
            if self.grid[t_cy, t_cx] == 1: # 벽이면 패스
                continue
                
            # 2. 동적 객체 위치 선정 (에이전트 주변 3.0 거리)
            # 에이전트 위치가 잡히면, 그 주변에서 벽이 아닌 곳을 찾음 (최대 20번 시도)
            valid_obj = False
            t_oy, t_ox = 0.0, 0.0
            
            for _ in range(20):
                angle = self.rng.uniform(0, 2 * np.pi)
                dist = 3.0
                ty = t_cy + dist * np.sin(angle)
                tx = t_cx + dist * np.cos(angle)
                
                ity, itx = int(ty), int(tx)
                if 0 <= ity < H and 0 <= itx < W:
                    if self.grid[ity, itx] == 0: # 벽이 아니면 성공
                        t_oy, t_ox = ty, tx
                        valid_obj = True
                        break
            
            if valid_obj:
                cy, cx = t_cy, t_cx
                oy, ox = t_oy, t_ox
                found_valid_setup = True
                break

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
        
        # (5) 동적 객체 배치 (이미 위에서 계산된 안전한 위치 사용)
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
        
        # [추가 패널티] 위험 구역 내부에 있을 때 지속적인 패널티 부여
        # 부모 클래스에서는 '동적 객체와의 거리'만으로 패널티를 주지만,
        # 여기서는 '인공 위험 구역(Trap)'에 갇힌 상황이므로, Trap 위에 있는 것 자체로 패널티를 줘야 함.
        if self.danger_zone_map is not None and getattr(self.danger_zone_map, "soft", None) is not None:
            soft = self.danger_zone_map.soft
            ry, rx = int(self.agent_rc[0]), int(self.agent_rc[1])
            if 0 <= ry < self.H and 0 <= rx < self.W:
                danger_val = soft[ry, rx]
                if danger_val > 0.1:
                    # 위험도에 비례한 패널티 (최대 -0.5)
                    # 가만히 있으면 계속 깎이므로 밖으로 나가야 함.
                    reward -= 0.5 * danger_val

        # [종료 조건 추가]
        # 원래 환경에서는 escape가 끝나면 다시 FOLLOW 모드로 가지만,
        # 여기서는 '탈출 성공'이 곧 에피소드 클리어임.
        if not self.escape_active and not done:
            done = True
            reward += 1.3  # 탈출 성공 보상 (충돌 패널티 -2.0보다 작게 설정하여 안전 우선)
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
