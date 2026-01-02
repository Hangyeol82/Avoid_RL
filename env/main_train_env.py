import numpy as np
from env.dyn_env_one import DynAvoidOneObjEnv
from env.moving_object import MovingObj

class MainTrainingEnv(DynAvoidOneObjEnv):
    """
    [Main Policy 집중 훈련 환경]
    - 일반적인 주행(Coverage)과 회피(Avoidance)를 모두 학습하지만,
    - 'focused_training_prob' 확률로 '회피 시나리오'를 강제로 발생시켜
      데이터 수집 효율을 극대화함.
    - 회피 시나리오 성공 시 에피소드를 조기 종료하여 '왔다갔다(Reward Hacking)' 방지.
    """
    def __init__(self, *args, **kwargs):
        # 집중 훈련 확률 (기본 50%)
        self.focused_training_prob = float(kwargs.pop("focused_training_prob", 0.5))
        super().__init__(*args, **kwargs)
        self.focused_mode = False
        self.success_counter = 0
        self._focused_post_safe_steps = 0
        self._focused_safe_bonus_given = False

    def reset(self, seed=None, options=None):
        # 1. 기본 초기화
        obs, info = super().reset(seed=seed, options=options)
        
        # 2. 확률적으로 회피 시나리오 모드 진입
        self.focused_mode = False
        self.success_counter = 0
        self._focused_post_safe_steps = 0
        self._focused_safe_bonus_given = False
        if self.focused_training_prob > 0.0 and self.rng.random() < self.focused_training_prob:
            self._setup_avoid_scenario()
            self.focused_mode = True
            # 시나리오 셋업 후 관측값 갱신 필요
            return self._obs(), info
            
        return obs, info

    def _setup_avoid_scenario(self):
        """
        [Focused Training] Main Policy의 회피 능력 향상을 위해
        에이전트와 동적 객체를 충돌 코스에 배치함.
        """
        H, W = self.grid.shape
        
        # (1) 에이전트 위치: 벽이 아닌 랜덤 위치
        # 맵 중앙 부근에서 찾는 것이 좋음 (가장자리보다는)
        for _ in range(100):
            cy = self.rng.integers(5, H - 5)
            cx = self.rng.integers(5, W - 5)
            if self.grid[cy, cx] == 0:
                self.agent_rc = np.array([float(cy), float(cx)], dtype=float)
                break
        
        # (2) 동적 객체: 에이전트를 향해 다가오는 위치에 배치
        # 거리 5.0 ~ 8.0 (안전 거리 밖에서 시작하여 접근)
        if not self.dynamic_objs:
            self.dynamic_objs.append(MovingObj(np.array([0,0]), np.array([0,0]), 1.0, "cv", 999))
        
        obj = self.dynamic_objs[0]
        
        for _ in range(50):
            angle = self.rng.uniform(0, 2 * np.pi)
            dist = self.rng.uniform(5.0, 8.0)
            oy = self.agent_rc[0] + dist * np.sin(angle)
            ox = self.agent_rc[1] + dist * np.cos(angle)
            
            ioy, iox = int(oy), int(ox)
            if 0 <= ioy < H and 0 <= iox < W and self.grid[ioy, iox] == 0:
                obj.p = np.array([oy, ox], dtype=float)
                
                # 속도 벡터: 에이전트를 향하도록 설정 (충돌 유도)
                speed = self.rng.uniform(0.5, 1.0)
                # 약간의 노이즈를 섞어 완벽한 정면 충돌만 있는 것은 아니게 함 (-15도 ~ +15도)
                noise = self.rng.uniform(-0.26, 0.26) 
                aim_angle = np.arctan2(self.agent_rc[0] - oy, self.agent_rc[1] - ox) + noise
                obj.v = np.array([speed * np.sin(aim_angle), speed * np.cos(aim_angle)], dtype=float)
                obj.kind = "cv" # 등속 직선 운동
                break
        
        # (3) 웨이포인트 재설정 (현재 위치 근처에서 시작하도록)
        # 가장 가까운 웨이포인트를 찾아서 거기서부터 시작하게 함
        # 이렇게 해야 에이전트가 멍하니 있지 않고 주행을 시도하다가 회피를 하게 됨
        dists = np.linalg.norm(self.waypoints - np.array([self.agent_rc[1], self.agent_rc[0]]), axis=1)
        nearest_idx = np.argmin(dists)
        self.wp_idx = nearest_idx
        self.visited.fill(False)
        # 이미 지나온 곳들은 방문 처리 (단순화)
        if nearest_idx > 0:
            self.visited[:nearest_idx] = True

    def step(self, action):
        # 부모 클래스 step 실행
        obs, reward, done, trunc, info = super().step(action)
        
        # [Focused Mode 전용 종료 조건]
        # 회피 성공 후 바로 끝내지 않고, SAFE 유지가 연속 4스텝 되면 종료
        if self.focused_mode and not done:
            # AVOID가 다시 켜지면 카운터 초기화
            if info.get("mode", "") == "AVOID":
                self._focused_post_safe_steps = 0

            dist_to_obj_cells = self._distance_to_nearest_obj_cells()

            if self.steps > 20 and np.isfinite(dist_to_obj_cells) and dist_to_obj_cells >= self.safe_cells:
                # SAFE 도달 시 보너스는 한 번만 지급
                if not self._focused_safe_bonus_given:
                    reward += 0.3
                    self._focused_safe_bonus_given = True
                self._focused_post_safe_steps += 1
                if self._focused_post_safe_steps >= 4:
                    done = True
                    info["finish_reason"] = "focused_training_success"
            else:
                self._focused_post_safe_steps = 0
        
        return obs, reward, done, trunc, info

    def _distance_to_nearest_obj_cells(self):
        # 부모 클래스에 _distance_to_nearest_obj_m 은 있는데 cells 단위가 없어서 유틸로 추가
        # 혹은 m 단위를 cells로 변환해서 비교
        d_m = self._distance_to_nearest_obj_m()
        if d_m == float("inf"):
            return float("inf")
        return d_m / self.cell_size_m
