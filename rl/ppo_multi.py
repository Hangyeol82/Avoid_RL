from dataclasses import dataclass
from typing import Optional, Tuple, List
import numpy as np
import torch
import torch.optim as optim
from torch.distributions import Categorical

from rl.network import ActorCritic
from rl.buffer import RolloutBuffer, BufferConfig
from rl.ppo import PPOConfig  # Re-use config

class PPOTrainerMulti:
    def __init__(self, env, cfg: PPOConfig):
        """
        env: SubprocVecEnv (must have num_envs, step, reset)
        """
        self.env = env
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        
        if cfg.seed is not None:
            np.random.seed(cfg.seed)
            torch.manual_seed(cfg.seed)

        # Network
        self.model = ActorCritic(
            obs_dim=cfg.obs_dim,
            act_dim=cfg.act_dim,
            hidden_sizes=cfg.hidden_sizes,
            feat_dim=cfg.feat_dim,
        ).to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=cfg.lr)

        # Multiple buffers for multiple environments
        self.num_envs = env.num_envs
        # Distribute total rollout steps across environments
        self.steps_per_env = cfg.rollout_steps // self.num_envs
        
        # [Fix] Allocate full rollout_steps capacity to each buffer.
        # This allows "fast" environments (where escape triggers often) to collect more data
        # and compensate for "slow" environments, preventing the collector from waiting.
        self.buffer_capacity = cfg.rollout_steps

        self.buffers = [
            RolloutBuffer(
                BufferConfig(
                    obs_dim=cfg.obs_dim,
                    max_size=self.buffer_capacity, 
                    gamma=cfg.gamma,
                    gae_lambda=cfg.gae_lambda,
                    device=cfg.device,
                    dtype=torch.float32,
                )
            ) for _ in range(self.num_envs)
        ]

        self.global_step = 0
        self.last_info = {}
        
        # Track mode for each environment
        self._last_modes = ["FOLLOW_CPP"] * self.num_envs
        self.mask_retroactive_steps = 4

        # Initialize observations
        # SubprocVecEnv reset returns stacked observations
        obs = self.env.reset()
        self._curr_obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)

    def collect_rollout(self):
        self.model.eval()
        collected_per_env = 0
        
        # We collect until each buffer has steps_per_env samples
        # Since all envs step together, we just loop steps_per_env times
        
        with torch.no_grad():
            while collected_per_env < self.steps_per_env:
                # 1) Inference (Batched)
                # _curr_obs is (num_envs, obs_dim)
                logits, values = self.model(self._curr_obs) # (N, A), (N, 1)
                values = values.squeeze(-1) # (N,)
                
                dist = Categorical(logits=logits)
                actions = dist.sample() # (N,)
                logprobs = dist.log_prob(actions) # (N,)

                # 2) Step Environment (Batched)
                # actions needs to be numpy array for SubprocVecEnv
                next_obs, rewards, dones, infos = self.env.step(actions.cpu().numpy())
                
                # 3) Store in buffers
                for i in range(self.num_envs):
                    current_mode = infos[i].get("mode", "FOLLOW_CPP")
                    # [Fix] ESCAPE 모드도 학습 데이터에 포함 (Escape 전용 훈련을 위해)
                    mask = (current_mode == "AVOID" or current_mode == "ESCAPE")
                    
                    # Retroactive masking logic
                    if (current_mode == "AVOID" or current_mode == "ESCAPE") and self._last_modes[i] == "FOLLOW_CPP":
                        self.buffers[i].apply_mask_retroactively(self.mask_retroactive_steps)
                    
                    self.buffers[i].store(
                        obs=self._curr_obs[i],
                        action=actions[i],
                        logprob=logprobs[i],
                        reward=float(rewards[i]),
                        done=dones[i],
                        value=values[i],
                        mask=mask
                    )
                    
                    self._last_modes[i] = current_mode
                    
                    # Handle episode completion
                    if dones[i]:
                        # Finish path with 0 value
                        self.buffers[i].finish_path(last_value=torch.zeros((), device=self.device))
                        # Reset mode tracking
                        self._last_modes[i] = "FOLLOW_CPP"
                        # Note: next_obs[i] is already the reset observation from SubprocVecEnv
                
                self._curr_obs = torch.as_tensor(next_obs, dtype=torch.float32, device=self.device)
                collected_per_env += 1
                self.global_step += self.num_envs

            # End of rollout: bootstrap for incomplete paths
            logits, v_boots = self.model(self._curr_obs)
            v_boots = v_boots.squeeze(-1) # (N,)
            
            for i in range(self.num_envs):
                self.buffers[i].finish_path(last_value=v_boots[i])

        self.model.train()
        return collected_per_env * self.num_envs

    def update(self):
        self.model.train()
        
        # Aggregate data from all buffers
        all_obs = []
        all_actions = []
        all_logprobs = []
        all_returns = []
        all_advantages = []
        all_values = []
        all_masks = []
        
        for buf in self.buffers:
            n = buf.ptr
            if n == 0: continue
            
            # Calculate advantages for this buffer (finish_path was called)
            # But we need to normalize globally, so we just take raw advantages first?
            # RolloutBuffer.get() normalizes internally. 
            # Let's access raw tensors to do global normalization.
            
            all_obs.append(buf.obs[:n])
            all_actions.append(buf.actions[:n])
            all_logprobs.append(buf.logprobs[:n])
            all_returns.append(buf.returns[:n])
            all_advantages.append(buf.advantages[:n])
            all_values.append(buf.values[:n])
            all_masks.append(buf.masks[:n].float())
            
        if not all_obs:
            return {}

        # Concatenate
        obs = torch.cat(all_obs)
        actions = torch.cat(all_actions)
        logprobs = torch.cat(all_logprobs)
        returns = torch.cat(all_returns)
        advantages = torch.cat(all_advantages)
        values = torch.cat(all_values)
        masks = torch.cat(all_masks)
        
        # Global Normalization of Advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Dataset size
        dataset_size = obs.shape[0]
        indices = np.arange(dataset_size)
        
        clip_eps = self.cfg.clip_eps
        vf_coef = self.cfg.vf_coef
        ent_coef = self.cfg.ent_coef
        max_norm = self.cfg.max_grad_norm
        batch_size = self.cfg.batch_size
        
        log_policy, log_value, log_entropy = 0.0, 0.0, 0.0
        log_loss, log_kl, log_clipfrac = 0.0, 0.0, 0.0
        n_batches = 0
        
        for _ in range(self.cfg.epochs):
            np.random.shuffle(indices)
            
            for start in range(0, dataset_size, batch_size):
                end = start + batch_size
                idx = indices[start:end]
                
                b_obs = obs[idx]
                b_actions = actions[idx]
                b_logprobs = logprobs[idx]
                b_returns = returns[idx]
                b_advantages = advantages[idx]
                b_values = values[idx]
                b_masks = masks[idx]
                
                mask_sum = torch.clamp(b_masks.sum(), min=1.0)
                
                # Forward
                logits, curr_values = self.model(b_obs)
                curr_values = curr_values.squeeze(-1)
                dist = Categorical(logits=logits)
                curr_logprobs = dist.log_prob(b_actions)
                entropy = (dist.entropy() * b_masks).sum() / mask_sum
                
                ratio = torch.exp(curr_logprobs - b_logprobs)
                
                # Policy Loss
                pg_loss1 = -b_advantages * ratio * b_masks
                pg_loss2 = -b_advantages * torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * b_masks
                policy_loss = torch.max(pg_loss1, pg_loss2).sum() / mask_sum
                
                # Value Loss
                v_pred_clipped = b_values + (curr_values - b_values).clamp(-clip_eps, clip_eps)
                v_loss_unclipped = ((curr_values - b_returns) ** 2) * b_masks
                v_loss_clipped = ((v_pred_clipped - b_returns) ** 2) * b_masks
                value_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).sum() / mask_sum
                
                loss = policy_loss + vf_coef * value_loss - ent_coef * entropy
                
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm)
                self.optimizer.step()
                
                # Logging
                with torch.no_grad():
                    approx_kl = ((b_logprobs - curr_logprobs) * b_masks).sum().item() / mask_sum.item()
                    clipfrac = ((torch.abs(ratio - 1.0) > clip_eps).float() * b_masks).sum().item() / mask_sum.item()
                
                log_policy += policy_loss.item()
                log_value += value_loss.item()
                log_entropy += entropy.item()
                log_loss += loss.item()
                log_kl += approx_kl
                log_clipfrac += clipfrac
                n_batches += 1
                
        # Clear all buffers
        for buf in self.buffers:
            buf.clear()
            
        if n_batches > 0:
            return dict(
                loss=log_loss / n_batches,
                policy_loss=log_policy / n_batches,
                value_loss=log_value / n_batches,
                entropy=log_entropy / n_batches,
                approx_kl=log_kl / n_batches,
                clipfrac=log_clipfrac / n_batches,
            )
        else:
            return {}

    def learn(self, total_steps: int):
        while self.global_step < total_steps:
            steps = self.collect_rollout()
            update_info = self.update()
            self.last_info = {"steps": steps, **update_info}
            
            # Optional: Print progress
            print(f"Step: {self.global_step}, Return: {self.last_info.get('ep_return', 'N/A')}") # ep_return is tricky in vec env
            
        return self.last_info
