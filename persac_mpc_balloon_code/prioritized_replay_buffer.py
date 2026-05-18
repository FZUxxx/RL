# prioritized_replay_buffer.py
# prioritized_replay_buffer.py
import numpy as np
import torch
import random
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.type_aliases import ReplayBufferSamples
from typing import Union, Dict, List, Tuple, Any
from gymnasium import spaces
class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.write = 0

    def update(self, idx, p):
        idx += self.capacity - 1
        delta = p - self.tree[idx]
        self.tree[idx] = p
        while idx > 0:
            idx = (idx - 1) // 2
            self.tree[idx] += delta

    def get_leaf(self, s):
        idx = 0
        while idx < self.capacity - 1:
            left = 2 * idx + 1
            right = left + 1
            if s <= self.tree[left]:
                idx = left
            else:
                s -= self.tree[left]
                idx = right
        data_idx = idx - self.capacity + 1
        return data_idx, self.tree[idx]

    @property
    def total_p(self):
        return self.tree[0]

    @property
    def min_p(self):
        leaves = self.tree[-self.capacity:]
        valid_leaves = leaves[leaves > 0]
        return np.min(valid_leaves) if len(valid_leaves) > 0 else 1e-8

class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[torch.device, str] = "auto",
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
        alpha: float = 0.6,
        beta: float = 0.4,
        beta_anneal: float = 1e-6,
    ):
        super().__init__(buffer_size, observation_space, action_space, device, n_envs=n_envs, optimize_memory_usage=optimize_memory_usage)
        self.alpha = alpha
        self.beta = beta
        self.beta_anneal = beta_anneal
        self.tree = SumTree(buffer_size)
        self.max_priority = 1.0

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: List[Dict[str, Any]],
    ) -> None:
        super().add(obs, next_obs, action, reward, done, infos)
        idx = (self.pos - 1) % self.buffer_size  # 最近添加的索引
        priority = self.max_priority
        self.tree.update(idx, priority ** self.alpha)
        if priority > self.max_priority:
            self.max_priority = priority

    def sample(self, batch_size: int, beta: float = None) -> Tuple[ReplayBufferSamples, torch.Tensor, np.ndarray]:
        if beta is None:
            beta = self.beta

        total_p = self.tree.total_p
        if total_p <= 0 or self.size() < batch_size:
            # 回退到均匀采样
            replay_data = super().sample(batch_size)
            weights = torch.ones(batch_size, device=self.device)
            idxs = np.arange(self.pos - batch_size, self.pos) % self.buffer_size
            return replay_data, weights, idxs

        ind = []
        priorities = []
        segment = total_p / batch_size
        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            idx, p = self.tree.get_leaf(s)
            ind.append(idx)
            priorities.append(p)

        sampling_probabilities = np.array(priorities) / total_p
        is_weight = np.power(self.size() * sampling_probabilities, -beta)
        is_weight /= is_weight.max() + 1e-10  # 归一化

        ind = np.array(ind)

        observations = self.observations[ind]
        if self.optimize_memory_usage:
            next_observations = self.observations[(ind + 1) % self.buffer_size]
        else:
            next_observations = self.next_observations[ind]
        actions = self.actions[ind]
        rewards = self.rewards[ind].reshape(-1, 1)
        dones = self.dones[ind].reshape(-1, 1)

        replay_data = ReplayBufferSamples(
            torch.as_tensor(observations, device=self.device),
            torch.as_tensor(actions, device=self.device),
            torch.as_tensor(next_observations, device=self.device),
            torch.as_tensor(dones, device=self.device),
            torch.as_tensor(rewards, device=self.device),
        )

        weights = torch.tensor(is_weight, dtype=torch.float32, device=self.device)

        return replay_data, weights, ind

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray):
        td_errors = np.abs(td_errors.flatten()) + 1e-6
        for idx, td_err in zip(indices, td_errors):
            priority = td_err
            self.max_priority = max(self.max_priority, priority)
            self.tree.update(idx, priority ** self.alpha)

    def anneal_beta(self):
        self.beta = min(1.0, self.beta + self.beta_anneal)