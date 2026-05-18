# per_buffer.py
import numpy as np
import torch as th
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.type_aliases import ReplayBufferSamples


class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, buffer_size, observation_space, action_space, device="cpu", n_envs=1, alpha=0.6, eps=1e-6):
        super().__init__(buffer_size, observation_space, action_space, device, n_envs=n_envs)
        self.alpha = alpha
        self.eps = eps

        # 优先级（初始最大优先级）
        it_capacity = 1
        while it_capacity < buffer_size:
            it_capacity *= 2
        self._it_sum = SegmentTree(it_capacity)      # 用于采样
        self._it_min = SegmentTree(it_capacity)      # 用于归一化
        self._max_priority = 1.0

    def add(self, obs, action, reward, next_obs, done, infos):
        # 使用当前最大优先级
        priority = self._max_priority ** self.alpha
        super().add(obs, action, reward, next_obs, done, infos)
        self._it_sum[self.pos - 1] = priority
        self._it_min[self.pos - 1] = priority

    def _get_samples(self, batch_inds: np.ndarray, env=None) -> ReplayBufferSamples:
        # 获取采样概率
        p_total = self._it_sum.sum(0, len(self))
        p_samples = self._it_sum.sum(batch_inds) / p_total

        # IS weights: (1 / (N * P(i))) ^ beta
        is_weights = np.power(p_total * p_samples + self.eps, -self.beta)  # beta 会通过 set_beta 更新
        is_weights /= is_weights.max() + self.eps  # 归一化

        # 原始数据采样
        data = super()._get_samples(batch_inds, env=env)

        return ReplayBufferSamples(
            observations=data.observations,
            actions=data.actions,
            next_observations=data.next_observations,
            dones=data.dones,
            rewards=data.rewards,
            # 新增字段
            weights=th.tensor(is_weights, device=data.observations.device, dtype=th.float32).unsqueeze(1),
            batch_inds=th.tensor(batch_inds, device=data.observations.device)
        )

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        priorities = np.abs(priorities) + self.eps
        clipped = np.clip(priorities, 0, 1e6)
        powered = clipped ** self.alpha
        self._it_sum.update_batch(indices, powered)
        self._it_min.update_batch(indices, powered)
        self._max_priority = max(self._max_priority, clipped.max())

    def set_beta(self, beta: float):
        self.beta = beta


# Segment Tree 实现（用于高效采样）
class SegmentTree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)

    def sum(self, start=0, end=None):
        if end is None:
            end = self.capacity
        return self._sum(0, 0, self.capacity - 1, start, end - 1)

    def _sum(self, node, node_start, node_end, query_start, query_end):
        if query_start > node_end or query_end < node_start:
            return 0
        if query_start <= node_start and node_end <= query_end:
            return self.tree[node]
        mid = (node_start + node_end) // 2
        return self._sum(2 * node + 1, node_start, mid, query_start, query_end) + \
               self._sum(2 * node + 2, mid + 1, node_end, query_start, query_end)

    def update_batch(self, indices, values):
        indices = indices + self.capacity - 1
        self.tree[indices] = values
        while np.any(indices > 0):
            indices = (indices - 1) // 2
            left = 2 * indices + 1
            right = 2 * indices + 2
            self.tree[indices] = self.tree[left] + self.tree[right]

    def sample(self, batch_size):
        total = self.sum()
        segment = total / batch_size
        samples = []
        for i in range(batch_size):
            mass = np.random.uniform(i * segment, (i + 1) * segment)
            samples.append(self._retrieve(0, 0, self.capacity - 1, mass))
        return np.array(samples)

    def _retrieve(self, node, start, end, mass):
        if start == end:
            return start
        mid = (start + end) // 2
        if self.tree[2 * node + 1] >= mass:
            return self._retrieve(2 * node + 1, start, mid, mass)
        else:
            return self._retrieve(2 * node + 2, mid + 1, end, mass - self.tree[2 * node + 1])