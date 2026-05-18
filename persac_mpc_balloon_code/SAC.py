# SAC.py
from PERSAC import PERSAC  # 导入新 PERSAC
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
import os
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from prioritized_replay_buffer import PrioritizedReplayBuffer

class SACPolicy:
    def __init__(self, env, **kwargs):
        self.env = env
        self.model = PERSAC(
            "MlpPolicy",
            env,
            verbose=1,
            alpha=0.6,           # ← 直接传给 PERSAC
            beta=0.4,
            beta_anneal=1e-6,
            replay_buffer_class=PrioritizedReplayBuffer,  # 必须指定
            **kwargs
        )

    def set_tensorboard(self, log_dir):
        """手动设置 TensorBoard（兼容旧版）"""
        self.writer = SummaryWriter(log_dir=log_dir)
        self.model.tensorboard_log = log_dir  # 兼容性设置

    def learn(self, total_timesteps, callback=None):
        self.model.learn(total_timesteps=total_timesteps, callback=callback)

    def save(self, path):
        self.model.save(path)

    def load(self, path, env):
        self.model = PERSAC.load(path, env=env)

    def predict(self, obs, deterministic=True):
        return self.model.predict(obs, deterministic=deterministic)

    def select_action(self, state):
        action, _ = self.predict(state, deterministic=True)
        return action