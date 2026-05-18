# PERSAC.py
import torch as th
import numpy as np
from stable_baselines3 import SAC
import torch.nn.functional as F


class PERSAC(SAC):
    def __init__(
            self,
            *args,
            replay_buffer_class=None,
            replay_buffer_kwargs=None,
            alpha=0.6,
            beta=0.4,
            beta_anneal=1e-6,
            **kwargs
    ):
        if replay_buffer_kwargs is None:
            replay_buffer_kwargs = {}

        super().__init__(
            *args,
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            **kwargs
        )

        self.alpha = alpha
        self.beta = beta
        self.beta_anneal = beta_anneal

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        self.policy.set_training_mode(True)

        optimizers = [self.actor.optimizer, self.critic.optimizer]
        if self.ent_coef_optimizer is not None:
            optimizers += [self.ent_coef_optimizer]

        self._update_learning_rate(optimizers)

        ent_coef_losses, ent_coefs = [], []
        actor_losses, critic_losses = [], []

        for gradient_step in range(gradient_steps):
            replay_data, weights, idxs = self.replay_buffer.sample(batch_size, beta=self.beta)

            # 关键修复：确保所有数据是 2维
            replay_data = replay_data._replace(
                observations=replay_data.observations.reshape(batch_size, -1),
                actions=replay_data.actions.reshape(batch_size, -1),
                next_observations=replay_data.next_observations.reshape(batch_size, -1),
            )

            with th.no_grad():
                next_actions, next_log_prob = self.actor.action_log_prob(replay_data.next_observations)
                next_q_values = th.cat(
                    self.critic_target(replay_data.next_observations, next_actions), dim=1
                )
                next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)

                if self.ent_coef_optimizer is not None:
                    current_ent_coef = th.exp(self.log_ent_coef.detach()).item()
                else:
                    current_ent_coef = float(self.ent_coef)

                next_q_values = next_q_values - current_ent_coef * next_log_prob.reshape(-1, 1)
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values

            current_q_values = self.critic(replay_data.observations, replay_data.actions)

            critic_loss = sum(
                F.mse_loss(curr_q, target_q_values, reduction="none") for curr_q in current_q_values
            )
            critic_loss = (critic_loss * weights.view(-1, 1)).mean() / 2

            critic_losses.append(critic_loss.item())

            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            with th.no_grad():
                abs_td_errors = [th.abs(curr_q - target_q_values) for curr_q in current_q_values]
                stacked_errors = th.stack(abs_td_errors, dim=0)
                min_td_errors, _ = th.min(stacked_errors, dim=0)
                td_errors = min_td_errors.squeeze(-1).cpu().numpy().flatten()

            self.replay_buffer.update_priorities(idxs, td_errors)

            actions_pi, log_prob = self.actor.action_log_prob(replay_data.observations)
            log_prob = log_prob.reshape(-1, 1)

            ent_coef_loss = None
            if self.ent_coef_optimizer is not None:
                ent_coef_loss = -(self.log_ent_coef * (log_prob + self.target_entropy).detach()).mean()
                ent_coef_losses.append(ent_coef_loss.item())

                self.ent_coef_optimizer.zero_grad()
                ent_coef_loss.backward()
                self.ent_coef_optimizer.step()

            ent_coefs.append(current_ent_coef)

            q_values_pi = th.cat(self.critic(replay_data.observations, actions_pi), dim=1)
            min_qf_pi, _ = th.min(q_values_pi, dim=1, keepdim=True)
            actor_loss = (current_ent_coef * log_prob - min_qf_pi).mean()
            actor_losses.append(actor_loss.item())

            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()

            # SAC 只软更新 critic target
            with th.no_grad():
                for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        self.replay_buffer.anneal_beta()

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/ent_coef", np.mean(ent_coefs))
        self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))
        if len(ent_coef_losses) > 0:
            self.logger.record("train/ent_coef_loss", np.mean(ent_coef_losses))