import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from torch.utils.tensorboard import SummaryWriter
import os
import shutil
from datetime import datetime
from mpcwheel_withnoise import MPCBalloonController

class BalloonPlatformEnv(gym.Env):
    def __init__(self):
        super(BalloonPlatformEnv, self).__init__()
        self.mpc_controller = MPCBalloonController()
        self.state_dim = 5
        self.action_dim = 1
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.state_dim,), dtype=np.float32)
        self.action_space = gym.spaces.Box(low=-0.5, high=0.5, shape=(self.action_dim,), dtype=np.float32)
        self.state = None
        self.step_count = 0
        self.max_steps = 10000
        self.theta_ref = self.mpc_controller.get_reference()
        self.times = []
        self.actions = []
        self.angles_mpc = []
        self.angles_rl = []
        self.all_episode_data = []

    def reset(self):
        self.state = self.mpc_controller.reset()
        self.step_count = 0
        self.theta_ref = self.mpc_controller.get_reference()
        self.times = []
        self.actions = []
        self.angles_mpc = []
        self.angles_rl = []
        return self.state

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)

        # 1. 纯 MPC 控制（无 RL 补偿）
        state_mpc, y_mpc, self.theta_ref, noise_mpc = self.mpc_controller.step(rl_torque=0)
        theta_g_mpc = state_mpc[0].item() * 180 / np.pi

        # 2. RL 补偿控制（实际执行）
        self.state, y_rl, self.theta_ref, noise_rl = self.mpc_controller.step(rl_torque=action[0])
        theta_g_rl = self.state[0].item() * 180 / np.pi

        self.step_count += 1

        angle_error = np.clip(self.state[0] - self.theta_ref, -np.pi, np.pi)
        velocity = np.clip(self.state[1], -100, 100)
        flywheel_speed = np.clip(self.state[2], -100, 100)
        reward = -50 * angle_error ** 2 - 0.02 * velocity ** 2 - 0.02 * action.dot(action) - 0.01 * flywheel_speed ** 2
        if abs(angle_error) < 0.1:
            reward += 500
        if abs(angle_error) < 0.05:
            reward += 2000
        if abs(self.state[0]) > np.pi / 4:
            reward -= 100 * (abs(self.state[0]) - np.pi / 4) ** 2

        done = self.step_count >= self.max_steps
        if abs(self.state[0]) > 5 * np.pi or abs(self.state[1]) > 500 or abs(angle_error) > 2.0:
            done = True
            reward = -2000

        current_time = self.step_count * 0.05
        self.times.append(current_time)
        self.actions.append(float(action[0]))
        self.angles_mpc.append(theta_g_mpc)
        self.angles_rl.append(theta_g_rl)

        action_value = float(action[0]) if hasattr(action[0], 'item') else action[0]
        print(f"步数: {self.step_count}, 动作: {action_value:.6f}, 纯 MPC 角度: {theta_g_mpc:.2f}°, RL 补偿角度: {theta_g_rl:.2f}°, 奖励: {reward}, Done: {done}")

        return self.state.flatten(), reward, done, {}

    def close(self):
        self.all_episode_data.append({
            'times': self.times,
            'actions': self.actions,
            'angles_mpc': self.angles_mpc,
            'angles_rl': self.angles_rl
        })
        with open('last_three_episodes_angles2.txt', 'w') as f:
            for episode_idx, episode_data in enumerate(self.all_episode_data):
                f.write(f"\nEpisode {episode_idx}\n")
                for i in range(len(episode_data['times'])):
                    f.write(f"Step {i}, Time {episode_data['times'][i]:.2f}s, "
                            f"Action {episode_data['actions'][i]:.6f}, "
                            f"Pure MPC Theta_g {episode_data['angles_mpc'][i]:.2f}°, "
                            f"RL Theta_g {episode_data['angles_rl'][i]:.2f}°\n")

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, net_width, max_action):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(state_dim, net_width)
        self.l2 = nn.Linear(net_width, net_width)
        self.l3 = nn.Linear(net_width, action_dim)
        self.max_action = max_action

    def forward(self, state):
        a = torch.tanh(self.l1(state))
        a = torch.tanh(self.l2(a))
        a = torch.tanh(self.l3(a)) * self.max_action
        return a

class DoubleQCritic(nn.Module):
    def __init__(self, state_dim, action_dim, net_width):
        super(DoubleQCritic, self).__init__()
        self.l1 = nn.Linear(state_dim + action_dim, net_width)
        self.l2 = nn.Linear(net_width, net_width)
        self.l3 = nn.Linear(net_width, 1)
        self.l4 = nn.Linear(state_dim + action_dim, net_width)
        self.l5 = nn.Linear(net_width, net_width)
        self.l6 = nn.Linear(net_width, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)
        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        q2 = F.relu(self.l4(sa))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2

    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)
        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1

class ReplayBuffer:
    def __init__(self, state_dim, action_dim, max_size, device):
        self.max_size = max_size
        self.device = device
        self.ptr = 0
        self.size = 0
        self.s = torch.zeros((max_size, state_dim), dtype=torch.float, device=device)
        self.a = torch.zeros((max_size, action_dim), dtype=torch.float, device=device)
        self.r = torch.zeros((max_size, 1), dtype=torch.float, device=device)
        self.s_next = torch.zeros((max_size, state_dim), dtype=torch.float, device=device)
        self.dw = torch.zeros((max_size, 1), dtype=torch.bool, device=device)

    def add(self, s, a, r, s_next, dw):
        self.s[self.ptr] = torch.from_numpy(s).float().view(-1).to(self.device)
        self.a[self.ptr] = torch.from_numpy(a).float().view(-1).to(self.device)
        self.r[self.ptr] = torch.tensor(r, dtype=torch.float, device=self.device)
        self.s_next[self.ptr] = torch.from_numpy(s_next).float().view(-1).to(self.device)
        self.dw[self.ptr] = torch.tensor(dw, dtype=torch.bool, device=self.device)
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        ind = torch.randint(0, self.size, size=(batch_size,), device=self.device)
        return (self.s[ind], self.a[ind], self.r[ind], self.s_next[ind], self.dw[ind])

class TD3Agent:
    def __init__(self, state_dim, action_dim, max_action, device, net_width=128, a_lr=3e-5, c_lr=3e-5, gamma=0.99,
                 tau=0.005, batch_size=256, explore_noise=0.1, policy_noise=0.2, noise_clip=0.5, delay_freq=2):
        self.device = device
        self.actor = Actor(state_dim, action_dim, net_width, max_action).to(device)
        self.actor_target = Actor(state_dim, action_dim, net_width, max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=a_lr)

        self.critic = DoubleQCritic(state_dim, action_dim, net_width).to(device)
        self.critic_target = DoubleQCritic(state_dim, action_dim, net_width).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=c_lr)

        self.replay_buffer = ReplayBuffer(state_dim, action_dim, max_size=int(1e6), device=device)
        self.max_action = max_action
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.explore_noise = explore_noise
        self.policy_noise = policy_noise * max_action
        self.noise_clip = noise_clip * max_action
        self.delay_freq = delay_freq
        self.delay_counter = 0

    def select_action(self, state, deterministic=False):
        state = torch.FloatTensor(state[np.newaxis, :]).to(self.device)
        with torch.no_grad():
            action = self.actor(state).cpu().numpy()[0]
        if not deterministic:
            noise = np.random.normal(0, self.max_action * self.explore_noise, size=action.shape)
            action = (action + noise).clip(-self.max_action, self.max_action)
        return action

    def train(self):
        self.delay_counter += 1
        s, a, r, s_next, dw = self.replay_buffer.sample(self.batch_size)

        with torch.no_grad():
            noise = (torch.randn_like(a) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_action = (self.actor_target(s_next) + noise).clamp(-self.max_action, self.max_action)
            target_q1, target_q2 = self.critic_target(s_next, next_action)
            target_q = torch.min(target_q1, target_q2)
            target_q = r + (~dw) * self.gamma * target_q

        current_q1, current_q2 = self.critic(s, a)
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        if self.delay_counter >= self.delay_freq:
            actor_loss = -self.critic.Q1(s, self.actor(s)).mean()
            l2_reg = sum(torch.norm(param) for param in self.actor.parameters())
            actor_loss = actor_loss + 1e-4 * l2_reg
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            self.delay_counter = 0

    def save(self, filename):
        torch.save(self.actor.state_dict(), f"{filename}_actor.pth")
        torch.save(self.critic.state_dict(), f"{filename}_critic.pth")

    def load(self, filename):
        self.actor.load_state_dict(torch.load(f"{filename}_actor.pth"))
        self.critic.load_state_dict(torch.load(f"{filename}_critic.pth"))

def evaluate_policy(env, agent, turns=3):
    total_reward = 0
    for _ in range(turns):
        s = env.reset()
        done = False
        episode_reward = 0
        while not done:
            a = agent.select_action(s, deterministic=True)
            s, r, done, _ = env.step(a)
            episode_reward += r
        total_reward += episode_reward
    return total_reward / turns

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--write', type=bool, default=True)
    parser.add_argument('--max_train_steps', type=int, default=int(5e5))
    parser.add_argument('--eval_interval', type=int, default=int(5e3))
    parser.add_argument('--save_interval', type=int, default=int(5e4))
    parser.add_argument('--seed', type=int, default=0)
    opt = parser.parse_args()
    device = torch.device(opt.device)
    print(f"初始设备: {device}")

    env = BalloonPlatformEnv()
    agent = TD3Agent(state_dim=env.state_dim, action_dim=env.action_dim, max_action=0.5, device=device, batch_size=256,
                     net_width=128, explore_noise=0.1)
    if opt.write:
        timenow = str(datetime.now())[0:-10].replace(' ', '_').replace(':', '-')
        writepath = f'runs/BalloonPlatform_{timenow}'
        if os.path.exists(writepath): shutil.rmtree(writepath)
        writer = SummaryWriter(log_dir=writepath)

    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)

    total_steps = 0
    episode_count = 0
    while total_steps < opt.max_train_steps:
        s = env.reset()
        done = False
        episode_reward = 0
        episode_count += 1
        print(f"开始第 {episode_count} 回合, 参考角度: {env.theta_ref * 180 / np.pi:.2f} 度")

        while not done:
            try:
                if total_steps < 10000:
                    a = np.random.uniform(-0.5, 0.5, size=env.action_dim)
                else:
                    a = agent.select_action(s, deterministic=False)
                s_next, r, done, _ = env.step(a)
                agent.replay_buffer.add(s, a, r, s_next, done)
                s = s_next
                total_steps += 1
                episode_reward += r

                if total_steps >= 10000 and total_steps % 5 == 0:
                    for _ in range(10):
                        agent.train()

                if total_steps % opt.eval_interval == 0:
                    eval_reward = evaluate_policy(env, agent)
                    smoothed_reward = eval_reward if total_steps == opt.eval_interval else (eval_reward + smoothed_reward * (opt.eval_interval - 1)) / opt.eval_interval
                    print(f"步数: {total_steps // 1000}k, 评估奖励: {eval_reward}, 平滑奖励: {smoothed_reward}")
                    if opt.write:
                        writer.add_scalar('eval_reward', eval_reward, total_steps)
                        writer.add_scalar('smoothed_eval_reward', smoothed_reward, total_steps)
                        writer.add_scalar('theta_g', s[0], total_steps)
                        writer.add_scalar('theta_g_dot', s[1], total_steps)
                        writer.add_scalar('action_0', a[0], total_steps)

                if total_steps % opt.save_interval == 0:
                    agent.save(f"model/balloon_platform_{total_steps // 1000}k")

            except RuntimeError as e:
                if 'CUDA' in str(e) and 'out of memory' in str(e):
                    print(f"GPU 内存不足，切换到 CPU。错误: {e}")
                    device = torch.device('cpu')
                    agent.actor = agent.actor.to(device)
                    agent.actor_target = agent.actor_target.to(device)
                    agent.critic = agent.critic.to(device)
                    agent.critic_target = agent.critic_target.to(device)
                    agent.replay_buffer.device = device
                    torch.cuda.empty_cache()
                    print(f"已切换到设备: {device}")
                else:
                    raise e

        print(f"第 {episode_count} 回合结束，回合奖励: {episode_reward}")

    env.close()

if __name__ == "__main__":
    if not os.path.exists('model'): os.mkdir('model')
    main()