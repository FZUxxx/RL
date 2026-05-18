import numpy as np
import argparse
import os
from datetime import datetime
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

from env import BalloonPlatformEnv
from SAC import SACPolicy
from stable_baselines3.common.callbacks import BaseCallback
from torch.utils.tensorboard import SummaryWriter


class CustomCallback(BaseCallback):
    def __init__(self, eval_env, eval_freq, seed, file_name, verbose=0):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.seed = seed
        self.file_name = file_name
        self.best_mean_reward = -np.inf
        self.episode_rewards = []
        self.episode_angle_errors_mpc = []
        self.episode_angle_errors_rl = []

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            reward, mpc_err, rl_err = eval_policy(self.model, self.eval_env, self.seed)
            self.episode_rewards.append(reward)
            self.episode_angle_errors_mpc.append(mpc_err)
            self.episode_angle_errors_rl.append(rl_err)

            self.logger.record("eval/mean_reward", reward)
            self.logger.record("eval/mean_angle_error_mpc", mpc_err)
            self.logger.record("eval/mean_angle_error_rl", rl_err)

            if reward > self.best_mean_reward:
                self.best_mean_reward = reward
                self.model.save(f"./models/{self.file_name}_best")

        return True

    def _on_training_end(self) -> None:
        np.save(f"./results/{self.file_name}_rewards.npy", self.episode_rewards)
        np.save(f"./results/{self.file_name}_angle_errors_mpc.npy", self.episode_angle_errors_mpc)
        np.save(f"./results/{self.file_name}_angle_errors_rl.npy", self.episode_angle_errors_rl)


def eval_policy(model, eval_env, seed, eval_episodes=5):
    eval_env.seed(seed)
    eval_env.mpc_controller_pure.seed(seed)
    avg_reward = 0.0
    angle_errors_mpc = []
    angle_errors_rl = []

    for _ in range(eval_episodes):
        obs, _ = eval_env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _, info = eval_env.step(action)
            avg_reward += reward
            t = info["time"]
            theta_ref = eval_env.ref_angles_rad[int(t // eval_env.duration_per_ref) % eval_env.num_refs]
            angle_errors_mpc.append(np.abs(info["angle_mpc"] - theta_ref) * 180 / np.pi)
            angle_errors_rl.append(info["angle_error"] * 180 / np.pi)

    avg_reward /= eval_episodes
    mean_mpc = np.mean(angle_errors_mpc)
    mean_rl = np.mean(angle_errors_rl)
    print(f"Eval: Reward {avg_reward:.3f} | MPC误差 {mean_mpc:.3f}° | MPC+PER-SAC误差 {mean_rl:.3f}°")
    return avg_reward, mean_mpc, mean_rl


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=10, type=int)
    parser.add_argument("--start_timesteps", default=6000, type=int)
    parser.add_argument("--eval_freq", default=6000, type=int)
    parser.add_argument("--max_timesteps", default=6000000, type=int)
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--save_model", action="store_true")
    parser.add_argument("--load_model", default="")
    args = parser.parse_args()

    file_name = f"PERSAC_MPC_BalloonPlatform_{args.seed}"
    print(f"Policy: PER-SAC, Base Controller: MPC(do-mpc), Env: BalloonPlatform, Seed: {args.seed}")

    if not os.path.exists("./results"):
        os.makedirs("./results")
    if args.save_model and not os.path.exists("./models"):
        os.makedirs("./models")

    timenow = str(datetime.now())[0:-7].replace(' ', '_').replace(':', '-')
    log_dir = f"runs/{file_name}_{timenow}"
    writer = SummaryWriter(log_dir=log_dir)
    print(f"TensorBoard 日志保存至: {log_dir}")

    env = BalloonPlatformEnv()
    eval_env = BalloonPlatformEnv()

    env.seed(args.seed)
    eval_env.seed(args.seed + 100)

    policy = SACPolicy(
        env,
        learning_rate=3e-4,
        buffer_size=1000000,
        batch_size=args.batch_size,
        tau=0.005,
        gamma=0.99,
        train_freq=1,
        gradient_steps=1,
        ent_coef='auto',
        seed=args.seed,
        tensorboard_log=log_dir
    )

    if args.load_model != "":
        policy.load(f"./models/{args.load_model}", env)

    callback = CustomCallback(
        eval_env=eval_env,
        eval_freq=args.eval_freq,
        seed=args.seed,
        file_name=file_name
    )

    print("开始训练 PER-SAC（基础控制器为 do-mpc MPC）...")
    policy.learn(total_timesteps=args.max_timesteps, callback=callback)

    if args.save_model:
        policy.save(f"./models/{file_name}")

    # 最终绘图（使用最后一个 episode 的数据）
    obs, _ = env.reset()
    done = False
    while not done:
        action, _ = policy.predict(obs, deterministic=True)
        obs, reward, done, _, info = env.step(action)

    t = np.array(env.times)
    y_mpc = np.array(env.angles_mpc)
    y_rl = np.array(env.angles_rl)
    u_mpc = np.array(env.u_mpc_history)
    u_rl = np.array(env.u_rl_history)
    u_pure_mpc = np.array(env.u_pure_mpc_history)
    noise = np.array(env.noise_history)
    actions = np.array(env.actions)
    ref_angles_deg = np.array([-50, 20, 50, 30, -30])
    r_plot = np.zeros_like(t)
    for i, ti in enumerate(t):
        r_plot[i] = ref_angles_deg[int(ti // 60.0) % 5]

    plt.figure(figsize=(10, 16))

    plt.subplot(5, 1, 1)
    plt.plot(t, y_mpc, 'b', linewidth=2, label='纯 MPC')
    plt.plot(t, y_rl, 'g', linewidth=2, label='MPC+PER-SAC')
    plt.plot(t, r_plot, 'r--', linewidth=1.5, label='参考角度')
    plt.grid()
    plt.xlabel('时间 (s)')
    plt.ylabel('气球转角 θ_g (deg)')
    plt.legend()

    plt.subplot(5, 1, 2)
    plt.plot(t, u_mpc, 'b', linewidth=2, label='MPC 基础控制输入')
    plt.plot(t, u_rl, 'g', linewidth=2, label='MPC+SAC 总控制输入')
    plt.grid()
    plt.xlabel('时间 (s)')
    plt.ylabel('总控制输入 u_a (V)')
    plt.legend()

    plt.subplot(5, 1, 3)
    plt.plot(t, actions, 'purple', linewidth=2)
    plt.grid()
    plt.xlabel('时间 (s)')
    plt.ylabel('PER-SAC 残差控制 u_sac (V)')

    plt.subplot(5, 1, 4)
    plt.plot(t, u_pure_mpc, 'orange', linewidth=2, label='纯 MPC 控制输入')
    plt.grid()
    plt.xlabel('时间 (s)')
    plt.ylabel('MPC 控制输入 u_mpc (V)')
    plt.legend()

    plt.subplot(5, 1, 5)
    plt.plot(t, noise, 'g', linewidth=1.5)
    plt.grid()
    plt.xlabel('时间 (s)')
    plt.ylabel('风速干扰 (deg/s)')

    plt.tight_layout()
    plt.savefig(f"./results/{file_name}_300s_results.png")
    plt.show()

    writer.close()
