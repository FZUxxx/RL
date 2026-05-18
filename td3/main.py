import numpy as np
import torch
import argparse
import os
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
import utils
import TD3
from env import BalloonPlatformEnv


def eval_policy(policy, seed, eval_episodes=5, writer=None):
    eval_env = BalloonPlatformEnv()
    eval_env.seed(seed + 0)  # 设置 RL 环境种子
    eval_env.mpc_controller_pure.seed(seed + 0)  # 单独为纯 MPC 设置种子
    avg_reward = 0.0
    angle_errors_mpc = []
    angle_errors_rl = []

    for _ in range(eval_episodes):
        state, done = eval_env.reset(), False
        episode_reward = 0.0
        episode_timesteps = 0

        while not done and episode_timesteps < eval_env.max_steps:
            episode_timesteps += 1
            action = policy.select_action(np.array(state))
            state, reward, done, info = eval_env.step(action)
            episode_reward += reward
            t = info["time"]
            theta_ref = eval_env.ref_angles_rad[int(t // eval_env.duration_per_ref) % eval_env.num_refs]
            angle_errors_mpc.append(np.abs(info["angle_mpc"] - theta_ref) * 180 / np.pi)
            angle_errors_rl.append(info["angle_error"] * 180 / np.pi)

        avg_reward += episode_reward

    avg_reward /= eval_episodes
    mean_angle_error_mpc = np.mean(angle_errors_mpc)
    mean_angle_error_rl = np.mean(angle_errors_rl)
    print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}, "
          f"Mean Angle Error (MPC): {mean_angle_error_mpc:.3f}°, "
          f"Mean Angle Error (MPC+RL): {mean_angle_error_rl:.3f}°")

    if writer is not None and len(angle_errors_mpc) > 0:
        writer.add_scalar(f'Eval/MPC_Seed_Influence_{seed}', seed + 1, global_step=0)
        half = max(len(angle_errors_mpc) // 2, 1)
        writer.add_scalar(
            f'Eval/MPC_Noise_Impact_{seed}',
            np.mean(angle_errors_mpc) - np.mean(angle_errors_mpc[:half]),
            global_step=0
        )

    return avg_reward, mean_angle_error_mpc, mean_angle_error_rl


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="TD3")
    parser.add_argument("--seed", default=10, type=int)
    parser.add_argument("--start_timesteps", default=6000, type=int)
    parser.add_argument("--eval_freq", default=6000, type=int)
    parser.add_argument("--max_timesteps", default=6000000, type=int)
    parser.add_argument("--expl_noise", default=0.05, type=float)
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--discount", default=0.999, type=float)
    parser.add_argument("--tau", default=0.001, type=float)
    parser.add_argument("--policy_noise", default=0.1, type=float)
    parser.add_argument("--noise_clip", default=0.2, type=float)
    parser.add_argument("--policy_freq", default=2, type=int)
    parser.add_argument("--save_model", action="store_true")
    parser.add_argument("--load_model", default="")
    args = parser.parse_args()

    file_name = f"{args.policy}_MPC_BalloonPlatform_{args.seed}"
    print(f"Policy: {args.policy}, Base Controller: MPC(do-mpc), Env: BalloonPlatform, Seed: {args.seed}")

    if not os.path.exists("./results"):
        os.makedirs("./results")
    if args.save_model and not os.path.exists("./models"):
        os.makedirs("./models")

    timenow = str(datetime.now())[0:-7].replace(' ', '_').replace(':', '-')
    log_dir = f"runs/{file_name}_{timenow}"
    writer = SummaryWriter(log_dir=log_dir)
    print(f"TensorBoard 日志保存至: {log_dir}")

    env = BalloonPlatformEnv()
    env.seed(args.seed)
    env.action_space.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "max_action": max_action,
        "discount": args.discount,
        "tau": args.tau,
        "policy_noise": args.policy_noise * max_action,
        "noise_clip": args.noise_clip * max_action,
        "policy_freq": args.policy_freq
    }

    policy = TD3.TD3(**kwargs)
    if args.load_model != "":
        policy_file = file_name if args.load_model == "default" else args.load_model
        policy.load(f"./models/{policy_file}")

    replay_buffer = utils.ReplayBuffer(state_dim, action_dim)
    evaluations = []
    episode_rewards = []
    episode_angle_errors_mpc = []
    episode_angle_errors_rl = []

    last_t = None
    last_y_mpc = None
    last_y_rl = None
    last_u_mpc = None
    last_u_rl = None
    last_u_pure_mpc = None
    last_noise = None
    last_actions = None
    last_r_plot = None

    state, done = env.reset(), False
    episode_reward = 0.0
    episode_timesteps = 0
    episode_num = 0
    episode_angle_errors = []

    for t in range(int(args.max_timesteps)):
        episode_timesteps += 1
        if t < args.start_timesteps:
            action = env.action_space.sample()
        else:
            action = (
                policy.select_action(np.array(state))
                + np.random.normal(0, max_action * args.expl_noise, size=action_dim)
            ).clip(-max_action, max_action)

        next_state, reward, done, info = env.step(action)
        done_bool = float(done) if episode_timesteps < env.max_steps else 0
        replay_buffer.add(state, action, next_state, reward, done_bool)

        state = next_state
        episode_reward += reward
        episode_angle_errors.append(info["angle_error"])

        writer.add_scalar('Step/Reward', reward, t)
        theta_ref = env.ref_angles_rad[int(info["time"] // env.duration_per_ref) % env.num_refs]
        writer.add_scalar('Step/Angle_Error_MPC', np.abs(info["angle_mpc"] - theta_ref) * 180 / np.pi, t)
        writer.add_scalar('Step/Angle_Error_RL', info["angle_error"] * 180 / np.pi, t)
        writer.add_scalar('Step/Action', action[0], t)
        writer.add_scalar('Step/U_MPC', info["u_mpc"], t)
        writer.add_scalar('Step/U_RL', info["u_rl"], t)
        writer.add_scalar('Step/Noise', info["noise"] * 180 / np.pi, t)

        if t >= args.start_timesteps:
            policy.train(replay_buffer, args.batch_size)

        if done:
            mean_angle_error = np.mean(episode_angle_errors) * 180 / np.pi
            # 使用方波参考计算 MPC 均值误差
            ref_angles_deg = np.array([-50, 20, 50, 30, -30])
            duration_per_ref = 60.0
            num_refs = len(ref_angles_deg)
            mpc_errors = []
            for ti, a in zip(env.times, env.angles_mpc):
                ref_index = int(ti // duration_per_ref) % num_refs
                ref_angle = ref_angles_deg[ref_index]
                mpc_errors.append(np.abs(a - ref_angle))
            mean_angle_error_mpc = np.mean(mpc_errors)
            print(f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} "
                  f"Reward: {episode_reward:.3f} Mean Angle Error: {mean_angle_error:.3f}°")
            writer.add_scalar('Episode/Reward', episode_reward, episode_num)
            writer.add_scalar('Episode/Mean_Angle_Error_MPC', mean_angle_error_mpc, episode_num)
            writer.add_scalar('Episode/Mean_Angle_Error_RL', mean_angle_error, episode_num)
            episode_rewards.append(episode_reward)
            episode_angle_errors_mpc.append(mean_angle_error_mpc)
            episode_angle_errors_rl.append(mean_angle_error)

            last_t = np.array(env.times)
            last_y_mpc = np.array(env.angles_mpc)
            last_y_rl = np.array(env.angles_rl)
            last_u_mpc = np.array(env.u_mpc_history)
            last_u_rl = np.array(env.u_rl_history)
            last_u_pure_mpc = np.array(env.u_pure_mpc_history)
            last_noise = np.array(env.noise_history)
            last_actions = np.array(env.actions)

            # 使用方波参考计算 r_plot
            ref_angles_deg = np.array([-50, 20, 50, 30, -30])
            duration_per_ref = 60.0
            num_refs = len(ref_angles_deg)
            last_r_plot = np.zeros_like(last_t)
            for i, ti in enumerate(last_t):
                ref_index = int(ti // duration_per_ref) % num_refs
                last_r_plot[i] = ref_angles_deg[ref_index]

            state, done = env.reset(), False
            episode_reward = 0.0
            episode_timesteps = 0
            episode_num += 1
            episode_angle_errors = []

        if (t + 1) % args.eval_freq == 0:
            eval_reward, eval_angle_error_mpc, eval_angle_error_rl = eval_policy(policy, args.seed, writer=writer)
            evaluations.append(eval_reward)
            writer.add_scalar('Eval/Reward', eval_reward, t + 1)
            writer.add_scalar('Eval/Mean_Angle_Error_MPC', eval_angle_error_mpc, t + 1)
            writer.add_scalar('Eval/Mean_Angle_Error_RL', eval_angle_error_rl, t + 1)
            np.save(f"./results/{file_name}_rewards.npy", evaluations)
            np.save(f"./results/{file_name}_angle_errors_mpc.npy", episode_angle_errors_mpc)
            np.save(f"./results/{file_name}_angle_errors_rl.npy", episode_angle_errors_rl)
            if args.save_model:
                policy.save(f"./models/{file_name}")

    if last_t is not None:
        t = last_t
        y_mpc = last_y_mpc
        y_rl = last_y_rl
        u_mpc = last_u_mpc
        u_rl = last_u_rl
        u_pure_mpc = last_u_pure_mpc
        noise = last_noise
        actions = last_actions
        r_plot = last_r_plot
    else:
        t = np.array(env.times)
        y_mpc = np.array(env.angles_mpc)
        y_rl = np.array(env.angles_rl)
        u_mpc = np.array(env.u_mpc_history)
        u_rl = np.array(env.u_rl_history)
        u_pure_mpc = np.array(env.u_pure_mpc_history)
        noise = np.array(env.noise_history)
        actions = np.array(env.actions)
        # 使用方波参考计算 r_plot
        ref_angles_deg = np.array([-50, 20, 50, 30, -30])
        duration_per_ref = 60.0
        num_refs = len(ref_angles_deg)
        r_plot = np.zeros_like(t)
        for i, ti in enumerate(t):
            ref_index = int(ti // duration_per_ref) % num_refs
            r_plot[i] = ref_angles_deg[ref_index]

    plt.figure(figsize=(10, 16))

    plt.subplot(5, 1, 1)
    plt.plot(t, y_mpc, 'b', linewidth=2, label='纯 MPC')
    plt.plot(t, y_rl, 'g', linewidth=2, label='MPC+RL')
    plt.plot(t, r_plot, 'r--', linewidth=1.5, label='参考角度')
    plt.grid()
    plt.xlabel('时间 (s)')
    plt.ylabel('气球转角 θ_g (deg)')
    plt.legend()

    plt.subplot(5, 1, 2)
    plt.plot(t, u_mpc, 'b', linewidth=2, label='MPC 基础控制输入')
    plt.plot(t, u_rl, 'g', linewidth=2, label='MPC+RL 总控制输入')
    plt.grid()
    plt.xlabel('时间 (s)')
    plt.ylabel('总控制输入 u_a (V)')
    plt.legend()

    plt.subplot(5, 1, 3)
    plt.plot(t, actions, 'purple', linewidth=2)
    plt.grid()
    plt.xlabel('时间 (s)')
    plt.ylabel('RL 残差控制 u_rl (V)')

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
    plt.title('作用于 dot_theta_g 的风速干扰')

    plt.tight_layout()
    plt.savefig(f"./results/{file_name}_300s_results.png")
    plt.show()

    writer.close()
