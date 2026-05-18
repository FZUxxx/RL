import gymnasium as gym
import numpy as np
import do_mpc


# 风速扰动类
class WindDisturbance:
    def __init__(self, amplitude=0.005, freq_min=1/80, freq_max=1/70, noise_scale=0.0001, np_random=None):
        self.amplitude = amplitude
        self.freq_min = freq_min
        self.freq_max = freq_max
        self.noise_scale = noise_scale
        self.np_random = np_random if np_random is not None else np.random
        self.phi = self.np_random.uniform(0, 2 * np.pi)
        self.frequency = self.np_random.uniform(self.freq_min, self.freq_max)

    def wind_disturbance(self, t):
        """模拟高空风速干扰，包含周期性正弦波和随机扰动。"""
        periodic_component = self.amplitude * np.sin(2 * np.pi * self.frequency * t + self.phi)
        random_component = self.np_random.uniform(-self.noise_scale, self.noise_scale)
        return periodic_component + random_component

    def seed(self, seed):
        """设置随机种子。"""
        self.np_random, _ = gym.utils.seeding.np_random(seed)
        self.phi = self.np_random.uniform(0, 2 * np.pi)
        self.frequency = self.np_random.uniform(self.freq_min, self.freq_max)


# MPC 控制器类：使用 do-mpc 替换原来的 LQRController
class MPCController:
    def __init__(
        self,
        A,
        B,
        Q,
        R,
        u_max=10.0,
        delta_u_max=1.0,
        Ts=0.05,
        n_horizon=25,
        ref_angles_rad=None,
        duration_per_ref=60.0,
    ):
        self.A = np.array(A, dtype=float)
        self.B = np.array(B, dtype=float)
        self.Q = np.array(Q, dtype=float)
        self.R = np.array(R, dtype=float)
        self.u_max = float(u_max)
        self.delta_u_max = float(delta_u_max)
        self.Ts = float(Ts)
        self.n_horizon = int(n_horizon)
        self.ref_angles_rad = None if ref_angles_rad is None else np.array(ref_angles_rad, dtype=float)
        self.duration_per_ref = float(duration_per_ref)
        self.current_theta_ref = 0.0
        self.u_prev = 0.0
        self.wind_disturbance = WindDisturbance()

        self.model = self._build_model()
        self.mpc = self._build_mpc()
        self.reset()

    def _build_model(self):
        """构建与环境欧拉积分一致的离散状态空间模型。"""
        model = do_mpc.model.Model('discrete')

        theta_g = model.set_variable(var_type='_x', var_name='theta_g')
        omega_g = model.set_variable(var_type='_x', var_name='omega_g')
        theta_a = model.set_variable(var_type='_x', var_name='theta_a')
        i_a = model.set_variable(var_type='_x', var_name='i_a')
        u_a = model.set_variable(var_type='_u', var_name='u_a')
        theta_ref = model.set_variable(var_type='_tvp', var_name='theta_ref')

        x = [theta_g, omega_g, theta_a, i_a]
        dx = []
        for row in range(4):
            rhs = 0
            for col in range(4):
                rhs += self.A[row, col] * x[col]
            rhs += self.B[row, 0] * u_a
            dx.append(rhs)

        model.set_rhs('theta_g', theta_g + self.Ts * dx[0])
        model.set_rhs('omega_g', omega_g + self.Ts * dx[1])
        model.set_rhs('theta_a', theta_a + self.Ts * dx[2])
        model.set_rhs('i_a', i_a + self.Ts * dx[3])

        # 便于在 MPC 目标函数中引用
        self._x_vars = {
            'theta_g': theta_g,
            'omega_g': omega_g,
            'theta_a': theta_a,
            'i_a': i_a,
        }
        self._u_a = u_a
        self._theta_ref = theta_ref

        model.setup()
        return model

    def _build_mpc(self):
        mpc = do_mpc.controller.MPC(self.model)

        # 新版 do-mpc 推荐通过 settings 配置；set_param 仍然兼容。
        mpc.settings.n_horizon = self.n_horizon
        mpc.settings.t_step = self.Ts
        mpc.settings.n_robust = 0
        mpc.settings.store_full_solution = False
        mpc.settings.nlpsol_opts = {
            'ipopt.max_iter': 80,
            'ipopt.tol': 1e-4,
            'ipopt.print_level': 0,
            'print_time': False,
        }
        try:
            mpc.settings.supress_ipopt_output()
        except Exception:
            pass

        theta_g = self._x_vars['theta_g']
        omega_g = self._x_vars['omega_g']
        theta_a = self._x_vars['theta_a']
        i_a = self._x_vars['i_a']
        u_a = self._u_a
        theta_ref = self._theta_ref

        q = np.diag(self.Q).astype(float)
        r_u = float(np.asarray(self.R).reshape(-1)[0])

        # 跟踪角度参考，同时抑制速度、电机状态、电流和控制输入幅值。
        lterm = (
            q[0] * (theta_g - theta_ref) ** 2
            + q[1] * omega_g ** 2
            + q[2] * theta_a ** 2
            + q[3] * i_a ** 2
            + r_u * u_a ** 2
        )
        # 终端项不使用 tvp，避免不同 do-mpc 版本对 mterm 中 tvp 支持不一致。
        mterm = 0.0 * theta_g
        mpc.set_objective(mterm=mterm, lterm=lterm)

        # set_rterm 是对控制增量 Δu 的软惩罚；硬限幅仍在 compute_control 中保留。
        mpc.set_rterm(u_a=0.1)

        # 控制输入硬约束。
        mpc.bounds['lower', '_u', 'u_a'] = -self.u_max
        mpc.bounds['upper', '_u', 'u_a'] = self.u_max

        # 角度边界，和环境中的裁剪范围一致。
        mpc.bounds['lower', '_x', 'theta_g'] = -np.pi
        mpc.bounds['upper', '_x', 'theta_g'] = np.pi

        tvp_template = mpc.get_tvp_template()

        def tvp_fun(t_now):
            t0 = float(np.asarray(t_now).reshape(-1)[0])
            for k in range(self.n_horizon + 1):
                tvp_template['_tvp', k, 'theta_ref'] = self._reference_at_time(t0 + k * self.Ts)
            return tvp_template

        mpc.set_tvp_fun(tvp_fun)
        mpc.setup()
        return mpc

    def _reference_at_time(self, t):
        if self.ref_angles_rad is None or len(self.ref_angles_rad) == 0:
            return float(self.current_theta_ref)
        ref_index = int(max(float(t), 0.0) // self.duration_per_ref) % len(self.ref_angles_rad)
        return float(self.ref_angles_rad[ref_index])

    def compute_control(self, x, x_ref, t):
        x = np.asarray(x, dtype=float).reshape(4, 1)
        self.current_theta_ref = float(np.asarray(x_ref, dtype=float).reshape(-1)[0])

        # 让 tvp_fun 可以基于真实仿真时间预测未来参考。
        try:
            self.mpc.t0 = float(t)
        except Exception:
            pass

        try:
            u_mpc = self.mpc.make_step(x)
            u_nominal = float(np.asarray(u_mpc).reshape(-1)[0])
        except Exception as exc:
            # 训练时如果某一步优化失败，退回上一时刻控制量，避免环境崩溃。
            # 如需调试，可取消下一行注释查看异常。
            # print(f"MPC solve failed at t={t:.3f}: {exc}")
            u_nominal = self.u_prev

        noise_effect = self.wind_disturbance.wind_disturbance(t)
        u_with_noise = u_nominal + noise_effect

        # 保留原代码中的控制增量限制。
        delta_u = u_with_noise - self.u_prev
        delta_u = np.clip(delta_u, -self.delta_u_max, self.delta_u_max)
        u = self.u_prev + delta_u
        u = float(np.clip(u, -self.u_max, self.u_max))
        self.u_prev = u
        return u

    def reset(self, x0=None):
        self.u_prev = 0.0
        self.current_theta_ref = 0.0
        if x0 is None:
            x0 = np.zeros((4, 1))
        x0 = np.asarray(x0, dtype=float).reshape(4, 1)
        try:
            self.mpc.reset_history()
            self.mpc.x0 = x0
            self.mpc.u0 = np.zeros((1, 1))
            self.mpc.t0 = 0.0
            self.mpc.set_initial_guess()
        except Exception:
            pass

    def seed(self, seed):
        self.wind_disturbance = WindDisturbance()
        self.wind_disturbance.seed(seed)


# 气球平台环境
class BalloonPlatformEnv(gym.Env):
    def __init__(self):
        super(BalloonPlatformEnv, self).__init__()
        # 系统参数
        self.Ce = 0.1
        self.Ca = -19.35
        self.Ra = 0.5
        self.La = 0.35
        self.Ja = 0.03
        self.Ba = 0.1
        self.If = 30
        self.Ig = 166
        self.ks = 0
        self.theta_bal = 0
        self.Ts = 0.05

        # 状态空间矩阵
        self.A = np.array([
            [0, 1, 0, 0],
            [-self.ks / self.Ig, 0, self.If * self.Ba / (self.Ig * (self.Ja + self.If)), self.If * self.Ca / (self.Ig * (self.Ja + self.If))],
            [0, 0, -self.Ba / (self.Ja + self.If), self.Ca / (self.Ja + self.If)],
            [0, 0, -self.Ce / self.La, -self.Ra / self.La]
        ], dtype=float)
        self.B = np.array([[0], [0], [0], [1 / self.La]], dtype=float)
        self.C = np.array([[1, 0, 0, 0]], dtype=float)
        self.D = np.array([[0]], dtype=float)

        # MPC 参数：沿用原来的 Q / R 权重，R 同时进入控制输入幅值惩罚。
        self.Q = np.diag([50.0, 1.0, 10.0, 1.0])
        self.R = np.array([[0.2]])
        self.u_max = 10.0
        self.delta_u_max = 1.0
        self.mpc_horizon = 25

        # 方波参考信号参数
        self.ref_angles_rad = np.deg2rad(np.array([-50, 20, 50, 30, -30]))
        self.duration_per_ref = 60.0
        self.num_refs = len(self.ref_angles_rad)

        # 环境初始化
        self.np_random = None
        self.wind = WindDisturbance(np_random=self.np_random)
        self.mpc_controller = MPCController(
            self.A,
            self.B,
            self.Q,
            self.R,
            self.u_max,
            self.delta_u_max,
            Ts=self.Ts,
            n_horizon=self.mpc_horizon,
            ref_angles_rad=self.ref_angles_rad,
            duration_per_ref=self.duration_per_ref,
        )
        self.mpc_controller_pure = MPCController(
            self.A,
            self.B,
            self.Q,
            self.R,
            self.u_max,
            self.delta_u_max,
            Ts=self.Ts,
            n_horizon=self.mpc_horizon,
            ref_angles_rad=self.ref_angles_rad,
            duration_per_ref=self.duration_per_ref,
        )
        self.state_mpc = np.zeros((4, 1))
        self.state_dim = 6  # state (4), noise (1), u_mpc (1)
        self.action_dim = 1
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.state_dim,), dtype=np.float32)
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(self.action_dim,), dtype=np.float32)
        self.max_action = 1.0
        self.state = np.zeros((4, 1))
        self.u_mpc = 0.0
        self.step_count = 0
        self.max_steps = int(300 / self.Ts)  # 300秒，采样时间0.05秒

        # 历史记录
        self.angles_mpc = []
        self.angles_rl = []
        self.u_mpc_history = []
        self.u_rl_history = []
        self.u_pure_mpc_history = []
        self.noise_history = []
        self.actions = []
        self.times = []

    def seed(self, seed=None):
        """设置环境的随机种子。"""
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        self.action_space.seed(seed)
        self.wind.seed(seed)
        self.mpc_controller.seed(seed)
        self.mpc_controller_pure.seed(seed)
        return [seed]

    def reset(self):
        self.state = np.zeros((4, 1))
        self.state_mpc = np.zeros((4, 1))
        self.u_mpc = 0.0
        self.step_count = 0
        self.mpc_controller.reset(self.state)
        self.mpc_controller_pure.reset(self.state_mpc)
        self.angles_mpc = []
        self.angles_rl = []
        self.u_mpc_history = []
        self.u_rl_history = []
        self.u_pure_mpc_history = []
        self.noise_history = []
        self.actions = []
        self.times = []
        noise = self.wind.wind_disturbance(0)
        return np.concatenate([self.state.flatten(), [noise], [self.u_mpc]]).astype(np.float32)

    def step(self, action):
        action = np.clip(action, -self.max_action, self.max_action)
        t = self.step_count * self.Ts
        noise = self.wind.wind_disturbance(t)

        # 方波参考角度
        ref_index = int(t // self.duration_per_ref) % self.num_refs
        theta_ref = self.ref_angles_rad[ref_index]
        x_ref = np.array([[theta_ref], [0], [0], [0]])

        # 计算 MPC 控制（用于 RL 环境）
        u_mpc = self.mpc_controller.compute_control(self.state, x_ref, t)
        u_rl = float(action[0]) * 5.0
        u_total = u_mpc + u_rl
        u_total = float(np.clip(u_total, -self.u_max, self.u_max))

        # 更新状态（MPC+RL）
        u_input = np.array([[u_total]])
        x_dot = self.A @ self.state + self.B @ u_input
        state_rl = self.state + self.Ts * x_dot
        state_rl[0] = np.clip(state_rl[0], -np.pi, np.pi)
        state_rl[1] += noise
        y_rl = (self.C @ state_rl).item()

        # 使用独立的 MPC 控制器更新纯 MPC 状态
        u_mpc_pure = self.mpc_controller_pure.compute_control(self.state_mpc, x_ref, t)
        u_input_mpc = np.array([[u_mpc_pure]])
        x_dot_mpc = self.A @ self.state_mpc + self.B @ u_input_mpc
        self.state_mpc = self.state_mpc + self.Ts * x_dot_mpc
        self.state_mpc[0] = np.clip(self.state_mpc[0], -np.pi, np.pi)
        self.state_mpc[1] += self.mpc_controller_pure.wind_disturbance.wind_disturbance(t)
        y_mpc = (self.C @ self.state_mpc).item()

        # 计算奖励（保持原样）
        angle_error = np.abs(y_rl - theta_ref)
        reward = -0.1 * angle_error - 0.01 * u_total**2 + 1 * np.exp(-100 * angle_error)

        # 更新环境状态
        self.state = state_rl
        self.u_mpc = u_mpc
        self.step_count += 1
        done = self.step_count >= self.max_steps

        # 记录历史数据
        self.angles_mpc.append(y_mpc * 180 / np.pi)
        self.angles_rl.append(y_rl * 180 / np.pi)
        self.u_mpc_history.append(u_mpc)
        self.u_rl_history.append(u_total)
        self.u_pure_mpc_history.append(u_mpc_pure)
        self.noise_history.append(noise * 180 / np.pi)
        self.actions.append(u_rl)
        self.times.append(t)

        next_state = np.concatenate([self.state.flatten(), [noise], [self.u_mpc]]).astype(np.float32)
        info = {
            "noise": noise,
            "angle_error": angle_error,
            "angle_mpc": y_mpc,
            "time": t,
            "u_mpc": u_mpc,
            "u_rl": u_rl
        }
        return next_state, reward, done, info

    def close(self):
        pass
