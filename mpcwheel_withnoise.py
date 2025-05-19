import numpy as np
import do_mpc

class MPCBalloonController:
    def __init__(self):
        # 定义参数
        self.Ce = 0.1
        self.Ca = 19.35
        self.Ra = 0.5
        self.La = 0.35
        self.Ja = 0.03
        self.Ba = 0.1
        self.If = 30
        self.Ig = 166
        self.ks = 0
        self.theta_bal = 1
        self.Ts = 0.05

        # 状态空间矩阵
        self.A = np.array([
            [0, 1, 0, 0],
            [-self.ks/self.Ig, 0, self.If*self.Ba/(self.Ig*(self.Ja+self.If)), -self.If*self.Ca/(self.Ig*(self.Ja+self.If))],
            [0, 0, -self.Ba/(self.Ja+self.If), self.Ca/(self.Ja+self.If)],
            [0, 0, -self.Ce/self.La, -self.Ra/self.La]
        ])
        self.B = np.array([[0], [0], [0], [1/self.La]])
        self.C = np.array([[1, 0, 0, 0]])
        self.D = np.array([[0]])

        # 扩展状态空间以包含误差积分项
        self.A_aug = np.vstack((np.hstack((self.A, np.zeros((4, 1)))), np.hstack((-self.C, np.array([[0]])))))
        self.B_aug = np.vstack((self.B, np.zeros((1, 1))))
        self.E_aug = np.vstack((np.zeros((4, 1)), np.array([[1]])))

        # 创建 do_mpc 模型
        model_type = 'continuous'
        self.model = do_mpc.model.Model(model_type)

        # 定义状态变量
        self.theta_g = self.model.set_variable(var_type='_x', var_name='theta_g', shape=(1, 1))
        self.dot_theta_g = self.model.set_variable(var_type='_x', var_name='dot_theta_g', shape=(1, 1))
        self.dot_theta_a = self.model.set_variable(var_type='_x', var_name='dot_theta_a', shape=(1, 1))
        self.i_a = self.model.set_variable(var_type='_x', var_name='i_a', shape=(1, 1))
        self.int_error = self.model.set_variable(var_type='_x', var_name='int_error', shape=(1, 1))

        # 定义控制输入
        self.u_a = self.model.set_variable(var_type='_u', var_name='u_a', shape=(1, 1))

        # 定义参考输入
        self.r = self.model.set_variable(var_type='_tvp', var_name='r')

        # 状态方程
        x = np.array([self.theta_g, self.dot_theta_g, self.dot_theta_a, self.i_a, self.int_error])
        u = np.array([self.u_a])
        dxdt = self.A_aug @ x + self.B_aug @ u + self.E_aug * self.r
        self.model.set_rhs('theta_g', dxdt[0])
        self.model.set_rhs('dot_theta_g', dxdt[1])
        self.model.set_rhs('dot_theta_a', dxdt[2])
        self.model.set_rhs('i_a', dxdt[3])
        self.model.set_rhs('int_error', dxdt[4])

        # 输出
        self.model.set_expression('y', self.theta_g)
        self.model.setup()

        # 配置 MPC 控制器
        self.mpc = do_mpc.controller.MPC(self.model)
        delta_u_max = 0.5
        setup_mpc = {
            'n_horizon': 20,
            't_step': self.Ts,
            'n_robust': 0,
            'store_full_solution': True,
        }
        self.mpc.set_param(**setup_mpc)
        self.mpc.set_param(
            nlpsol_opts={
                'ipopt.max_iter': 200,
                'ipopt.tol': 1e-5,
                'ipopt.print_level': 0,
            }
        )

        # 定义代价函数
        Q = np.diag([200, 1, 10, 1, 50])
        x_vec = self.model.x.cat
        cost = x_vec.T @ Q @ x_vec
        self.mpc.set_objective(mterm=cost, lterm=cost)
        self.mpc.set_rterm(u_a=0.2)

        # 状态和输入约束
        self.mpc.bounds['lower', '_x', 'theta_g'] = -np.pi
        self.mpc.bounds['upper', '_x', 'theta_g'] = np.pi
        self.mpc.bounds['lower', '_u', 'u_a'] = -10
        self.mpc.bounds['upper', '_u', 'u_a'] = 10

        # 配置模拟器
        self.simulator = do_mpc.simulator.Simulator(self.model)
        self.simulator.set_param(t_step=self.Ts)

        # 初始化参考输入和时间
        self.r_value = np.random.uniform(-1, 1)
        self.r_history = [self.r_value]
        self.t = 0  # 初始化仿真时间

        # 定义时间变化参数函数
        def tvp_fun_mpc(t_now):
            steps_per_change = int(80 / self.Ts)
            current_step = int(t_now / self.Ts)
            if current_step % steps_per_change == 0 and current_step > 0:
                self.r_value = np.random.uniform(-1, 1)
            tvp_template = self.mpc.get_tvp_template()
            for k in range(setup_mpc['n_horizon'] + 1):
                tvp_template['_tvp', k, 'r'] = self.r_value
            return tvp_template

        def tvp_fun_sim(t_now):
            steps_per_change = int(80 / self.Ts)
            current_step = int(t_now / self.Ts)
            if current_step % steps_per_change == 0 and current_step > 0:
                self.r_value = np.random.uniform(-1, 1)
                self.r_history.append(self.r_value)
            tvp_template = self.simulator.get_tvp_template()
            tvp_template['r'] = self.r_value
            return tvp_template

        self.mpc.set_tvp_fun(tvp_fun_mpc)
        self.simulator.set_tvp_fun(tvp_fun_sim)

        # 设置 MPC 和模拟器
        self.mpc.setup()
        self.simulator.setup()

        # 初始状态
        self.x0 = np.array([self.theta_bal, 0, 0, 0, 0])
        self.simulator.x0 = self.x0
        self.mpc.x0 = self.x0
        self.mpc.set_initial_guess()

        # 初始化控制输入
        self.u_prev = 0.0

    def wind_disturbance(self, t, amplitude=0.3, frequency=1/80, noise_scale=0.02):
        """模拟高空风速干扰"""
        periodic_component = amplitude * np.sin(2 * np.pi * frequency * t)
        random_component = np.random.uniform(-noise_scale, noise_scale)
        return periodic_component + random_component

    def step(self, rl_torque=0.0):
        """执行一步仿真，接受 RL 补偿力矩"""
        # 计算 MPC 控制输入
        u_mpc = self.mpc.make_step(self.x0)

        # 加上 RL 补偿力矩（转换为电压形式，假设力矩到电压的转换系数为 10）
        torque_to_voltage = 10.0
        u_rl = rl_torque * torque_to_voltage
        u_total = u_mpc + u_rl
        u_total = np.clip(u_total, -10, 10)

        # 仿真一步
        self.x0 = self.simulator.make_step(u_total)

        # 添加风速干扰
        noise = self.wind_disturbance(self.t)
        self.x0[1] += noise

        # 更新仿真时间
        self.t += self.Ts

        # 计算输出
        y = (self.C @ self.x0[:4]).item()

        # 返回状态、输出和参考角度
        return self.x0, y, self.r_value, noise

    def reset(self):
        """重置状态"""
        self.x0 = np.array([self.theta_bal, 0, 0, 0, 0])
        self.simulator.x0 = self.x0
        self.mpc.x0 = self.x0
        self.mpc.set_initial_guess()
        self.r_value = np.random.uniform(-1, 1)
        self.r_history = [self.r_value]
        self.t = 0  # 重置仿真时间
        self.u_prev = 0.0
        return self.x0

    def get_reference(self):
        """获取当前参考角度"""
        return self.r_value

    def get_state(self):
        """获取当前状态"""
        return self.x0