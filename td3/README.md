# TD3 + do-mpc MPC Balloon Platform

此版本将原始 LQR 基础控制器替换为基于 do-mpc 的 MPC 控制器，TD3 作为残差控制器叠加到 MPC 控制输入上。

## 安装依赖

```bash
pip install gymnasium numpy scipy torch matplotlib tensorboard do-mpc casadi
```

## 运行

```bash
python main.py --save_model
```

如果 MPC 求解速度较慢，可以在 `env.py` 中调小 `self.mpc_horizon`，例如从 `25` 改为 `10`。
