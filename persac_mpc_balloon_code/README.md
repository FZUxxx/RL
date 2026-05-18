# PER-SAC + do-mpc MPC Balloon Platform

This version replaces the original LQR baseline controller with a do-mpc-based MPC controller.

Install dependencies:

```bash
pip install do-mpc casadi stable-baselines3 gymnasium torch tensorboard matplotlib numpy
```

Run:

```bash
python main.py --save_model
```

If training is too slow, reduce `self.mpc_horizon` in `env.py`, for example from 25 to 10.
