# Copyright (c) 2025-2026, The TienKung-Lab Project Developers.
# All rights reserved.
# Modifications are licensed under the BSD-3-Clause license.

"""Quick smoke test for the gp2_v2 walking environment."""

from legged_lab.envs.gp2_v2 import Gp2Env, Gp2WalkFlatEnvCfg
from rsl_rl.utils import parse_sim_params, update_sim_params_from_cfg
from scripts.train import AppLauncher


if __name__ == "__main__":
    app_launcher = AppLauncher(headless=True)
    app_launcher.parse_args()
    sim_params = parse_sim_params()

    cfg = Gp2WalkFlatEnvCfg()
    update_sim_params_from_cfg(sim_params, cfg.sim)

    env = Gp2Env(cfg=cfg, headless=True)
    print("Number of DOFs:", env.robot.num_dofs)
    print("Joint names:", env.robot.joint_names)
    print("Body names:", env.robot.body_names)

    actions = env.robot.data.default_joint_pos.repeat(env.num_envs, 1)
    for _ in range(5):
        env.step(actions)
    print("Stepped environment successfully.")

    app_launcher.close_app()