# Copyright (c) 2021-2024, The RSL-RL Project Developers.
# All rights reserved.
# Original code is licensed under the BSD-3-Clause license.
#
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# Copyright (c) 2025-2026, The Legged Lab Project Developers.
# All rights reserved.
#
# Copyright (c) 2025-2026, The TienKung-Lab Project Developers.
# All rights reserved.
# Modifications are licensed under the BSD-3-Clause license.
#
# This file contains code derived from the RSL-RL, Isaac Lab, and Legged Lab Projects,
# with additional modifications by the TienKung-Lab Project,
# and is distributed under the BSD-3-Clause license.

import argparse
import os
import sys

import mujoco
import mujoco_viewer
import numpy as np
import torch
from pynput import keyboard
import time
import threading

class SimToSimCfg:
    """Configuration class for sim2sim parameters.

    Must be kept consistent with the training configuration.
    """

    class sim:
        sim_duration = 100.0
        num_action = 14
        num_obs_per_step = 57   # 14*3+15
        actor_obs_history_length = 10
        dt = 0.005
        decimation = 4
        clip_observations = 100.0
        clip_actions = 100
        action_scale = 0.3
        realtime_factor = 1.0  # 1.0 = real time, 0.5 = 2x slower, 2.0 = 2x faster

    class robot:
        gait_air_ratio_l: float = 0.42
        gait_air_ratio_r: float = 0.42
        gait_phase_offset_l: float = 0.38
        gait_phase_offset_r: float = 0.88
        gait_cycle: float = 0.95


class MujocoRunner:
    """
    Sim2Sim runner that loads a policy and a MuJoCo model
    to run real-time humanoid control simulation.

    Args:
        cfg (SimToSimCfg): Configuration object for simulation.
        policy_path (str): Path to the TorchScript exported policy.
        model_path (str): Path to the MuJoCo XML model.
    """

    def __init__(self, cfg: SimToSimCfg, policy_path, model_path, use_joystick: bool = False):
        self.cfg = cfg
        self.use_joystick = use_joystick
        self.running = True
        network_path = policy_path
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.model.opt.timestep = self.cfg.sim.dt

        self.policy = torch.jit.load(network_path)
        self.data = mujoco.MjData(self.model)

        self.viewer = mujoco_viewer.MujocoViewer(self.model, self.data)
        self.viewer._render_every_frame = False
        self.init_variables()

    def init_variables(self) -> None:
        """Initialize simulation variables and joint index mappings."""
        self.dt = self.cfg.sim.decimation * self.cfg.sim.dt
        self.dof_pos = np.zeros(self.cfg.sim.num_action)
        self.dof_vel = np.zeros(self.cfg.sim.num_action)
        self.action = np.zeros(self.cfg.sim.num_action)
        self.default_dof_pos = np.array(
            [-0.28, 0.0, 0, 0.44, -0.16, 0, -0.28, 0.0, 0, 0.44, -0.16, 0, 0.05, 0.05]
        )
        self.episode_length_buf = 0
        self.gait_phase = np.zeros(2)
        self.gait_cycle = self.cfg.robot.gait_cycle
        self.phase_ratio = np.array([self.cfg.robot.gait_air_ratio_l, self.cfg.robot.gait_air_ratio_r])
        self.phase_offset = np.array([self.cfg.robot.gait_phase_offset_l, self.cfg.robot.gait_phase_offset_r])

        self.mujoco_to_isaac_idx = [
            0,  # left_hip_pitch 0
            12,  # left_shoulder_pitch 1
            6,  # right_hip_pitch 2
            13,  # right_shoulder_pitch 3
            1,  # left_hip_roll 4
            7,  # right_hip_roll 5
            2,  # left_hip_yaw 6
            8,  # right_hip_yaw 7
            3,  # left_knee 8
            9,  # right_knee 9
            4,  # left_ankle_pitch 10
            10,  # right_ankle_pitch 11
            5,  # left_ankle_roll 12
            11,  # right_ankle_roll 13
        ]
        self.isaac_to_mujoco_idx = [
            0,  # left_hip_pitch 0
            4,  # left_hip_roll 1
            6,  # left_hip_yaw 2
            8,  # left_knee 3
            10,  # left_ankle_pitch 4
            12,  # left_ankle_roll 5
            2,  # right_hip_pitch 6
            5,  # right_hip_roll 7
            7,  # right_hip_yaw 8
            9,  # right_knee 9
            11,  # right_ankle_pitch 10
            13,  # right_ankle_roll 11
            1,  # left_shoulder_pitch 12
            3,  # right_shoulder_pitch 13
        ]
        # Initial command vel
        self.command_vel = np.array([0.0, 0.0, 0.0])
        self.obs_history = np.zeros(
            (self.cfg.sim.num_obs_per_step * self.cfg.sim.actor_obs_history_length,), dtype=np.float32
        )

    def get_obs(self) -> np.ndarray:
        """
        Compute current observation vector from MuJoCo sensors and internal state.

        Returns:
            np.ndarray: Normalized and clipped observation history.
        """
        self.dof_pos = self.data.sensordata[16:30]  #first 16 elements are imu data, data starting from sensordata[16] are actual joint data
        self.dof_vel = self.data.sensordata[30:44]

        obs = np.concatenate(
            [
                self.data.sensor("body-angular-velocity").data.astype(np.double),  # 3
                self.quat_rotate_inverse(
                    self.data.sensor("body-orientation").data[[1, 2, 3, 0]].astype(np.double), np.array([0, 0, -1])
                ),  # 3
                self.command_vel,  # 3
                (self.dof_pos - self.default_dof_pos)[self.mujoco_to_isaac_idx],  # 14
                self.dof_vel[self.mujoco_to_isaac_idx],  # 14
                np.clip(self.action, -self.cfg.sim.clip_actions, self.cfg.sim.clip_actions),  # 14
                np.sin(2 * np.pi * self.gait_phase),  # 2
                np.cos(2 * np.pi * self.gait_phase),  # 2
                self.phase_ratio,  # 2
            ],
            axis=0,
        ).astype(np.float32)

        # Update observation history
        self.obs_history = np.roll(self.obs_history, shift=-self.cfg.sim.num_obs_per_step)
        self.obs_history[-self.cfg.sim.num_obs_per_step :] = obs.copy()

        return np.clip(self.obs_history, -self.cfg.sim.clip_observations, self.cfg.sim.clip_observations)

    def position_control(self) -> np.ndarray:
        """
        Apply position control using scaled action.

        Returns:
            np.ndarray: Target joint positions in MuJoCo order.
        """
        actions_scaled = self.action * self.cfg.sim.action_scale
        return actions_scaled[self.isaac_to_mujoco_idx] + self.default_dof_pos

    def run(self) -> None:
        """
        Run the simulation loop with keyboard-controlled commands.
        """
        self.setup_keyboard_listener()
        self.listener.start()
        if self.use_joystick:
            self.setup_joystick()

        while self.data.time < self.cfg.sim.sim_duration:
            self.obs_history = self.get_obs()
            self.action[:] = self.policy(torch.tensor(self.obs_history, dtype=torch.float32)).detach().numpy()[:14]
            self.action = np.clip(self.action, -self.cfg.sim.clip_actions, self.cfg.sim.clip_actions)

            for sim_update in range(self.cfg.sim.decimation):
                step_start_time = time.time()

                self.data.ctrl = self.position_control()
                mujoco.mj_step(self.model, self.data)
                self.viewer.render()

                elapsed = time.time() - step_start_time
                target_wall_dt = self.cfg.sim.dt / self.cfg.sim.realtime_factor
                sleep_time = target_wall_dt - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
            self.episode_length_buf += 1
            self.calculate_gait_para()

        self.running = False
        self.listener.stop()
        self.viewer.close()

    def quat_rotate_inverse(self, q: np.ndarray, v: np.ndarray) -> np.ndarray:
        """
        Rotate a vector by the inverse of a quaternion.

        Args:
            q (np.ndarray): Quaternion (x, y, z, w) format.
            v (np.ndarray): Vector to rotate.

        Returns:
            np.ndarray: Rotated vector.
        """
        q_w = q[-1]
        q_vec = q[:3]
        a = v * (2.0 * q_w**2 - 1.0)
        b = np.cross(q_vec, v) * q_w * 2.0
        c = q_vec * np.dot(q_vec, v) * 2.0

        return a - b + c

    def calculate_gait_para(self) -> None:
        """
        Update gait phase parameters based on simulation time and offset.
        """
        t = self.episode_length_buf * self.dt / self.gait_cycle
        self.gait_phase[0] = (t + self.phase_offset[0]) % 1.0
        self.gait_phase[1] = (t + self.phase_offset[1]) % 1.0

    def adjust_command_vel(self, idx: int, increment: float) -> None:
        """
        Adjust command velocity vector.

        Args:
            idx (int): Index of velocity component (0=x, 1=y, 2=yaw).
            increment (float): Value to increment.
        """
        self.command_vel[idx] += increment
        self.command_vel[idx] = np.clip(self.command_vel[idx], -1.0, 1.0)  # vel clip

    def setup_keyboard_listener(self) -> None:
        """
        Set up keyboard event listener for user control input.
        """

        def on_press(key):
            try:
                if key.char == "8":  # NumPad 8      x += 0.2
                    self.adjust_command_vel(0, 0.2)
                elif key.char == "2":  # NumPad 2      x -= 0.2
                    self.adjust_command_vel(0, -0.2)
                elif key.char == "4":  # NumPad 4      y -= 0.2
                    self.adjust_command_vel(1, -0.2)
                elif key.char == "6":  # NumPad 6      y += 0.2
                    self.adjust_command_vel(1, 0.2)
                elif key.char == "7":  # NumPad 7      yaw += 0.2
                    self.adjust_command_vel(2, -0.2)
                elif key.char == "9":  # NumPad 9      yaw -= 0.2
                    self.adjust_command_vel(2, 0.2)
            except AttributeError:
                pass

        self.listener = keyboard.Listener(on_press=on_press)

    def setup_joystick(self, max_lin: float = 1.0, max_yaw: float = 1.0) -> None:
        """
        Set up joystick/gamepad control for command velocity.
        Left stick:  x/y translation
        Right stick (X axis): yaw rate.
        """
        try:
            import os
            os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
            import pygame
        except ImportError:
            print("[WARN] pygame not installed, joystick disabled. `pip install pygame` to enable.")
            return

        # Only init joystick, don't touch display/video
        pygame.joystick.init()

        pygame.init()
        pygame.joystick.init()
        if pygame.joystick.get_count() == 0:
            print("[WARN] No joystick detected, joystick control disabled.")
            return

        js = pygame.joystick.Joystick(0)
        js.init()
        print(f"[INFO] Joystick connected: {js.get_name()}")

        def joystick_loop():
            clock = pygame.time.Clock()
            while self.running:
                # Process events so joystick state updates
                pygame.event.pump()

                # Typical mapping:
                #   axis 0: left stick X  (left/right)
                #   axis 1: left stick Y  (up/down)
                #   axis 3: right stick X (yaw)
                lx = js.get_axis(0)   # left/right
                ly = js.get_axis(1)   # forward/back
                try:
                    rx = js.get_axis(3)
                except IndexError:
                    rx = 0.0

                # Map to command velocities
                vx = -ly * max_lin          # forward: push stick up
                vy = lx * max_lin           # left/right
                yaw = rx * max_yaw          # yaw rate

                # Write into same command_vel used by keyboard
                self.command_vel[0] = np.clip(vx, -1.0, 1.0)
                self.command_vel[1] = np.clip(vy, -1.0, 1.0)
                self.command_vel[2] = np.clip(yaw, -1.0, 1.0)

                clock.tick(60)  # ~60 Hz joystick polling

            # Cleanup on exit
            js.quit()
            pygame.joystick.quit()

        self.joystick_thread = threading.Thread(target=joystick_loop, daemon=True)
        self.joystick_thread.start()


if __name__ == "__main__":
    LEGGED_LAB_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
    parser = argparse.ArgumentParser(description="Run sim2sim Mujoco controller.")
    parser.add_argument(
        "--task",
        type=str,
        default="gp2_walk",
        choices=["gp2_walk", "gp2_run"],
        help="Task type: 'walk' or 'run' to set gait parameters",
    )
    parser.add_argument(
        "--policy",
        type=str,
        default=None,
        help="Path to policy.pt. If not specified, it will be set automatically based on --task",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=os.path.join(LEGGED_LAB_ROOT_DIR, "legged_lab/assets/gp2_v2/mjcf/robot/xyber_gp2/xyber_gp2_serial.xml"),
        help="Path to model.xml",
    )
    parser.add_argument("--duration", type=float, default=100.0, help="Simulation duration in seconds")
    parser.add_argument(
        "--joystick",
        action="store_true",
        help="Use joystick/gamepad to control command velocity",
    )
    args = parser.parse_args()

    if args.policy is None:
        args.policy = os.path.join(LEGGED_LAB_ROOT_DIR, "Exported_policy", f"{args.task}.pt")

    if not os.path.isfile(args.policy):
        print(f"[ERROR] Policy file not found: {args.policy}")
        sys.exit(1)
    if not os.path.isfile(args.model):
        print(f"[ERROR] MuJoCo model file not found: {args.model}")
        sys.exit(1)

    print(f"[INFO] Loaded task preset: {args.task.upper()}")
    print(f"[INFO] Loaded policy: {args.policy}")
    print(f"[INFO] Loaded model: {args.model}")

    sim_cfg = SimToSimCfg()
    sim_cfg.sim.sim_duration = args.duration

    # Set gait parameters according to task
    if args.task == "gp2_walk":
        sim_cfg.robot.gait_air_ratio_l = 0.42
        sim_cfg.robot.gait_air_ratio_r = 0.42
        sim_cfg.robot.gait_phase_offset_l = 0.38
        sim_cfg.robot.gait_phase_offset_r = 0.88
        sim_cfg.robot.gait_cycle = 0.95
    elif args.task == "gp2_run":
        sim_cfg.robot.gait_air_ratio_l = 0.6
        sim_cfg.robot.gait_air_ratio_r = 0.6
        sim_cfg.robot.gait_phase_offset_l = 0.6
        sim_cfg.robot.gait_phase_offset_r = 0.1
        sim_cfg.robot.gait_cycle = 0.5

    runner = MujocoRunner(
        cfg=sim_cfg,
        policy_path=args.policy,
        model_path=args.model,
        use_joystick=args.joystick,
    )
    runner.run()
