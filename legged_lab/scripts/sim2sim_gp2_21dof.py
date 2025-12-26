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
        num_action = 21
        num_obs_per_step = 78   # 21*3+15
        actor_obs_history_length = 10
        dt = 0.005
        decimation = 4
        clip_observations = 100.0
        clip_actions = 100
        action_scale = 0.25
        realtime_factor = 1.0  # 1.0 = real time, 0.5 = 2x slower, 2.0 = 2x faster

    class robot:
        gait_air_ratio_l: float = 0.55
        gait_air_ratio_r: float = 0.55
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

        self.mode = "stand"
        # define default_dof_pos etc. first
        self.init_variables()

        self.build_pd_profiles()
        self.cache_default_actuator_gains()
        self.set_mode(self.mode)
        self.apply_pd_profile("stand" if self.mode == "stand" else "walk")

        n = self.cfg.sim.num_action  # 21
        mujoco.mj_resetData(self.model, self.data)

        # free joint takes first 7 qpos entries
        self.data.qpos[7:7+n] = self.default_dof_pos.copy()
        self.data.qvel[:] = 0.0

        mujoco.mj_forward(self.model, self.data)

        self.viewer = mujoco_viewer.MujocoViewer(self.model, self.data)
        self.viewer._render_every_frame = False
        self.init_variables()

        # --- Push-test state ---
        self.push_steps_remaining = 0
        self.push_force = np.zeros(3)  # world-frame force [Fx, Fy, Fz]
        # pick the body you want to push, adjust name if needed
        self.push_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "base_link"
        )

    def init_variables(self) -> None:
        """Initialize simulation variables and joint index mappings."""
        self.dt = self.cfg.sim.decimation * self.cfg.sim.dt
        self.dof_pos = np.zeros(self.cfg.sim.num_action)
        self.dof_vel = np.zeros(self.cfg.sim.num_action)
        self.action = np.zeros(self.cfg.sim.num_action)
        self.default_dof_pos = np.array(
            [-0.2, 0.0, 0.0, 0.3, -0.17, 0.0, #left leg
             -0.2, 0.0, 0.0, 0.3, -0.17, 0.0, #right leg
             0.0, #waist
             0.2, 0.25, 0.0, 0.97, #left arm
             0.2, -0.25, 0.0, 0.97] #right arm
        )
        self.episode_length_buf = 0
        self.gait_phase = np.zeros(2)
        self.gait_cycle = self.cfg.robot.gait_cycle
        self.phase_ratio = np.array([self.cfg.robot.gait_air_ratio_l, self.cfg.robot.gait_air_ratio_r])
        self.phase_offset = np.array([self.cfg.robot.gait_phase_offset_l, self.cfg.robot.gait_phase_offset_r])

        self.mujoco_to_isaac_idx = [
            0,  # left_hip_pitch 0
            6,  # right_hip_pitch 1
            12,  # waist_yaw 2
            1,  # left_hip_roll 3
            7,  # right_hip_roll 4
            13,  # left_shoulder_pitch 5
            17,  # right_shoulder_pitch 6
            2,  # left_hip_yaw 7
            8,  # right_hip_yaw 8
            14,  # left_shoulder_roll 9
            18,  # right_shoulder_roll 10
            3,  # left_knee 11
            9,  # right_knee 12
            15,  # left_shoulder_yaw 13
            19,  # right_shoulder_yaw 14
            4,  # left_ankle_pitch 15
            10,  # right_ankle_pitch 16
            16,  # left_elbow 17
            20,  # right_elbow 18
            5,  # left_ankle_roll 19
            11,  # right_ankle_roll 20

        ]
        self.isaac_to_mujoco_idx = [
            0,  # left_hip_pitch 0
            3,  # left_hip_roll 1
            7,  # left_hip_yaw 2
            11,  # left_knee 3
            15,  # left_ankle_pitch 4
            19,  # left_ankle_roll 5
            1,  # right_hip_pitch 6
            4,  # right_hip_roll 7
            8,  # right_hip_yaw 8
            12,  # right_knee 9
            16,  # right_ankle_pitch 10
            20,  # right_ankle_roll 11
            2,  # waist_yaw 12
            5,  # left_shoulder_pitch 13
            9,  # left_shoulder_roll 14
            13,  # left_shoulder_yaw 15
            17,  # left_elbow 16
            6,  # right_shoulder_pitch 17
            10,  # right_shoulder_roll 18
            14,  # right_shoulder_yaw 19
            18,  # right_elbow 20
        ]
        # Initial command vel
        self.command_vel = np.array([0.0, 0.0, 0.0])
        self.obs_history = np.zeros(
            (self.cfg.sim.num_obs_per_step * self.cfg.sim.actor_obs_history_length,), dtype=np.float32
        )

        # --- NEW: locomotion state machine params ---
        # thresholds on command magnitude
        self.loco_enter_thresh = 0.20   # cmd_mag above this -> consider walking
        self.loco_exit_thresh  = 0.05   # cmd_mag below this -> consider standing
        # dwell times
        self.loco_enter_time   = 0.20   # [s] cmd must be active to enter WALK
        self.loco_exit_time    = 0.50   # [s] cmd must be neutral to return STAND
        # timers
        self.loco_cmd_active_time  = 0.0
        self.loco_cmd_neutral_time = 0.0

    def build_pd_profiles(self):
        # From your 2real pd_stand (screenshot): arms 150/1, waist_yaw 150/1,
        # legs: hip_pitch 400/2, hip_roll 400/2, hip_yaw 200/1.5, knee 500/3, ankle_pitch 150/1
        # Extra: ankle_roll is not in 2real list -> pick something reasonable (150/1).
        self.pd_stand_kp = {
            # arms
            "left_shoulder_pitch_joint": 150.0,
            "left_shoulder_roll_joint":  150.0,
            "left_shoulder_yaw_joint":   150.0,
            "left_elbow_joint":          150.0,
            "right_shoulder_pitch_joint": 150.0,
            "right_shoulder_roll_joint":  150.0,
            "right_shoulder_yaw_joint":   150.0,
            "right_elbow_joint":          150.0,

            # waist (you currently have waist_yaw in 21dof)
            "waist_yaw_joint": 150.0,

            # left leg
            "left_hip_pitch_joint": 400.0,
            "left_hip_roll_joint":  400.0,
            "left_hip_yaw_joint":   200.0,
            "left_knee_joint":      500.0,
            "left_ankle_pitch_joint": 150.0,
            "left_ankle_roll_joint":  150.0,  # not in 2real list -> added

            # right leg
            "right_hip_pitch_joint": 400.0,
            "right_hip_roll_joint":  400.0,
            "right_hip_yaw_joint":   200.0,
            "right_knee_joint":      500.0,
            "right_ankle_pitch_joint": 150.0,
            "right_ankle_roll_joint":  150.0, # not in 2real list -> added
        }

        self.pd_stand_kd = {
            # arms
            "left_shoulder_pitch_joint": 1.0,
            "left_shoulder_roll_joint":  1.0,
            "left_shoulder_yaw_joint":   1.0,
            "left_elbow_joint":          1.0,
            "right_shoulder_pitch_joint": 1.0,
            "right_shoulder_roll_joint":  1.0,
            "right_shoulder_yaw_joint":   1.0,
            "right_elbow_joint":          1.0,

            # waist
            "waist_yaw_joint": 1.0,

            # left leg
            "left_hip_pitch_joint": 2.0,
            "left_hip_roll_joint":  2.0,
            "left_hip_yaw_joint":   1.5,
            "left_knee_joint":      3.0,
            "left_ankle_pitch_joint": 1.0,
            "left_ankle_roll_joint":  1.0,

            # right leg
            "right_hip_pitch_joint": 2.0,
            "right_hip_roll_joint":  2.0,
            "right_hip_yaw_joint":   1.5,
            "right_knee_joint":      3.0,
            "right_ankle_pitch_joint": 1.0,
            "right_ankle_roll_joint":  1.0,
        }

        self.pd_walk_kp = {
            # legs
            "left_hip_pitch_joint": 150.0,
            "left_hip_roll_joint":  100.0,
            "left_hip_yaw_joint":   100.0,
            "left_knee_joint":      180.0,
            "left_ankle_pitch_joint": 150.0,
            "left_ankle_roll_joint":   40.0,

            "right_hip_pitch_joint": 150.0,
            "right_hip_roll_joint":  100.0,
            "right_hip_yaw_joint":   100.0,
            "right_knee_joint":      180.0,
            "right_ankle_pitch_joint": 150.0,
            "right_ankle_roll_joint":   40.0,

            # waist
            "waist_yaw_joint": 150.0,

            # arms
            "left_shoulder_pitch_joint": 90.0,
            "left_shoulder_roll_joint":  20.0,
            "left_shoulder_yaw_joint":   20.0,
            "left_elbow_joint":          30.0,

            "right_shoulder_pitch_joint": 90.0,
            "right_shoulder_roll_joint":  20.0,
            "right_shoulder_yaw_joint":   20.0,
            "right_elbow_joint":          30.0,
        }

        self.pd_walk_kd = {
            # arms
            "left_shoulder_pitch_joint": 4.0,
            "left_shoulder_roll_joint":  4.0,
            "left_shoulder_yaw_joint":   4.0,
            "left_elbow_joint":          4.0,
            "right_shoulder_pitch_joint": 4.0,
            "right_shoulder_roll_joint":  4.0,
            "right_shoulder_yaw_joint":   4.0,
            "right_elbow_joint":          4.0,

            # waist
            "waist_yaw_joint": 6.0,

            # left leg
            "left_hip_pitch_joint": 5.0,
            "left_hip_roll_joint":  4.0,
            "left_hip_yaw_joint":   4.0,
            "left_knee_joint":      10.0,
            "left_ankle_pitch_joint": 8.0,
            "left_ankle_roll_joint":  4.0,

            # right leg
            "right_hip_pitch_joint": 5.0,
            "right_hip_roll_joint":  4.0,
            "right_hip_yaw_joint":   4.0,
            "right_knee_joint":      10.0,
            "right_ankle_pitch_joint": 8.0,
            "right_ankle_roll_joint":  4.0,
        }

    def cache_default_actuator_gains(self):
        # so "walk" can restore the XML defaults
        self._gainprm_default = self.model.actuator_gainprm.copy()
        self._biasprm_default = self.model.actuator_biasprm.copy()

    def _set_position_actuator_kp_kd(self, act_id: int, kp: float, kd: float):
        # For MuJoCo position actuators: gainprm[0]=kp, biasprm[1]=-kp, biasprm[2]=-kd
        self.model.actuator_gainprm[act_id, 0] = kp
        self.model.actuator_biasprm[act_id, 1] = -kp
        self.model.actuator_biasprm[act_id, 2] = -kd

    def apply_pd_profile(self, profile: str):
        if profile == "walk":
            # restore XML defaults
            self.model.actuator_gainprm[:] = self._gainprm_default
            self.model.actuator_biasprm[:] = self._biasprm_default

            for act_id in range(self.model.nu):
                j_id = int(self.model.actuator_trnid[act_id, 0])
                j_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, j_id)
                if j_name is None:
                    continue
                if j_name in self.pd_walk_kp:
                    kp = float(self.pd_walk_kp[j_name])

                    # "get damping later":
                    # Option A (recommended for stability): keep kd from XML
                    # kd = float(-self._biasprm_default[act_id, 2])

                    # Option B (pure stiffness-only test): 
                    kd = float(self.pd_walk_kd[j_name])

                    self._set_position_actuator_kp_kd(act_id, kp, kd)
            return

        if profile != "stand":
            return

        # stand profile: set kp/kd by the JOINT each actuator drives
        for act_id in range(self.model.nu):
            # actuator_trnid[act_id, 0] is joint id for joint actuators
            j_id = int(self.model.actuator_trnid[act_id, 0])
            j_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, j_id)
            if j_name is None:
                continue

            if j_name in self.pd_stand_kp:
                kp = self.pd_stand_kp[j_name]
                kd = self.pd_stand_kd[j_name]
                self._set_position_actuator_kp_kd(act_id, kp, kd)

    def set_mode(self, new_mode: str):
        cur_mode = getattr(self, "mode", None)
        if new_mode == cur_mode:
            return
            
        if new_mode == self.mode:
            return
        self.mode = new_mode
        if self.mode == "stand":
            self.apply_pd_profile("stand")
        else:
            self.apply_pd_profile("walk")

    def get_obs(self) -> np.ndarray:
        """
        Compute current observation vector from MuJoCo sensors and internal state.

        Returns:
            np.ndarray: Normalized and clipped observation history.
        """
        self.dof_pos = self.data.sensordata[16:37]  #first 16 elements are imu data, data starting from sensordata[16] are actual joint data
        self.dof_vel = self.data.sensordata[37:58]

        obs = np.concatenate(
            [
                self.data.sensor("body-angular-velocity").data.astype(np.double),  # 3
                self.quat_rotate_inverse(
                    self.data.sensor("body-orientation").data[[1, 2, 3, 0]].astype(np.double), np.array([0, 0, -1])
                ),  # 3
                self.command_vel,  # 3
                (self.dof_pos - self.default_dof_pos)[self.mujoco_to_isaac_idx],  # 21
                self.dof_vel[self.mujoco_to_isaac_idx],  # 21
                np.clip(self.action, -self.cfg.sim.clip_actions, self.cfg.sim.clip_actions),  # 21
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
        Compute target joint positions in MuJoCo order, depending on mode.

        - 'stand': hold default_dof_pos using PD.
        - 'walk' : RL policy output around default_dof_pos.
        """
        if self.mode == "stand":
            # In stand mode, ignore RL action and just hold the default pose.
            # (Later you can tweak PD gains per mode at the real controller level.)
            return self.default_dof_pos

        # Walk mode: same as before
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
            self.action[:] = self.policy(torch.tensor(self.obs_history, dtype=torch.float32)).detach().numpy()[:21]
            self.action = np.clip(self.action, -self.cfg.sim.clip_actions, self.cfg.sim.clip_actions)

            # mute = [16, 17, 22, 23]  # Isaac order
            # self.action[mute] = 0.0

            # --- NEW: automatic stand/walk switching based on joystick command ---
            self.update_locomotion_mode(self.dt)

            for sim_update in range(self.cfg.sim.decimation):
                step_start_time = time.time()

                # Apply external push if scheduled
                if self.push_steps_remaining > 0:
                    # xfrc_applied: 6D force/torque for each body, in world frame
                    # [Fx, Fy, Fz, Tx, Ty, Tz]
                    base = self.push_body_id * 6
                    self.data.xfrc_applied[base + 0] = self.push_force[0]
                    self.data.xfrc_applied[base + 1] = self.push_force[1]
                    self.data.xfrc_applied[base + 2] = self.push_force[2]
                    # no torque
                    self.data.xfrc_applied[base + 3: base + 6] = 0.0

                    self.push_steps_remaining -= 1
                else:
                    # clear any previous external forces
                    self.data.xfrc_applied[:] = 0.0

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

    def update_locomotion_mode(self, dt: float):
        """
        Automatic stand/walk switching based on command_vel magnitude.

        - If joystick is moved for loco_enter_time -> mode = 'walk'
        - If joystick stays near zero for loco_exit_time -> mode = 'stand'
        """
        # command magnitude: translational + yaw
        cmd_mag = float(np.linalg.norm(self.command_vel[:2]) + abs(self.command_vel[2]))

        # track how long command has been "active" vs "neutral"
        if cmd_mag > self.loco_enter_thresh:
            self.loco_cmd_active_time  += dt
            self.loco_cmd_neutral_time  = 0.0
        elif cmd_mag < self.loco_exit_thresh:
            self.loco_cmd_neutral_time += dt
            self.loco_cmd_active_time   = 0.0
        else:
            # between thresholds -> reset timers (hold current mode)
            self.loco_cmd_active_time  = 0.0
            self.loco_cmd_neutral_time = 0.0

        # state transitions
        if self.mode == "stand":
            if self.loco_cmd_active_time > self.loco_enter_time:
                print("[LOCO] auto-switch STAND -> WALK (cmd active)")
                self.set_mode("walk")
                # reset timer so we don't immediately flip back
                self.loco_cmd_active_time = 0.0

        elif self.mode == "walk":
            if self.loco_cmd_neutral_time > self.loco_exit_time:
                print("[LOCO] auto-switch WALK -> STAND (cmd neutral)")
                self.set_mode("stand")
                self.command_vel[:] = 0.0
                self.loco_cmd_neutral_time = 0.0
            pass

    def adjust_command_vel(self, idx: int, increment: float) -> None:
        """
        Adjust command velocity vector.

        Args:
            idx (int): Index of velocity component (0=x, 1=y, 2=yaw).
            increment (float): Value to increment.
        """
        self.command_vel[idx] += increment
        self.command_vel[idx] = np.clip(self.command_vel[idx], -1.0, 1.0)  # vel clip

    def apply_push(self, force_xyz, duration=0.2):
        """
        Schedule a push on the torso.

        Args:
            force_xyz: np.array or list of 3 floats, world-frame force [Fx, Fy, Fz] in Newtons.
            duration:  time to apply the force (seconds).
        """
        self.push_force = np.array(force_xyz, dtype=float)
        self.push_steps_remaining = int(duration / self.cfg.sim.dt)
        print(f"[INFO] Scheduled push: F={self.push_force}, steps={self.push_steps_remaining}")

    def print_action_debug(self):
        """Print current action vector and highlight ankle pitches."""
        a = self.action
        # full vector
        print("[DEBUG] action (isaac order):")
        print("  ", np.array2string(a, precision=4, floatmode="fixed"))

        # highlight a few key joints
        print("[DEBUG] key joints (isaac indices):")
        print(f"  left_hip_pitch   [0]  = {a[0]: .4f}")
        print(f"  left_knee        [3]  = {a[3]: .4f}")
        print(f"  left_ankle_pitch [4]  = {a[4]: .4f}")
        print(f"  right_hip_pitch  [6]  = {a[6]: .4f}")
        print(f"  right_knee       [9]  = {a[9]: .4f}")
        print(f"  right_ankle_pitch[10] = {a[10]: .4f}")

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
                    self.adjust_command_vel(2, 0.2)
                elif key.char == "9":  # NumPad 9      yaw -= 0.2
                    self.adjust_command_vel(2, -0.2)
                # --- NEW: mode switching ---
                elif key.char in ("s", "S"):
                    print("[INFO] Switch mode -> STAND")
                    self.set_mode("stand")
                    # optional: zero command when entering stand
                    self.command_vel[:] = 0.0
                elif key.char in ("w", "W"):
                    print("[INFO] Switch mode -> WALK")
                    self.set_mode("walk")
                    print("[INFO] Printing current action vector at zero-cmd state")
                    self.print_action_debug()
                # --- NEW: print root height ---
                elif key.char in ("h", "H"):
                    # qpos[0:3] = root position (x, y, z)
                    z = float(self.data.qpos[2])
                    print(f"[DEBUG] Current root height z = {z:.4f}")
                # --- NEW: small push test ---
                elif key.char in ("p", "P"):
                    # example: forward push in +x direction of 200 N for 0.2 s
                    # tune magnitude/duration as needed
                    self.apply_push(force_xyz=[14.0, 0.0, 0.0], duration=0.2)
                # --- NEW: print current action vector ---
                elif key.char in ("a", "A"):
                    print("[INFO] Printing current action vector at zero-cmd state")
                    self.print_action_debug()
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

                try:
                    if js.get_button(0):   # A button -> walk
                        self.set_mode("walk")
                    if js.get_button(1):   # B button -> stand
                        self.set_mode("stand")
                        self.command_vel[:] = 0.0
                except Exception:
                    pass

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
        default=os.path.join(LEGGED_LAB_ROOT_DIR, "legged_lab/assets/gp2_v2/mjcf/gp2_21dof.xml"),
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
        sim_cfg.robot.gait_air_ratio_l = 0.55
        sim_cfg.robot.gait_air_ratio_r = 0.55
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
