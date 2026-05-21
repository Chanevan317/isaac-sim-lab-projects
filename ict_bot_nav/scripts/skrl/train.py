# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to train RL agent with skrl.

Visit the skrl documentation (https://skrl.readthedocs.io) to see the examples structured in
a more user-friendly way.
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys
from unittest import runner

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with skrl.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent",
    type=str,
    default=None,
    help=(
        "Name of the RL agent configuration entry point. Defaults to None, in which case the argument "
        "--algorithm is used to determine the default agent configuration entry point."
    ),
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint to resume training.")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument("--export_io_descriptors", action="store_true", default=False, help="Export IO descriptors.")
parser.add_argument(
    "--ml_framework",
    type=str,
    default="torch",
    choices=["torch", "jax", "jax-numpy"],
    help="The ML framework used for training the skrl agent.",
)
parser.add_argument(
    "--algorithm",
    type=str,
    default="PPO",
    choices=["AMP", "PPO", "IPPO", "MAPPO"],
    help="The RL algorithm used for training the skrl agent.",
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import random
from datetime import datetime

import omni
import skrl
from packaging import version

# check for minimum supported skrl version
SKRL_VERSION = "1.4.3"
if version.parse(skrl.__version__) < version.parse(SKRL_VERSION):
    skrl.logger.error(
        f"Unsupported skrl version: {skrl.__version__}. "
        f"Install supported version using 'pip install skrl>={SKRL_VERSION}'"
    )
    exit()

if args_cli.ml_framework.startswith("torch"):
    from skrl.utils.runner.torch import Runner
elif args_cli.ml_framework.startswith("jax"):
    from skrl.utils.runner.jax import Runner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml

from isaaclab_rl.skrl import SkrlVecEnvWrapper

import ict_bot_nav.tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config


# config shortcuts
if args_cli.agent is None:
    algorithm = args_cli.algorithm.lower()
    agent_cfg_entry_point = "skrl_cfg_entry_point" if algorithm in ["ppo"] else f"skrl_{algorithm}_cfg_entry_point"
else:
    agent_cfg_entry_point = args_cli.agent
    algorithm = agent_cfg_entry_point.split("_cfg")[0].split("skrl_")[-1].lower()


@hydra_task_config(args_cli.task, agent_cfg_entry_point)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict):
    """Train with skrl agent."""
    # override configurations with non-hydra CLI arguments
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # check for invalid combination of CPU device with distributed training
    if args_cli.distributed and args_cli.device is not None and "cpu" in args_cli.device:
        raise ValueError(
            "Distributed training is not supported when using CPU device. "
            "Please use GPU device (e.g., --device cuda) for distributed training."
        )

    # multi-gpu training config
    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
    # max iterations for training
    if args_cli.max_iterations:
        agent_cfg["trainer"]["timesteps"] = args_cli.max_iterations * agent_cfg["agent"]["rollouts"]
    agent_cfg["trainer"]["close_environment_at_exit"] = False
    # configure the ML framework into the global skrl variable
    if args_cli.ml_framework.startswith("jax"):
        skrl.config.jax.backend = "jax" if args_cli.ml_framework == "jax" else "numpy"

    # randomly sample a seed if seed = -1
    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)

    # set the agent and environment seed from command line
    # note: certain randomization occur in the environment initialization so we set the seed here
    agent_cfg["seed"] = args_cli.seed if args_cli.seed is not None else agent_cfg["seed"]
    env_cfg.seed = agent_cfg["seed"]

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "skrl", agent_cfg["agent"]["experiment"]["directory"])
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + f"_{algorithm}_{args_cli.ml_framework}"
    # The Ray Tune workflow extracts experiment name using the logging line below, hence, do not change it (see PR #2346, comment-2819298849)
    print(f"Exact experiment name requested from command line: {log_dir}")
    if agent_cfg["agent"]["experiment"]["experiment_name"]:
        log_dir += f'_{agent_cfg["agent"]["experiment"]["experiment_name"]}'
    # set directory into agent config
    agent_cfg["agent"]["experiment"]["directory"] = log_root_path
    agent_cfg["agent"]["experiment"]["experiment_name"] = log_dir
    # update log_dir
    log_dir = os.path.join(log_root_path, log_dir)

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    # get checkpoint path (to resume training)
    resume_path = retrieve_file_path(args_cli.checkpoint) if args_cli.checkpoint else None

    # set the IO descriptors output directory if requested
    if isinstance(env_cfg, ManagerBasedRLEnvCfg):
        env_cfg.export_io_descriptors = args_cli.export_io_descriptors
        env_cfg.io_descriptors_output_dir = os.path.join(log_root_path, log_dir)
    else:
        omni.log.warn(
            "IO descriptors are only supported for manager based RL environments. No IO descriptors will be exported."
        )

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv) and algorithm in ["ppo"]:
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for skrl
    env = SkrlVecEnvWrapper(env, ml_framework=args_cli.ml_framework)  # same as: `wrap_env(env, wrapper="auto")`

    # configure and instantiate the skrl runner
    # https://skrl.readthedocs.io/en/latest/api/utils/runner.html
    runner = Runner(env, agent_cfg)
    env.unwrapped.runner = runner


    # ---- TRAINING LOGS ----
    import torch

    ep_rewards          = torch.zeros(env_cfg.scene.num_envs, device=env_cfg.sim.device)
    ep_lengths          = torch.zeros(env_cfg.scene.num_envs, device=env_cfg.sim.device)
    ep_distance_x       = torch.zeros(env_cfg.scene.num_envs, device=env_cfg.sim.device)
    ep_collisions       = torch.zeros(env_cfg.scene.num_envs, device=env_cfg.sim.device)
    ep_count            = 0

    original_record_transition = runner.agent.record_transition

    def custom_record_transition(
        states, actions, rewards, next_states,
        terminated, truncated, infos, timestep, timesteps
    ):
        nonlocal ep_rewards, ep_lengths, ep_count
        nonlocal ep_distance_x, ep_collisions

        original_record_transition(
            states, actions, rewards, next_states,
            terminated, truncated, infos, timestep, timesteps
        )

        ep_rewards    += rewards.squeeze(-1)
        ep_lengths    += 1

        # X corridor progress this step
        current_x      = env.scene["robot"].data.root_pos_w[:, 0]
        ep_distance_x  = current_x - env.episode_start_x   # always positive if moving forward

        # Collision terminations — terminated but not stagnation
        # stagnation_termination fires terminated, time_out fires truncated
        # so terminated = stagnation OR collision depending on which fired
        # we disambiguate via the lidar_collision termination flag if accessible,
        # otherwise use terminated as a proxy
        dones = (terminated | truncated).squeeze(-1)

        if dones.any():
            if not hasattr(custom_record_transition, "_buf"):
                custom_record_transition._buf = {
                    "rewards":     [],
                    "lengths":     [],
                    "distance_x":  [],
                    "timeouts":    [],
                    "stagnations": [],
                    "collisions":  [],
                    "obs_level":   [],
                    "success":     [],
                }

            buf = custom_record_transition._buf

            done_rewards    = ep_rewards[dones].clone()
            done_lengths    = ep_lengths[dones].clone()
            done_distance_x = ep_distance_x[dones].clone()
            done_timeouts   = truncated.squeeze(-1)[dones].float()
            done_stagnated  = terminated.squeeze(-1)[dones].float()

            # Success: robot traveled at least 4.0 m along corridor X in episode
            done_success = (done_distance_x >= 4.0).float()

            # Curriculum level
            obs_level = getattr(env, "_obs_curr_level", 0)

            buf["rewards"].append(done_rewards)
            buf["lengths"].append(done_lengths)
            buf["distance_x"].append(done_distance_x)
            buf["timeouts"].append(done_timeouts)
            buf["stagnations"].append(done_stagnated)
            buf["success"].append(done_success)
            buf["obs_level"].append(
                torch.full((dones.sum().item(),), obs_level,
                        dtype=torch.float32, device=env_cfg.sim.device)
            )

            ep_count              += dones.sum().item()
            ep_rewards[dones]      = 0.0
            ep_lengths[dones]      = 0.0

        if timestep % 500 == 0:
            buf = getattr(custom_record_transition, "_buf", {})

            print("\n" + "=" * 70)
            print(f"  Timestep            : {timestep:>8} / {timesteps}  "
                f"({100 * timestep / timesteps:.1f}%)")
            print(f"  Episodes finished   : {int(ep_count)}")

            if buf.get("rewards"):
                all_r      = torch.cat(buf["rewards"])
                all_l      = torch.cat(buf["lengths"])
                all_dx     = torch.cat(buf["distance_x"])
                all_to     = torch.cat(buf["timeouts"])
                all_st     = torch.cat(buf["stagnations"])
                all_succ   = torch.cat(buf["success"])
                all_level  = torch.cat(buf["obs_level"])

                step_dt     = env_cfg.sim.dt * env_cfg.decimation
                mean_ep_s   = all_l.mean().item() * step_dt
                mean_speed  = all_dx.mean().item() / (mean_ep_s + 1e-6)
                success_rt  = all_succ.mean().item()
                timeout_rt  = all_to.mean().item()
                stagnate_rt = all_st.mean().item()
                curr_level  = int(all_level[-1].item())

                # Curriculum schedule labels for readability
                level_labels = {
                    0: "L0 — warm-up (0 obs)",
                    1: "L1 — 1 static",
                    2: "L2 — 2 static",
                    3: "L3 — 2 slow moving",
                    4: "L4 — 3 moderate",
                    5: "L5 — 4 full speed",
                }

                print("-" * 70)
                print(f"  Mean Reward         : {all_r.mean().item():>8.3f}   ↑ want rising")
                print(f"  Max  Reward         : {all_r.max().item():>8.3f}")
                print(f"  Min  Reward         : {all_r.min().item():>8.3f}")
                print("-" * 70)
                print(f"  Mean Ep Duration    : {mean_ep_s:>7.1f} s")
                print(f"  Mean X Progress     : {all_dx.mean().item():>7.2f} m   ↑ want rising")
                print(f"  Max  X Progress     : {all_dx.max().item():>7.2f} m")
                print(f"  Mean Speed (X/t)    : {mean_speed:>7.3f} m/s  ↑ toward 0.6+")
                print("-" * 70)
                print(f"  Success Rate        : {success_rt:>7.1%}   ↑ want > 70%")
                print(f"  Timeout Rate        : {timeout_rt:>7.1%}   ↑ want high (not stagnating)")
                print(f"  Stagnation Rate     : {stagnate_rt:>7.1%}   ↓ want low")
                print("-" * 70)
                print(f"  Curriculum Level    : {curr_level}  {level_labels.get(curr_level, '')}")

                # Show level distribution if mixed
                for lvl in range(6):
                    frac = (all_level == lvl).float().mean().item()
                    if frac > 0.0:
                        bar = "█" * int(frac * 20)
                        print(f"    L{lvl} episodes      : {frac:>5.1%}  {bar}")

                # Warn if stagnation is dominating
                if stagnate_rt > 0.5:
                    print("  ⚠ stagnation > 50% — check reward weights or spawn range")
                if success_rt < 0.1 and curr_level > 0:
                    print("  ⚠ success < 10% at non-zero level — curriculum may need demotion")
                if mean_speed < 0.2:
                    print("  ⚠ mean speed < 0.2 m/s — policy may be learning to stand still")

                # Clear buffers
                for key in buf:
                    buf[key] = []

            else:
                print("  No episodes finished yet in this window")

            print("=" * 70 + "\n")

    runner.agent.record_transition = custom_record_transition
    # ---- END TRAINING LOGS ----


    # load checkpoint (if specified)
    if resume_path:
        print(f"[INFO] Loading model checkpoint from: {resume_path}")
        runner.agent.load(resume_path)

    # run training
    runner.run()

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
