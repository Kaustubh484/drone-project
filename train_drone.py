import os
import json
import torch
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import deque

# Imports
from pybullet_env import PyBulletDroneEnv
from waypoint_manager import WaypointManager
from utils import initialize_agent

def parse_args():
    parser = argparse.ArgumentParser(description="Train drone in PyBullet")
    
    # Core
    parser.add_argument("--run_tag", type=str, default="pb_lidar_run")
    parser.add_argument("--algo", type=str, default="ddpg", choices=["ddpg", "ppo", "sac"])
    parser.add_argument("--episodes", type=int, default=5000)
    parser.add_argument("--headless", action="store_true", help="Run without GUI (faster)")
    
    # Hyperparams
    parser.add_argument("--actor_lr", type=float, default=1e-4)
    parser.add_argument("--critic_lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--buffer_size", type=int, default=100000)
    parser.add_argument("--ppo_epochs", type=int, default=10)
    parser.add_argument("--ppo_clip", type=float, default=0.2)
    
    # Environment
    parser.add_argument("--control_mode", type=str, default="velocity")
    parser.add_argument("--use_rotation_matrix", action="store_true")
    parser.add_argument("--save_interval", type=int, default=100)
    
    return parser.parse_args()

def get_log_columns(use_rot_mat, num_rays):
    cols = ["step", "reward", "rel_x", "rel_y", "rel_z"]
    cols += [f"rot_{i}" for i in range(9)] if use_rot_mat else ["qw", "qx", "qy", "qz"]
    cols += ["vx", "vy", "vz", "wx", "wy", "wz"]
    cols += [f"lidar_{i}" for i in range(num_rays)]
    cols += ["act_vx", "act_vy", "act_vz", "act_yaw"]
    return cols

# --- FIX 1: Pass wpm to run_episode ---
def run_episode(agent, env, args, wpm, episode_num=0):
    state, _ = env.reset()
    
    # --- FIX 2: Redraw waypoints AFTER reset ---
    # (Because env.reset() wipes all debug lines)
    if not args.headless:
        wpm.redraw_waypoints()

    episode_reward = 0
    terminated, truncated = False, False
    history = []
    
    while not terminated and not truncated:
        if args.algo == "ppo":
            raw_action, log_prob, value = agent.get_action(state)
        else:
            raw_action = agent.get_action(state, exploration_noise=0.1)
            
        low, high = env.action_space.low, env.action_space.high
        action = low + (raw_action + 1.0) * 0.5 * (high - low)
        
        next_state, reward, terminated, truncated, info = env.step(action)
        
        # Log less frequently to save speed
        if env.current_step % 5 == 0:
            step_data = [env.current_step, reward, *state, *action]
            history.append(step_data)

        done_bool = float(terminated or truncated)
        if args.algo == "ppo":
            agent.store(state, raw_action, reward, done_bool, log_prob, value)
        else:
            agent.remember(state, raw_action, reward, done_bool, next_state)
            agent.learn() 

        state = next_state
        episode_reward += reward
        
    if args.algo == "ppo":
        agent.learn(0.0, terminated)
        
    return episode_reward, env.current_step, history

def main():
    args = parse_args()
    
    save_dir = os.path.join("training_runs", f"{args.run_tag}_{args.algo}")
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "args.json"), 'w') as f:
        json.dump(vars(args), f, indent=4)

    wpm = WaypointManager(visualize=not args.headless)

    # Dummy Env for dimensions
    dummy_env = PyBulletDroneEnv(
        waypoints_list=wpm.spawn_default_path(),
        control_mode=args.control_mode,
        use_rotation_matrix=args.use_rotation_matrix,
        gui=False
    )
    state_dim = dummy_env.observation_space.shape[0]
    print(f"Detected State Dimension: {state_dim} (Includes LiDAR)")
    dummy_env.close()

    agent = initialize_agent(args, state_dim, 4, 1.0)

    print(f"--- Starting {args.algo} Training (PyBullet) ---")
    
    rewards_window = deque(maxlen=100)
    log_cols = get_log_columns(args.use_rotation_matrix, num_rays=36)

    for ep in tqdm(range(args.episodes)):
        if ep > 100:
            waypoints = wpm.generate_random_walk_path()
        else:
            waypoints = wpm.spawn_default_path()
            
        env = PyBulletDroneEnv(
            waypoints_list=waypoints,
            control_mode=args.control_mode,
            use_rotation_matrix=args.use_rotation_matrix,
            gui=not args.headless
        )
        
        try:
            # --- FIX 3: Pass wpm here ---
            reward, length, history = run_episode(agent, env, args, wpm, ep)
            rewards_window.append(reward)
            
            if ep % 10 == 0:
                tqdm.write(f"Ep {ep} | R: {reward:.1f} | Avg100: {np.mean(rewards_window):.1f} | Steps: {length}")
            
            if ep % args.save_interval == 0:
                agent.save_models(save_dir)
                pd.DataFrame(history, columns=log_cols).to_csv(os.path.join(save_dir, f"ep_{ep}_log.csv"), index=False)
                
        finally:
            env.close()
            
    agent.save_models(save_dir)
    print("Done.")

if __name__ == "__main__":
    main()