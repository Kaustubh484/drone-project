import os
import sys
import json
import torch
import argparse
import numpy as np
from pybullet_env import PyBulletDroneEnv
from waypoint_manager import WaypointManager
from utils import initialize_agent

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--load_dir", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--headless", action="store_true")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Load Config
    with open(os.path.join(args.load_dir, "args.json"), 'r') as f:
        train_args_dict = json.load(f)
    train_args = argparse.Namespace(**train_args_dict)
    
    # Init Agent
    state_dim = 18 if train_args.use_rotation_matrix else 13
    agent = initialize_agent(train_args, state_dim, 4, 1.0)
    agent.load_models(args.load_dir)
    
    wpm = WaypointManager(visualize=not args.headless)
    
    for i in range(args.episodes):
        print(f"--- Test Episode {i+1} ---")
        waypoints = wpm.spawn_default_path() # Or random
        
        env = PyBulletDroneEnv(
            waypoints_list=waypoints,
            control_mode=train_args.control_mode,
            use_rotation_matrix=train_args.use_rotation_matrix,
            gui=not args.headless
        )
        
        state, _ = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            action_raw = agent.get_deterministic_action(state)
            
            # Scale
            low, high = env.action_space.low, env.action_space.high
            action = low + (action_raw + 1.0) * 0.5 * (high - low)
            
            state, r, term, trunc, _ = env.step(action)
            total_reward += r
            done = term or trunc
            
        print(f"Result: Reward {total_reward:.2f}")
        env.close()

if __name__ == "__main__":
    main()