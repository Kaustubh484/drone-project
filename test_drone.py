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
    parser.add_argument("--load_dir", type=str, required=True, help="Path to model directory")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--headless", action="store_true")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 1. Load Training Arguments
    args_path = os.path.join(args.load_dir, "args.json")
    if not os.path.exists(args_path):
        print(f"Error: args.json not found in {args.load_dir}")
        return

    with open(args_path, 'r') as f:
        train_args_dict = json.load(f)
    train_args = argparse.Namespace(**train_args_dict)
    
    # 2. Initialize Waypoint Manager
    wpm = WaypointManager(visualize=not args.headless)

    # 3. FIX: Detect correct state dimension (Kinematics + LiDAR)
    # We create a temporary env just to read the observation space size
    print("Detecting state dimension...")
    dummy_env = PyBulletDroneEnv(
        waypoints_list=wpm.spawn_default_path(),
        control_mode=train_args.control_mode,
        use_rotation_matrix=train_args.use_rotation_matrix,
        gui=False
    )
    state_dim = dummy_env.observation_space.shape[0]
    print(f"State Dimension: {state_dim}")
    dummy_env.close()
    
    # 4. Initialize and Load Agent
    agent = initialize_agent(train_args, state_dim, 4, 1.0)
    try:
        agent.load_models(args.load_dir)
        print(f"Successfully loaded models from {args.load_dir}")
    except Exception as e:
        print(f"Error loading models: {e}")
        return
    
    # 5. Run Test Loop
    for i in range(args.episodes):
        print(f"\n--- Test Episode {i+1} ---")
        waypoints = wpm.spawn_default_path() # Or generate_random_walk_path()
        
        env = PyBulletDroneEnv(
            waypoints_list=waypoints,
            control_mode=train_args.control_mode,
            use_rotation_matrix=train_args.use_rotation_matrix,
            gui=not args.headless
        )
        
        state, _ = env.reset()
        
        # FIX: Redraw waypoints because reset() clears debug lines
        if not args.headless:
            wpm.redraw_waypoints()

        total_reward = 0
        done = False
        
        while not done:
            # Get deterministic action for testing (no noise)
            action_raw = agent.get_deterministic_action(state)
            
            # Scale Action
            low, high = env.action_space.low, env.action_space.high
            action = low + (action_raw + 1.0) * 0.5 * (high - low)
            
            # Step
            state, r, term, trunc, _ = env.step(action)
            total_reward += r
            done = term or trunc
            
        print(f"Result: Reward {total_reward:.2f}")
        env.close()

if __name__ == "__main__":
    main()