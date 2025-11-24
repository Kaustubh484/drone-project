import numpy as np
import gymnasium
from gymnasium import spaces
import pybullet as p
import pybullet_data
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.utils.enums import DroneModel, Physics

class PyBulletDroneEnv(gymnasium.Env):
    def __init__(self, waypoints_list, 
                 waypoint_threshold=0.5, # Stricter threshold for precision
                 waypoint_bonus=100.0, 
                 crash_penalty=100.0, 
                 timeout_penalty=50.0, 
                 per_step_penalty=0.0, # FIXED: Set to 0 to prevent suicide behavior
                 spawn_range_xy=0.5, # Reduced spawn noise for easier learning
                 max_steps=2000, control_mode="velocity", 
                 use_rotation_matrix=False, gui=False):
        
        super().__init__()
        
        self.waypoints_list = np.array(waypoints_list)
        self.waypoint_threshold = waypoint_threshold
        self.waypoint_bonus = waypoint_bonus
        self.crash_penalty = crash_penalty
        self.timeout_penalty = timeout_penalty
        self.per_step_penalty = per_step_penalty
        self.spawn_range_xy = spawn_range_xy
        self.max_steps = max_steps
        self.control_mode = control_mode
        self.use_rotation_matrix = use_rotation_matrix
        self.gui = gui

        self.env = CtrlAviary(
            drone_model=DroneModel.CF2X,
            num_drones=1,
            neighbourhood_radius=10,
            physics=Physics.PYB,
            gui=self.gui
        )
        
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        
        self.ctrl = DSLPIDControl(drone_model=DroneModel.CF2X)
        
        self.num_rays = 36
        self.lidar_range = 3.0 

        kinematics_dim = 18 if self.use_rotation_matrix else 13
        obs_dim = kinematics_dim + self.num_rays 
        
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

        if self.control_mode == "velocity":
            low = np.array([-5.0, -5.0, -1.0, -45.0]) 
            high = np.array([5.0, 5.0, 1.0, 45.0])
            self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)
        else:
            raise NotImplementedError("Only 'velocity' mode is currently implemented.")

        self.current_step = 0
        self.current_waypoint_index = 0
        self.target_position = self.waypoints_list[0]
        self.prev_distance = 0.0
        self.current_distance = 0.0

    def _get_lidar_reading(self):
        pos, quat = self.env.pos[0], self.env.quat[0]
        rot_mat = np.array(p.getMatrixFromQuaternion(quat)).reshape(3, 3)
        
        ray_from = []
        ray_to = []
        
        for i in range(self.num_rays):
            angle = 2 * np.pi * i / self.num_rays
            ray_dir_body = np.array([np.cos(angle), np.sin(angle), 0])
            ray_dir_world = rot_mat @ ray_dir_body
            
            start = pos
            end = pos + ray_dir_world * self.lidar_range
            
            ray_from.append(start)
            ray_to.append(end)
            
        results = p.rayTestBatch(ray_from, ray_to, physicsClientId=self.env.CLIENT)
        lidar_dist = [res[2] * self.lidar_range for res in results]
        return np.array(lidar_dist, dtype=np.float32)

    def _get_observation(self, obs_input):
        state_vec = None
        if isinstance(obs_input, dict):
            val = None
            if 0 in obs_input: val = obs_input[0]
            elif "0" in obs_input: val = obs_input["0"]
            else: val = next(iter(obs_input.values()))
            
            if isinstance(val, dict) and "state" in val:
                state_vec = val["state"]
            else:
                state_vec = val
        elif isinstance(obs_input, (list, tuple)):
            if len(obs_input) > 0:
                state_vec = obs_input[0]
                if isinstance(state_vec, dict) and "state" in state_vec:
                    state_vec = state_vec["state"]
            else:
                state_vec = np.zeros(20)
        elif isinstance(obs_input, np.ndarray):
            state_vec = obs_input
            
        state_vec = np.array(state_vec)
        if state_vec.ndim > 1:
            state_vec = state_vec.flatten()
            
        pos = state_vec[0:3]
        quat = state_vec[3:7]
        quat_wxyz = np.array([quat[3], quat[0], quat[1], quat[2]])
        
        vel = state_vec[10:13]
        ang_vel = state_vec[13:16]

        current_pos = pos
        rel_pos = self.target_position - current_pos
        self.current_distance = np.linalg.norm(rel_pos)

        if self.use_rotation_matrix:
            rot_mat = np.array(p.getMatrixFromQuaternion(quat)).reshape(9)
            orientation_feature = rot_mat
        else:
            orientation_feature = quat_wxyz

        kinematic_state = np.concatenate([
            rel_pos,
            orientation_feature,
            vel,
            ang_vel
        ])
        
        lidar_state = self._get_lidar_reading()
        return np.concatenate([kinematic_state, lidar_state]).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.env.reset()
        
        # Spawn very close to 0,0,0 to make starting easier
        x = np.random.uniform(-0.2, 0.2)
        y = np.random.uniform(-0.2, 0.2)
        z = 0.1 
        init_pos = np.array([x, y, z])
        init_quat = np.array([0, 0, 0, 1]) 
        
        drone_id = self.env.DRONE_IDS[0]
        p.resetBasePositionAndOrientation(drone_id, init_pos, init_quat, physicsClientId=self.env.CLIENT)
        self.env.pos[0] = init_pos
        self.env.quat[0] = init_quat

        # Spawn Obstacles (Offset so they don't block the easy path immediately)
        for _ in range(3):
            obs_x = np.random.uniform(-5, 5)
            obs_y = np.random.uniform(-5, 5)
            if np.linalg.norm([obs_x, obs_y]) > 2.0: # Keep center clear
                p.loadURDF("cube.urdf", [obs_x, obs_y, 1.0], globalScaling=0.5, physicsClientId=self.env.CLIENT)

        self.current_step = 0
        self.current_waypoint_index = 0
        self.target_position = self.waypoints_list[self.current_waypoint_index]
        
        obs_dict = self.env._computeObs()
        state = self._get_observation(obs_dict)
        self.prev_distance = self.current_distance
        
        return state, {}

    def step(self, action):
        self.current_step += 1
        target_vel = action[:3]
        
        # Get state for control
        obs_raw = self.env._computeObs()
        # Safe extraction for control logic
        if isinstance(obs_raw, dict):
            key = 0 if 0 in obs_raw else "0"
            val = obs_raw[key]
            state_vec = val["state"] if (isinstance(val, dict) and "state" in val) else val
        elif isinstance(obs_raw, (list, tuple)):
            state_vec = obs_raw[0]
            if isinstance(state_vec, dict) and "state" in state_vec: state_vec = state_vec["state"]
        else:
            state_vec = obs_raw
        
        state_vec = np.array(state_vec)
        if state_vec.ndim > 1: state_vec = state_vec.flatten()
        
        cur_pos = state_vec[0:3]
        cur_quat = state_vec[3:7]
        cur_vel = state_vec[10:13]
        cur_ang_vel = state_vec[13:16]
        
        dt = 1.0 / 240.0 
        target_pos = cur_pos + (target_vel * 0.1) 
        
        rpm, _, _ = self.ctrl.computeControl(
            control_timestep=dt,
            cur_pos=cur_pos,
            cur_quat=cur_quat,
            cur_vel=cur_vel,
            cur_ang_vel=cur_ang_vel,
            target_pos=target_pos,
            target_rpy=np.array([0,0,0]),
            target_vel=target_vel
        )
        rpm = rpm.reshape(1, 4)

        obs_output, _, terminated, truncated, info = self.env.step(rpm)
        
        next_state = self._get_observation(obs_output)
        lidar_data = next_state[-self.num_rays:] 

        # --- NEW REWARD FUNCTION ---
        reward = 0
        
        # 1. Progress Reward (Velocity towards target)
        # Scale up significantly so it dominates noise
        dist_progress = (self.prev_distance - self.current_distance) * 50.0 
        
        # 2. Stability Penalty (Slightly reduced)
        instability_penalty = np.linalg.norm(state_vec[13:16]) * 0.02
        
        # 3. Action Smoothness (Penalize jerky control)
        action_penalty = np.linalg.norm(action) * 0.01
        
        # 4. Survival Reward (Encourage staying in the air)
        survival_reward = 0.1
        
        reward = dist_progress + survival_reward - instability_penalty - action_penalty
        
        self.prev_distance = self.current_distance
        
        info["reward_dist"] = dist_progress
        info["reward_survive"] = survival_reward

        # Waypoints
        if self.current_distance < self.waypoint_threshold:
            reward += self.waypoint_bonus
            self.current_waypoint_index += 1
            
            if self.current_waypoint_index >= len(self.waypoints_list):
                terminated = True
                print("--- Sequence Complete! ---")
            else:
                self.target_position = self.waypoints_list[self.current_waypoint_index]
                print(f"--- Reached Waypoint {self.current_waypoint_index}! ---")

        # Crash / OOB
        z_alt = state_vec[2]
        if z_alt < 0.05 or z_alt > 10.0 or abs(state_vec[0]) > 20 or abs(state_vec[1]) > 20:
            terminated = True
            reward -= self.crash_penalty
            info["reward_crash"] = -self.crash_penalty
            
        if self.current_step >= self.max_steps:
            truncated = True
            reward -= self.timeout_penalty

        if np.min(lidar_data) < 0.2: 
            reward -= self.crash_penalty
            terminated = True
            info["reward_crash"] = -self.crash_penalty
            
        return next_state, reward, terminated, truncated, info

    def close(self):
        self.env.close()