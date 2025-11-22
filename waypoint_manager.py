import numpy as np
import pybullet as p

class WaypointManager:
    def __init__(self, visualize=False):
        self.waypoints = []
        self.visualize = visualize
        self.debug_items = []
        print(f"--- WaypointManager initialized (Visualize: {self.visualize}) ---")

    def set_visualize(self, visualize):
        self.visualize = visualize

    def clear_waypoints(self):
        self.waypoints = []
        # Safe clear that checks connection
        if self.visualize and p.isConnected():
            try:
                p.removeAllUserDebugItems()
            except p.error:
                pass
        self.debug_items = []

    def add_waypoint(self, x, y, z, color=None):
        # Store as XYZ
        waypoint = np.array([x, y, z])
        self.waypoints.append(waypoint)
        
        # Only try to draw if we are actually connected to a simulation
        if self.visualize and p.isConnected():
            self._draw_single_waypoint(x, y, z, color, len(self.waypoints))

        return waypoint
    
    def _draw_single_waypoint(self, x, y, z, color, idx):
        try:
            size = 0.2
            # RGB list [R, G, B]
            c = [1, 0, 0] if color == "red" else ([0, 1, 0] if color == "green" else [0, 0, 1])
            
            line1 = p.addUserDebugLine([x-size, y, z], [x+size, y, z], lineColorRGB=c, lineWidth=2)
            line2 = p.addUserDebugLine([x, y-size, z], [x, y+size, z], lineColorRGB=c, lineWidth=2)
            line3 = p.addUserDebugLine([x, y, z-size], [x, y, z+size], lineColorRGB=c, lineWidth=2)
            
            self.debug_items.extend([line1, line2, line3])
            
            # Add Text Label
            label_id = p.addUserDebugText(str(idx), [x, y, z+0.3], textColorRGB=[0,0,0])
            self.debug_items.append(label_id)
        except p.error:
            # Pass if connection dropped or weird state
            pass

    def redraw_waypoints(self):
        """Call this AFTER the environment is initialized to visualize points"""
        if not self.visualize or not p.isConnected():
            return
            
        # Clear old IDs just in case
        self.debug_items = [] 
        
        for i, wp in enumerate(self.waypoints):
            # Re-infer color logic roughly
            color = "green" if i == 0 else ("blue" if i == len(self.waypoints)-1 else "red")
            self._draw_single_waypoint(wp[0], wp[1], wp[2], color, i+1)
    
    def get_waypoints(self):
        return np.array(self.waypoints)

    # --- Generators ---

    def generate_random_walk_path(self, num_waypoints=6, max_step_dist=5.0):
        self.clear_waypoints()
        last_wp = np.array([0.0, 0.0, 1.0]) # Start at 1m height
        self.add_waypoint(*last_wp, color="green")
        
        for i in range(1, num_waypoints):
            angle = np.random.uniform(0, 2 * np.pi)
            dist = np.random.uniform(max_step_dist / 2, max_step_dist)
            
            new_x = last_wp[0] + dist * np.cos(angle)
            new_y = last_wp[1] + dist * np.sin(angle)
            new_z = np.clip(last_wp[2] + np.random.uniform(-1, 1), 1.0, 5.0)
            
            color = "red" if i < num_waypoints - 1 else "blue"
            self.add_waypoint(new_x, new_y, new_z, color=color)
            last_wp = np.array([new_x, new_y, new_z])
            
        return self.get_waypoints()

    def spawn_default_path(self):
        self.clear_waypoints()
        # Simple Square at 2m height
        self.add_waypoint(0, 0, 1, color="green")
        self.add_waypoint(2, 0, 2, color="red")
        self.add_waypoint(2, 2, 2, color="red")
        self.add_waypoint(0, 2, 2, color="red")
        self.add_waypoint(0, 0, 1, color="blue")
        return self.get_waypoints()

    def spawn_from_file(self, file_path):
        try:
            self.clear_waypoints()
            wps = np.load(file_path)
            for i, wp in enumerate(wps):
                color = "green" if i == 0 else "red"
                self.add_waypoint(wp[0], wp[1], wp[2], color=color)
            return self.get_waypoints()
        except Exception as e:
            print(f"Failed to load waypoints: {e}")
            return self.spawn_default_path()