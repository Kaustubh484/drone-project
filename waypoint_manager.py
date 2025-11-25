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
        if self.visualize and p.isConnected():
            try:
                p.removeAllUserDebugItems()
            except p.error:
                pass
        self.debug_items = []

    def add_waypoint(self, x, y, z, color=None):
        waypoint = np.array([x, y, z])
        self.waypoints.append(waypoint)
        
        if self.visualize and p.isConnected():
            self._draw_single_waypoint(x, y, z, color, len(self.waypoints))

        return waypoint
    
    def _draw_single_waypoint(self, x, y, z, color, idx):
        try:
            size = 0.2
            c = [1, 0, 0] if color == "red" else ([0, 1, 0] if color == "green" else [0, 0, 1])
            line1 = p.addUserDebugLine([x-size, y, z], [x+size, y, z], lineColorRGB=c, lineWidth=2)
            line2 = p.addUserDebugLine([x, y-size, z], [x, y+size, z], lineColorRGB=c, lineWidth=2)
            line3 = p.addUserDebugLine([x, y, z-size], [x, y, z+size], lineColorRGB=c, lineWidth=2)
            self.debug_items.extend([line1, line2, line3])
            label_id = p.addUserDebugText(str(idx), [x, y, z+0.3], textColorRGB=[0,0,0])
            self.debug_items.append(label_id)
        except p.error:
            pass

    def redraw_waypoints(self):
        if not self.visualize or not p.isConnected():
            return
        self.debug_items = [] 
        for i, wp in enumerate(self.waypoints):
            color = "green" if i == 0 else ("blue" if i == len(self.waypoints)-1 else "red")
            self._draw_single_waypoint(wp[0], wp[1], wp[2], color, i+1)
    
    def get_waypoints(self):
        return np.array(self.waypoints)

    def spawn_simple_static_path(self):
        """Level 1: Straight Line"""
        self.clear_waypoints()
        self.add_waypoint(0, 0, 1, color="green") # Start
        self.add_waypoint(2, 0, 1, color="red")   # Mid
        self.add_waypoint(4, 0, 1, color="blue")  # End
        return self.get_waypoints()

    def spawn_default_path(self):
        """Level 2: Square Path (with altitude change)"""
        self.clear_waypoints()
        self.add_waypoint(0, 0, 1, color="green") # Start at 1m
        self.add_waypoint(2, 0, 2, color="red")   # Forward & Up to 2m
        self.add_waypoint(2, 2, 2, color="red")   # Left
        self.add_waypoint(0, 2, 2, color="red")   # Back
        self.add_waypoint(0, 0, 1, color="blue")  # Home & Down to 1m
        return self.get_waypoints()
    
    def spawn_simple_path(self):
        """Level 2: Square Path (with altitude change)"""
        self.clear_waypoints()
        self.add_waypoint(0, 0, 1, color="green") # Start at 1m
        self.add_waypoint(2, 0, 1, color="red")   # Forward & Up to 2m
        self.add_waypoint(2, 2, 1, color="red")   # Left
        self.add_waypoint(0, 2, 1, color="red")   # Back
        self.add_waypoint(0, 0, 1, color="blue")  # Home & Down to 1m
        return self.get_waypoints()

    def generate_random_walk_path(self, num_waypoints=6, max_step_dist=5.0):
        # Fallback to square for now until we want Level 3
        return self.spawn_default_path()
    
    def spawn_from_file(self, file_path):
        try:
            self.clear_waypoints()
            wps = np.load(file_path)
            for i, wp in enumerate(wps):
                color = "green" if i == 0 else "red"
                self.add_waypoint(wp[0], wp[1], wp[2], color=color)
            return self.get_waypoints()
        except Exception:
            return self.spawn_default_path()