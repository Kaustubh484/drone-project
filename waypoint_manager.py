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
        """Creates a very simple straight line path for initial training"""
        self.clear_waypoints()
        # Start (near 0,0,0)
        self.add_waypoint(0, 0, 1, color="green")
        # Target 1 (Forward 2m)
        self.add_waypoint(2, 0, 1, color="red")
        # Target 2 (Forward 4m)
        self.add_waypoint(4, 0, 1, color="blue")
        return self.get_waypoints()

    def spawn_default_path(self):
        # Fallback to simple static for now
        return self.spawn_simple_static_path()

    def generate_random_walk_path(self, num_waypoints=6, max_step_dist=5.0):
        # Fallback to simple static for now to force learning
        return self.spawn_simple_static_path()
    
    def spawn_from_file(self, file_path):
        try:
            self.clear_waypoints()
            wps = np.load(file_path)
            for i, wp in enumerate(wps):
                color = "green" if i == 0 else "red"
                self.add_waypoint(wp[0], wp[1], wp[2], color=color)
            return self.get_waypoints()
        except Exception:
            return self.spawn_simple_static_path()