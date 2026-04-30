import numpy as np
from matplotlib.patches import Circle
import config as cfg

class Sun:
    def __init__(self, ax):
        self.ax = ax
        # Simulation view
        self.position = (cfg.SUN_DISTANCE, 0)
        self.radius = cfg.SUN_RADIUS
        self.color = np.clip(cfg.SUN_COLOR, 0, 1)
        self.angle_deg = 0
        self.angle_rad = 0
        self.outer_sun = Circle(self.position, self.radius, color=self.color, zorder=20)
        self.ax.add_patch(self.outer_sun)

        # Line
        self.sampling_line = np.array([[0, 0], [0, 0]])
        self.line_pos = None
        self.ray_dir = np.array([1, 0])
        self.line_segments = cfg.NUM_LINE_POINTS

        # Canvas 
        self.canvas_y = None


    def set_position(self, position):
        self.position = position
        self.outer_sun.set_center(position)

    def rotate_deg(self, angle_deg):
        self.angle_deg = angle_deg
        self.angle_rad = np.deg2rad(angle_deg)
        self.position = np.array([cfg.SUN_DISTANCE * np.cos(self.angle_rad), cfg.SUN_DISTANCE * np.sin(self.angle_rad)])
        
        self.outer_sun.set_center(self.position)

    def set_color(self, color):
        self.color = np.clip(color, 0, 1)
        self.outer_sun.set_facecolor(self.color)


    def calculate_sun_metrics(self):
        """
        Calculates metrics based on the sun's angle relative to the midpoint of the sampling line.
        Returns a dictionary with normalized positions for the canvas and the line highlight, or None.
        """

        # Calculate the midpoint of the sampling line's start and end
        mid_point = (self.sampling_line[0] + self.sampling_line[-1]) / 2.0

        if np.linalg.norm(mid_point) == 0 or np.linalg.norm(self.position) == 0: return None

        vec_to_midpoint_norm = mid_point / np.linalg.norm(mid_point)
        vec_to_sun_norm = self.position / np.linalg.norm(self.position)

        cos_angle = np.dot(vec_to_midpoint_norm, vec_to_sun_norm)
        angle_deg = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
        cross_product = np.cross(vec_to_midpoint_norm, vec_to_sun_norm)
        
        degree = 120.0  # Max angle for visibility
        self.canvas_y = None
        self.line_pos = None
        if cross_product > 0 and 0 <= angle_deg <= degree:
            self.canvas_y = 1.0 - (angle_deg / degree)
            highlight_index = int(np.clip(self.canvas_y * (self.line_segments - 1), 0, self.line_segments - 1))
            self.line_pos = self.sampling_line[highlight_index]  # get the corresponding line point for highlighting


    

    def get_sampling_line_points(self, start_deg, end_deg, bend_factor, sampling_radius):
        start_rad, end_rad = np.deg2rad(start_deg), np.deg2rad(end_deg)
        start_angle, end_angle = np.pi + start_rad, np.pi + end_rad
        start_x, start_y = sampling_radius*np.cos(start_angle), sampling_radius*np.sin(start_angle)
        end_x, end_y = sampling_radius*np.cos(end_angle), sampling_radius*np.sin(end_angle)

        cx, cy = 0.0, 0.0; b_amt, b_dir = abs(bend_factor), 1 if bend_factor > 0 else -1
        dx, dy = end_x - start_x, end_y - start_y; length = np.sqrt(dx**2 + dy**2)
        px, py = (0.0, 1.0) if length < 1e-6 else (-dy/length, dx/length)
        b_dist = b_amt * cfg.ATMOSPHERE_RADIUS * b_dir
        cx += px*b_dist; cy += py*b_dist
        cr2 = cx**2 + cy**2
        if cr2 > cfg.ATMOSPHERE_RADIUS**2:
            scale = cfg.ATMOSPHERE_RADIUS / np.sqrt(cr2); cx *= scale; cy *= scale

        t_vals, points = np.linspace(0, 1, self.line_segments), []
        for t in t_vals:
            x = (1-t)**2*start_x + 2*(1-t)*t*cx + t**2*end_x
            y = (1-t)**2*start_y + 2*(1-t)*t*cy + t**2*end_y
            r2 = x*x + y*y
            if r2 > cfg.ATMOSPHERE_RADIUS**2: scale = cfg.ATMOSPHERE_RADIUS/np.sqrt(r2); x*=scale; y*=scale
            points.append([x, y])

        return np.array(points)

    
    def update_sampling_line(self, start_deg, end_deg, bend_factor, sampling_radius):
        """Update line and dot, returning line points and the highlight coordinate."""
        self.sampling_line = self.get_sampling_line_points(start_deg, end_deg, bend_factor, sampling_radius)
        
        self.calculate_sun_metrics()

