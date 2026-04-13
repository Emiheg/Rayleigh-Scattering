# canvas.py
"""
Defines the ColorCanvas class for visualizing the scattered light and
includes the core physics calculations for the simulation.
"""
import numpy as np
import config as cfg
import SunClass
from matplotlib.patches import Circle

# --- Core Calculation Functions ---
def get_ray_atmosphere_intersections(ray_origin, ray_direction, radius):
    a = np.dot(ray_direction, ray_direction); b = 2 * np.dot(ray_origin, ray_direction)
    c = np.dot(ray_origin, ray_origin) - radius**2; discriminant = b**2 - 4 * a * c
    if discriminant < 0: return None
    sqrt_d = np.sqrt(discriminant)
    t1, t2 = (-b - sqrt_d) / (2 * a), (-b + sqrt_d) / (2 * a)
    return ray_origin + t1 * ray_direction, ray_origin + t2 * ray_direction


def get_closest_half_circle(center_angle_rad, radius, num_points=cfg.NUM_RAYS):
    # 1. Define the limits first (Rotation is handled here)
    # This 'rotates' the entire arc by shifting the start/end parameters
    start = center_angle_rad - np.pi/2
    end = center_angle_rad + np.pi/2

    # 2. Generate the span directly in the correct orientation
    angles = np.linspace(start, end, num_points)
    length = np.linspace(radius, radius, num_points)
    # 3. Single-pass coordinate calculation
    # No further rotation needed
    
    x = radius * np.cos(angles)
    y = radius * np.sin(angles)

    end_vector = radius * np.cos(angles-center_angle_rad) * 2

    return (x, y, end_vector)


def get_perpendicular_from_origin(p2, distance=1.0):
    """
    Calculates a point 90 degrees from p2, assuming p1 is always at (0,0).
    """
    x2, y2 = p2
    
    # Calculate length from origin (Pythagoras)
    length = cfg.SUN_DISTANCE
    
    if length == 0:
        return (0.0, 0.0)
    
    # Rotate and normalize directly using p2's coordinates
    cw_dx = (y2 / length) * distance
    cw_dy = (-x2 / length) * distance
        
    nw_dx = (-y2 / length) * distance
    nw_dy = (x2 / length) * distance
        
    return ([x2 + cw_dx, y2 + cw_dy], [x2 + nw_dx, y2 + nw_dy])


def simulate_scattering_middle(max_enhancement, exp_coefficient = cfg.EXPONENTIAL_COEFFICIENT, color=cfg.SUN_COLOR, clip_max=1):
    scattered_light, direct_light = [], []

    for _ in range(cfg.NUM_SEGMENTS + 1):
        scattered = color * cfg.SCATTERING_COEFFICIENTS * cfg.SEGMENT_LENGTH * cfg.SCATTERING_STRENGTH
        enh_factor = 1 + np.exp(exp_coefficient * scattered[2]) * max_enhancement
        scattered[0] *= enh_factor; scattered[1] *= enh_factor
        scattered_color = scattered * 5
        color = color - scattered

        scattered_light.append(np.clip(scattered_color,0,clip_max))
        direct_light.append(np.clip(color.copy(),0,clip_max))
    
    scattered_light = np.clip(scattered_light, 0, clip_max)
    direct_light = np.clip(direct_light, 0, clip_max)

    return scattered_light, direct_light

def sample_colors_along_line(line_points, sun_rad):
    #Measures the distance the ray has travelled through the atmosphere and returns the segment number of the color
    x_coords, y_coords = line_points[:, 0], line_points[:, 1]
    distance, _, _  = calculate_distance_to_edge(x_coords, y_coords, sun_rad)


    return (distance/cfg.SEGMENT_LENGTH).astype(int)

def calculate_distance_to_edge(x0, y0, angle_rad, radius=cfg.ATMOSPHERE_RADIUS):
    """
    Calculates the distance 'd' from a point (x0, y0) to the circle's edge
    along a specific angle. Works with both single values and NumPy arrays.
    """
    # 1. create a direction unit vector
    vx = np.cos(angle_rad)
    vy = np.sin(angle_rad)
    
    # 2. Calculate Dot Product (P · v) and squared distance from origin (|P|²)
    # This identifies the point's position relative to the center
    
    dot_pv = x0 * vx + y0 * vy
    p_sq = x0**2 + y0**2
    
    # 3. Use the quadratic formula to find the distance 'd'
    # Equation: d^2 + 2d(P·v) + (|P|^2 - R^2) = 0
    discriminant = dot_pv**2 - (p_sq - radius**2)
    
    # Safety check: if the point is outside or moving away, discriminant could be < 0
    # np.maximum handles potential tiny floating-point errors
    d = -dot_pv + np.sqrt(np.maximum(discriminant, 0))
    
    # 4. Calculate the exact exit coordinates on the edge
    x_exit = x0 + d * vx
    y_exit = y0 + d * vy
    
    return d, x_exit, y_exit

# --- Color Canvas Class ---
class ColorCanvas:
    def __init__(self, ax_scattered, ax_transmitted, ax_blended, background_color, line_color):
        self.axes = {'scattered': ax_scattered, 'transmitted': ax_transmitted, 'blended': ax_blended}
        self.background_color = background_color
        self.line_color = line_color
        self.num_points = cfg.NUM_RAYS

    


    def update_canvases(self, scattered_colors, transmitted_colors, blended_colors, sun_metrics=None, sun_draw_color=cfg.SUN_COLOR):
        """Update canvases and draw a circular sun."""
        titles = {'scattered':'Scattered', 'transmitted':'Transmitted', 'blended':'Blended'}
        colors_map = {'scattered': scattered_colors, 'transmitted': transmitted_colors, 'blended': blended_colors}

        width, height = 1000, 500

        for key, ax in self.axes.items():
            ax.clear()
            ax.set_facecolor(self.background_color)
            ax.set_title(titles[key], color=self.line_color, fontsize=10)
            ax.set_xlim(0, width); ax.set_ylim(0, height)
            
            if key == 'blended':
                ax.set_aspect('equal', adjustable='box')


            colors = colors_map[key]
            if colors:

                color_array = np.array(colors)          # Reshape the list of blended colors into a 2D array for imshow
                img_data = color_array.reshape((len(color_array), 1, 3)) # Reshape to (num_colors, 1, 3) for imshow

                img = ax.imshow(img_data, 
                                aspect='equal', 
                                extent=[0, width, 0, height], 
                                origin='lower',
                                animated=True)



            if sun_metrics is not None:
                y_coord = sun_metrics * height
                display_color = np.clip(sun_draw_color, 0, 1)
                sun_on_canvas = Circle((width * 0.5, y_coord), radius=10, color=display_color, zorder=10)
                ax.add_patch(sun_on_canvas)

            ax.axis('off') # Turn axis off at the end
