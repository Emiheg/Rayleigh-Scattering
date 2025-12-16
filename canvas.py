# canvas.py
"""
Defines the ColorCanvas class for visualizing the scattered light and
includes the core physics calculations for the simulation.
"""
import numpy as np
import config as cfg
from matplotlib.patches import Circle

# --- Core Calculation Functions ---
def get_ray_atmosphere_intersections(ray_origin, ray_direction, radius):
    a = np.dot(ray_direction, ray_direction); b = 2 * np.dot(ray_origin, ray_direction)
    c = np.dot(ray_origin, ray_origin) - radius**2; discriminant = b**2 - 4 * a * c
    if discriminant < 0: return None
    sqrt_d = np.sqrt(discriminant)
    t1, t2 = (-b - sqrt_d) / (2 * a), (-b + sqrt_d) / (2 * a)
    return ray_origin + t1 * ray_direction, ray_origin + t2 * ray_direction

def simulate_scattering(entry_point, exit_point, initial_color, max_enhancement, exp_coefficient, clip_max=1.0):
    path_vector = exit_point - entry_point
    total_path_length = np.linalg.norm(path_vector)

    if total_path_length == 0: return [], [], initial_color
    path_dir, seg_len = path_vector / total_path_length, total_path_length / cfg.NUM_SEGMENTS
    segs, trans_segs, color = [], [], np.copy(initial_color)

    for i in range(cfg.NUM_SEGMENTS):
        start = entry_point + path_dir * i * seg_len
        end = start + path_dir * seg_len
        scattered = color * cfg.SCATTERING_COEFFICIENTS * seg_len * cfg.SCATTERING_STRENGTH
        enh_factor = 1 + np.exp(exp_coefficient * scattered[2]) * max_enhancement
        scattered[0] *= enh_factor; scattered[1] *= enh_factor
        scattered_color = np.clip(scattered * 5, 0, clip_max)
        color = np.clip(color - scattered, 0, clip_max)

        segs.append((start, end, scattered_color))
        trans_segs.append((start, end, color.copy()))

    return segs, trans_segs, color

# --- Color Canvas Class ---
class ColorCanvas:
    def __init__(self, ax_scattered, ax_transmitted, ax_blended, atmosphere_radius, background_color, line_color):
        self.axes = {'scattered': ax_scattered, 'transmitted': ax_transmitted, 'blended': ax_blended}
        self.atmosphere_radius = atmosphere_radius
        self.background_color = background_color
        self.line_color = line_color
        self.sampling_line = None
        self.control_point = None
        self.highlight_dot = None
        self.num_points = 200

    # In canvas.py

    def calculate_sun_metrics(self, sun_pos, atmosphere_center, line_start_point, line_end_point):
        """
        Calculates metrics based on the sun's angle relative to the midpoint of the sampling line.
        Returns a dictionary with normalized positions for the canvas and the line highlight, or None.
        """
        # Calculate the midpoint of the sampling line's start and end
        mid_point = (line_start_point + line_end_point) / 2.0

        # The rest of the calculation now uses the midpoint
        vec_to_midpoint = mid_point - atmosphere_center
        vec_to_sun = sun_pos - atmosphere_center
        if np.linalg.norm(vec_to_midpoint) == 0 or np.linalg.norm(vec_to_sun) == 0: return None

        vec_to_midpoint_norm = vec_to_midpoint / np.linalg.norm(vec_to_midpoint)
        vec_to_sun_norm = vec_to_sun / np.linalg.norm(vec_to_sun)
        
        cos_angle = np.dot(vec_to_midpoint_norm, vec_to_sun_norm)
        angle_deg = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
        cross_product = np.cross(vec_to_midpoint_norm, vec_to_sun_norm)
        
        degree = 120.0  # Max angle for visibility

        if cross_product > 0 and 0 <= angle_deg <= degree:
            line_t = 1.0 - (angle_deg / degree)
            canvas_y = 1.0 - (angle_deg / degree)
            return {'canvas_y': canvas_y, 'line_t': line_t}
        return None

    def get_sampling_line_points(self, start_deg, end_deg, bend_factor, sampling_radius):
        start_rad, end_rad = np.deg2rad(start_deg), np.deg2rad(end_deg)
        start_angle, end_angle = np.pi + start_rad, np.pi + end_rad
        start_x, start_y = sampling_radius*np.cos(start_angle), sampling_radius*np.sin(start_angle)
        end_x, end_y = sampling_radius*np.cos(end_angle), sampling_radius*np.sin(end_angle)

        cx, cy = 0.0, 0.0; b_amt, b_dir = abs(bend_factor), 1 if bend_factor > 0 else -1
        dx, dy = end_x - start_x, end_y - start_y; length = np.sqrt(dx**2 + dy**2)
        px, py = (0.0, 1.0) if length < 1e-6 else (-dy/length, dx/length)
        b_dist = b_amt * self.atmosphere_radius * b_dir
        cx += px*b_dist; cy += py*b_dist
        cr2 = cx**2 + cy**2
        if cr2 > self.atmosphere_radius**2:
            scale = self.atmosphere_radius / np.sqrt(cr2); cx *= scale; cy *= scale
        t_vals, points = np.linspace(0, 1, self.num_points), []
        for t in t_vals:
            x = (1-t)**2*start_x + 2*(1-t)*t*cx + t**2*end_x
            y = (1-t)**2*start_y + 2*(1-t)*t*cy + t**2*end_y
            r2 = x*x + y*y
            if r2 > self.atmosphere_radius**2: scale = self.atmosphere_radius/np.sqrt(r2); x*=scale; y*=scale
            points.append([x, y])
        return np.array(points)
    
    def sample_colors_along_line(self, line_points, sun_pos, max_enhancement, exp_coefficient):
        all_s, all_t = [], []; n_rays = 15
        indices = np.linspace(0, len(line_points)-1, n_rays, dtype=int)
        for p in line_points[indices]:
            direction = p - sun_pos; direction /= np.linalg.norm(direction)
            intersections = get_ray_atmosphere_intersections(sun_pos, direction, self.atmosphere_radius)
            if intersections:
                entry, exit = intersections
                s_segs, t_segs, _ = simulate_scattering(entry, exit, cfg.SUN_COLOR, max_enhancement, exp_coefficient)
                all_s.extend(s_segs); all_t.extend(t_segs)
        return self._find_colors_on_path(line_points, all_s), self._find_colors_on_path(line_points, all_t)

    def _find_colors_on_path(self, line_points, segments_to_search):
        colors = []
        if not segments_to_search: return [cfg.SUN_COLOR] * len(line_points)
        for p in line_points:
            min_d, best_c = float('inf'), cfg.SUN_COLOR
            for s, e, c in segments_to_search:
                v, v_sq = e - s, np.dot(e-s, e-s)
                if v_sq == 0: continue
                t = np.clip(np.dot(p-s, v)/v_sq, 0, 1)
                d = np.linalg.norm(p - (s + t*v))
                if d < min_d: min_d, best_c = d, c
            colors.append(best_c)
        return colors

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
                for i, color in enumerate(colors):
                    y0, y1 = i * (height/len(colors)), (i+1)*(height/len(colors))
                    ax.fill_betweenx([y0, y1], 0, width, color=color, alpha=1.0, zorder=1)

            if sun_metrics is not None:
                y_coord = sun_metrics['canvas_y'] * height
                display_color = np.clip(sun_draw_color, 0, 1)
                sun_on_canvas = Circle((width * 0.5, y_coord), radius=10, color=display_color, zorder=10)
                ax.add_patch(sun_on_canvas)

            ax.axis('off') # Turn axis off at the end

    def update_sampling_line(self, ax_main, start_deg, end_deg, bend_factor, sampling_radius, sun_pos):
        """Update line and dot, returning line points and the highlight coordinate."""
        line_points = self.get_sampling_line_points(start_deg, end_deg, bend_factor, sampling_radius)
        if self.sampling_line is None:
            self.sampling_line, = ax_main.plot([], [], 'r-', lw=3, zorder=10, alpha=0.8)
        self.sampling_line.set_data(line_points[:, 0], line_points[:, 1])

        # ... (Control point logic is unchanged)
        start_rad, end_rad = np.deg2rad(start_deg), np.deg2rad(end_deg)
        s_ang, e_ang = np.pi+start_rad, np.pi+end_rad
        sx, sy = sampling_radius*np.cos(s_ang), sampling_radius*np.sin(s_ang)
        ex, ey = sampling_radius*np.cos(e_ang), sampling_radius*np.sin(e_ang)
        cx,cy=0,0; dx,dy=ex-sx,ey-sy; length=np.sqrt(dx**2+dy**2)
        px,py = (0,1) if length < 1e-6 else (-dy/length, dx/length)
        cx+=px*abs(bend_factor)*self.atmosphere_radius*(1 if bend_factor > 0 else -1)
        cy+=py*abs(bend_factor)*self.atmosphere_radius*(1 if bend_factor > 0 else -1)
        if cx**2+cy**2 > self.atmosphere_radius**2:
            scale=self.atmosphere_radius/np.sqrt(cx**2+cy**2); cx*=scale; cy*=scale
        if self.control_point is None:
            self.control_point, = ax_main.plot([], [], 'ro', ms=8, zorder=12, alpha=0.7)
        self.control_point.set_data([cx], [cy])

        sun_metrics = self.calculate_sun_metrics(sun_pos, np.array([0,0]), line_points[0], line_points[-1])
        highlight_coord = None
        if sun_metrics is not None:
            index_pos = sun_metrics['line_t']
            highlight_index = int(np.clip(index_pos * (self.num_points - 1), 0, self.num_points - 1))
            highlight_coord = line_points[highlight_index]

        if self.highlight_dot is None:
            self.highlight_dot, = ax_main.plot([],[], 'o', ms=8, color='white', mec='black', mew=0.8, zorder=13)
        
        if highlight_coord is not None:
            self.highlight_dot.set_data([highlight_coord[0]], [highlight_coord[1]])
        else:
            self.highlight_dot.set_data([np.nan], [np.nan])
            
        return line_points, highlight_coord