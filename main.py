# main.py
"""
Main script to run the atmospheric scattering simulation.
This file sets up the plot, widgets, and the main update loop.
To run: python main.py
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.widgets import Slider
import matplotlib.gridspec as gridspec

import config as cfg
from canvas import ColorCanvas, get_ray_atmosphere_intersections, simulate_scattering

# --- Setup Windows & Plots ---
background_color, line_color = '#1c1c1c', '#f0f0f0'

# --- Window 1: Main Simulation ---
fig = plt.figure(figsize=(12, 8))
fig.canvas.manager.set_window_title('Scattering Simulation Controls')
fig.set_facecolor(background_color)
gs_main = gridspec.GridSpec(2, 2, width_ratios=[4, 1], height_ratios=[3, 1])
ax = fig.add_subplot(gs_main[0, 0])
ax.set_aspect('equal')
ax.set_facecolor(background_color)

# --- Window 2: Blended Result ---
fig_blended = plt.figure(figsize=(3, 3.5)) # A smaller, squarish window
fig_blended.canvas.manager.set_window_title('Blended Result')
fig_blended.set_facecolor(background_color)
ax_blended_new = fig_blended.add_subplot(111) # A single axis for this figure

# --- Setup Canvases in Main Window (now only 2) ---
gs_side_canvases = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs_main[0, 1], hspace=0.6)
ax_scattered = fig.add_subplot(gs_side_canvases[0, 0])
ax_transmitted = fig.add_subplot(gs_side_canvases[1, 0])

# --- Draw Static Elements ---
ax.add_patch(Circle((0,0), cfg.ATMOSPHERE_RADIUS, fill=False, edgecolor=line_color, ls='--', lw=1.5, zorder=5))
ax.add_patch(Circle((0,0), cfg.SUN_DISTANCE, fill=False, edgecolor='gray', ls=':', lw=1, zorder=1))
viewer_dot, = ax.plot(0, 0, 'o', ms=10, color=line_color, zorder=11)

# --- Initialize Dynamic Elements ---
sun = Circle((0,0), cfg.SUN_RADIUS, facecolor=line_color, edgecolor=line_color, lw=1.5, zorder=15)
ax.add_patch(sun)
ray_plots = []
color_canvas = ColorCanvas(ax_scattered, ax_transmitted, ax_blended_new, cfg.ATMOSPHERE_RADIUS, background_color, line_color)

# --- Create Sliders ---
left, right, slider_h, v_start = 0.1, 0.6, 0.025, 0.20
sliders_def = [
    ('Sun Angle (°)', [right, v_start, 0.32, slider_h], {'valmin':0, 'valmax':360, 'valinit':0.0}),
    ('Max Enhancement', [right, v_start-0.03, 0.32, slider_h], {'valmin':0.1, 'valmax':cfg.MAX_COLOR_ENHANCEMENT, 'valinit':33}),
    ('Exp. Coefficient', [right, v_start-0.06, 0.32, slider_h], {'valmin':-200, 'valmax':0, 'valinit':cfg.EXPONENTIAL_COEFFICIENT}),
    ('Start Point', [left, v_start, 0.32, slider_h], {'valmin':0, 'valmax':180, 'valinit':45}),
    ('End Point', [left, v_start-0.03, 0.32, slider_h], {'valmin':0, 'valmax':180, 'valinit':135}),
    ('Line Bend', [left, v_start-0.06, 0.32, slider_h], {'valmin':-1, 'valmax':1, 'valinit':-1}),
    ('Sampling Radius', [left, v_start-0.09, 0.32, slider_h], {'valmin':0.1, 'valmax':cfg.ATMOSPHERE_RADIUS, 'valinit':cfg.ATMOSPHERE_RADIUS}),
]
sliders = {}
for label, position, params in sliders_def:
    slider_ax = fig.add_axes(position)
    sliders[label] = Slider(ax=slider_ax, label=label, **params, color='#444444')
    sliders[label].label.set_color(line_color)
    sliders[label].on_changed(lambda val: update(None))

# --- Define the Update Function ---
def update(val):
    global ray_plots
    [line.remove() for line in ray_plots]; ray_plots.clear()

    sun_angle_deg = sliders['Sun Angle (°)'].val
    sun_angle_rad = np.deg2rad(sun_angle_deg)
    sun_pos = np.array([cfg.SUN_DISTANCE * np.cos(sun_angle_rad), cfg.SUN_DISTANCE * np.sin(sun_angle_rad)])
    sun.set_center(sun_pos)
    
    line_points, highlight_coord = color_canvas.update_sampling_line(
        ax, sliders['Start Point'].val, sliders['End Point'].val, sliders['Line Bend'].val, sliders['Sampling Radius'].val, sun_pos
    )
    
    sun_metrics = color_canvas.calculate_sun_metrics(sun_pos, np.array([0,0]), line_points[0], line_points[-1])
    sun_draw_color = cfg.SUN_COLOR
    
    ray_direction = -sun_pos / np.linalg.norm(sun_pos)
    perp_dir = np.array([-ray_direction[1], ray_direction[0]])

    # --- Special "God Ray" Simulation ---
    if highlight_coord is not None:
        offset = np.dot(highlight_coord - sun_pos, perp_dir)
        god_ray_origin = sun_pos + perp_dir * offset
        intersections = get_ray_atmosphere_intersections(god_ray_origin, ray_direction, cfg.ATMOSPHERE_RADIUS)
        if intersections:
            entry, exit = intersections
            _, transmitted_segs, _ = simulate_scattering(
                entry, exit, cfg.GODRAY , sliders['Max Enhancement'].val, sliders['Exp. Coefficient'].val, clip_max=2.0
            )
            
            if highlight_coord is not None and intersections:
                entry, exit = intersections
                path_vec = exit - entry
                path_len_sq = np.dot(path_vec, path_vec)

                if path_len_sq > 1e-6:
                    # 1. Project the highlight_coord onto the ray path vector
                    vec_to_highlight = highlight_coord - entry
                    
                    # 2. Calculate the normalized distance (t) along the path
                    t = np.dot(vec_to_highlight, path_vec) / path_len_sq
                    t_clipped = np.clip(t, 0.0, 1.0)
                    
                    # 3. Find which two segment colors to interpolate between
                    segment_float_index = t_clipped * cfg.NUM_SEGMENTS
                    start_index = int(segment_float_index)
                    
                    # Ensure we don't go out of bounds
                    if start_index >= cfg.NUM_SEGMENTS:
                        start_index = cfg.NUM_SEGMENTS - 1
                    
                    # 4. Interpolate the color
                    # Get the color at the beginning of the segment
                    if start_index == 0:
                        # For the first segment, the start color is the initial bright color
                        color_start = cfg.GODRAY 
                    else:
                        # Otherwise, it's the color from the end of the previous segment
                        color_start = transmitted_segs[start_index - 1][2]
                    
                    # Get the color at the end of the segment
                    color_end = transmitted_segs[start_index][2]
                    
                    # The interpolation factor is the fractional part of the float_index
                    t_interp = segment_float_index - start_index
                    
                    # Perform the linear interpolation (lerp)
                    sun_draw_color = (1 - t_interp) * color_start + t_interp * color_end

            # Draw the special ray's segments
            for start, end, tcolor in transmitted_segs:
                p, = ax.plot([start[0], end[0]], [start[1], end[1]], color=np.clip(tcolor,0,1), lw=4, zorder=9, alpha=0.8)
                ray_plots.append(p)
            ray_end = exit + ray_direction * cfg.ATMOSPHERE_RADIUS * 2
            final_color = transmitted_segs[-1][2] if transmitted_segs else cfg.SUN_COLOR
            p_after, = ax.plot([exit[0], ray_end[0]], [exit[1], ray_end[1]], color=np.clip(final_color,0,1), lw=4, zorder=9, alpha=0.8)
            ray_plots.append(p_after)

    # --- Standard Simulation for Canvases ---
    scattered_colors, transmitted_colors = color_canvas.sample_colors_along_line(
        line_points, sun_pos, sliders['Max Enhancement'].val, sliders['Exp. Coefficient'].val
    )
    blended_colors = [np.clip((1-np.mean(t)) * t + np.mean(t) * s, 0, 1) for s, t in zip(scattered_colors, transmitted_colors)]
    
    color_canvas.update_canvases(scattered_colors, transmitted_colors, blended_colors, sun_metrics, sun_draw_color)

    # --- Draw Main View Ray Grid ---
    num_rays, viewer_color_set = 21, False
    for i in range(num_rays):
        start_offset = np.linspace(-cfg.ATMOSPHERE_RADIUS, cfg.ATMOSPHERE_RADIUS, num_rays)
        ray_origin = sun_pos + perp_dir * start_offset[i]
        intersections = get_ray_atmosphere_intersections(ray_origin, ray_direction, cfg.ATMOSPHERE_RADIUS)
        if intersections:
            entry, exit = intersections
            _, segments, t_color = simulate_scattering(entry, exit, cfg.SUN_COLOR, sliders['Max Enhancement'].val, sliders['Exp. Coefficient'].val)
            for s, e, c in segments:
                ray_plots.append(ax.plot([s[0],e[0]], [s[1],e[1]], color=c, lw=2.5, zorder=8)[0])
            ray_plots.append(ax.plot([ray_origin[0], entry[0]], [ray_origin[1], entry[1]], color=cfg.SUN_COLOR, lw=2.5, zorder=8)[0])
            ray_end = exit + ray_direction * cfg.ATMOSPHERE_RADIUS * 2
            ray_plots.append(ax.plot([exit[0], ray_end[0]], [exit[1], ray_end[1]], color=t_color, lw=2.5, zorder=8)[0])
            if i == num_rays // 2:
                viewer_dot.set_color(t_color)
                viewer_color_set = True
        else:
            ray_end = ray_origin + ray_direction * cfg.SUN_DISTANCE * 2
            ray_plots.append(ax.plot([ray_origin[0], ray_end[0]], [ray_origin[1], ray_end[1]], color=cfg.SUN_COLOR, lw=2.5, zorder=8)[0])
            
    if not viewer_color_set:
        viewer_dot.set_color(cfg.NIGHT_COLOR)
        
    ax.set_title(f'Volumetric Scattering (Sun at {sun_angle_deg:.1f}°)', color=line_color)
    fig.canvas.draw_idle()
    fig_blended.canvas.draw_idle()
    

# --- Final Setup ---
limit = cfg.SUN_DISTANCE + cfg.SUN_RADIUS + 1
ax.set_xlim(-limit, limit)
ax.set_ylim(-limit, limit)
ax.axis('off')

# Initial draw
update(None) 
plt.show()