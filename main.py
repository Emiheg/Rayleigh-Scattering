# main.py
"""
Main script to run the atmospheric scattering simulation.
This file sets up the plot, widgets, and the main update loop.
To run: python main.py
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.widgets import Slider, Button
import matplotlib.gridspec as gridspec

from SunClass import Sun
import config as cfg
from canvas import *
from video import create_video

# --- Setup Windows & Plots ---
background_color, line_color = '#1c1c1c', '#f0f0f0'

# --- Window 1: Main Simulation ---
fig = plt.figure(figsize=(12, 8))
fig.canvas.manager.set_window_title('Scattering Simulation Controls')
fig.set_facecolor(background_color)
gs_main = gridspec.GridSpec(2, 3, width_ratios=[1, 3, 1], height_ratios=[3, 1])
ax = fig.add_subplot(gs_main[0, 1])

ax.set_aspect('equal')
ax.set_facecolor(background_color)

# --- Video settings ---
vx = fig.add_subplot(gs_main[0, 0])
vx.set_title('video')
vx.axis('off')
gs_vid = gs_main[0, 0].subgridspec(4, 1, hspace=0.4)


# --- Window 2: Blended Result ---
fig_blended = plt.figure(figsize=(3, 3.5)) # A smaller, squarish window
fig_blended.canvas.manager.set_window_title('Blended Result')
fig_blended.set_facecolor(background_color)
ax_blended_new = fig_blended.add_subplot(111) # A single axis for this figure

# --- Setup Canvases in Main Window  ---
gs_side_canvases = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs_main[0, 2], hspace=0.6)
ax_scattered = fig.add_subplot(gs_side_canvases[0, 0])
ax_transmitted = fig.add_subplot(gs_side_canvases[1, 0])

# --- Draw Static Elements ---
ax.add_patch(Circle((0,0), cfg.ATMOSPHERE_RADIUS, fill=False, edgecolor=line_color, ls='--', lw=1.5, zorder=5))
ax.add_patch(Circle((0,0), cfg.SUN_DISTANCE, fill=False, edgecolor='gray', ls=':', lw=1, zorder=1))
viewer_dot, = ax.plot(0, 0, 'o', ms=10, color=line_color, zorder=11)

# --- Initialize Dynamic Elements ---
sun = Sun(ax)

ray_plots = []
color_canvas = ColorCanvas(ax_scattered, ax_transmitted, ax_blended_new, background_color, line_color)

# --- Create Sliders ---
left, right, slider_h, v_start = 0.1, 0.6, 0.025, 0.20
sliders_def = [
    ('Sun Angle Degree', [right, v_start, 0.32, slider_h], {'valmin':0, 'valmax':360, 'valinit':0.0}),
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



def btn_video(event):

    create_video(
        sliders['Max Enhancement'].val, 
        sliders['Exp. Coefficient'].val,
        sun,
        360,
        10
        )
    

btn = Button(fig.add_subplot(gs_vid[0]), 'Create', color=line_color)
btn.on_clicked(btn_video)

# --- Define the Update Function ---
def update(val):
    global ray_plots
    [line.remove() for line in ray_plots]; ray_plots.clear()

    sun.rotate_deg(sliders['Sun Angle Degree'].val)

    sun.update_sampling_line(
         sliders['Start Point'].val, sliders['End Point'].val, sliders['Line Bend'].val, sliders['Sampling Radius'].val
    )

    ray_plots.append(ax.plot(sun.sampling_line[:, 0], sun.sampling_line[:, 1], color="red", lw=2, zorder=15)[0])
    
    sun.ray_dir = -sun.position / np.linalg.norm(sun.position)

    # --- Special "God Ray" Simulation if sun is on canvas ---
    if sun.line_pos is not None:

        dist, x, y = calculate_distance_to_edge(sun.line_pos[0], sun.line_pos[1], sun.angle_rad)

        _, sun_color_segments = simulate_scattering_middle(
            sliders['Max Enhancement'].val, sliders['Exp. Coefficient'].val, cfg.GODRAY
        )

        num_seg = (dist//cfg.SEGMENT_LENGTH).astype(int)
        for i in range(num_seg):
            
            ray_plots.append(ax.plot((x,  x + sun.ray_dir[0] * cfg.SEGMENT_LENGTH ), (y, y + sun.ray_dir[1] * cfg.SEGMENT_LENGTH), color=sun_color_segments[i], lw=5, zorder=10)[0])

            x +=  sun.ray_dir[0] * cfg.SEGMENT_LENGTH
            y +=  sun.ray_dir[1] * cfg.SEGMENT_LENGTH
            sun.set_color(sun_color_segments[i])


    # --- Draw Main View Ray Grid ---

    a, b = get_perpendicular_from_origin(sun.position, cfg.ATMOSPHERE_RADIUS)
    
    ray_plots.append(ax.plot((a[0], b[0]), (a[1], b[1]),  color="white", lw=2.5, zorder=8)[0])


    scattered, direct = simulate_scattering_middle(sliders['Max Enhancement'].val, sliders['Exp. Coefficient'].val)

    x_coords, y_coords, ray_length = get_closest_half_circle(sun.angle_rad, cfg.ATMOSPHERE_RADIUS)
    ax.plot(x_coords, y_coords, label='Closest Half', color='blue', linewidth=4)

    n_segs = (ray_length // cfg.SEGMENT_LENGTH).astype(int)      # Number of whole segments fitting in the ray
    rest_length = ray_length % (n_segs * cfg.SEGMENT_LENGTH)    # The rest value finishing the ray in the atmosphere - smaller than one segment
    ray_color = direct                                          # If it shows scattered or direct light values
    
    for n in range(cfg.NUM_RAYS):
        x = x_coords[n]
        y = y_coords[n]

        if(n_segs[n]):
            for i in range(n_segs[n]):
                x_end = x + sun.ray_dir[0] * cfg.SEGMENT_LENGTH
                y_end = y + sun.ray_dir[1] * cfg.SEGMENT_LENGTH
                ray_plots.append(ax.plot((x,  x_end ), (y, y_end), color=ray_color[i], lw=3, zorder=0)[0])

                x = x_end
                y = y_end

        
        if(rest_length[n] > 0):
            rest_color = n_segs[n] + 1
            if rest_color > len(ray_color):
                rest_color = n_segs[n]
            ray_plots.append(ax.plot((x,  x + sun.ray_dir[0] * rest_length[n] ), (y, y + sun.ray_dir[1] * rest_length[n]), color=ray_color[n_segs[n]], lw=3, zorder= 0)[0])


    # --- Standard Simulation for Canvases ---
    color_num = sample_colors_along_line(sun.sampling_line, sun.angle_rad)

    scattered_colors, transmitted_colors = [],[]

    for i in color_num: scattered_colors.append(scattered[i]), transmitted_colors.append(direct[i])

    blended_colors = [np.clip((1-np.mean(t)) * t + np.mean(t) * s, 0, 1) for s, t in zip(scattered_colors, transmitted_colors)]
    
    sun.calculate_sun_metrics()
    color_canvas.update_canvases(scattered_colors, transmitted_colors, blended_colors, sun.canvas_y, sun.color)

    ax.set_title(f'Volumetric Scattering (Sun at {sun.angle_deg:.1f}°)', color=line_color)
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