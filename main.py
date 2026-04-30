# main.py
"""
Main script to run the atmospheric scattering simulation.
This version keeps previews, controls, and generated animation in one window.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.widgets import Button

from SunClass import Sun
import config as cfg
from canvas import (
    ColorCanvas,
    calculate_distance_to_edge,
    get_closest_half_circle,
    get_perpendicular_from_origin,
    sample_colors_along_line,
    simulate_scattering_middle,
)
from style import (
    ACCENT_COLOR,
    ACCENT_HOVER_COLOR,
    AFTER_SUN_EDGE_COLOR,
    BACKGROUND_COLOR,
    BUTTON_TEXT_COLOR,
    PANEL_COLOR,
    SCENE_EDGE_COLOR,
    SCENE_ORBIT_COLOR,
    SCENE_RAY_GUIDE_COLOR,
    SCENE_RING_COLOR,
    TEXT_COLOR,
    ANIMATION_SLIDER_BOTTOM_Y,
    ANIMATION_SLIDER_HEIGHT,
    ANIMATION_SLIDER_TOP_Y,
    add_slider_group,
    apply_panel_facecolors,
    build_layout,
    create_button_axis,
    create_figure,
    style_scene_axis,
    style_video_axis,
)
from video import create_video

fig = create_figure()
axes = build_layout(fig)
ax_preview_before = axes["preview_before"]
ax_preview_after = axes["preview_after"]
ax_scene = axes["scene"]
ax_atmosphere_controls = axes["atmosphere_controls"]
ax_line_controls = axes["line_controls"]
ax_animation_controls = axes["animation_controls"]
ax_video = axes["video"]

panel_axes = [
    ax_preview_before,
    ax_preview_after,
    ax_scene,
    ax_atmosphere_controls,
    ax_line_controls,
    ax_animation_controls,
    ax_video,
]
apply_panel_facecolors(panel_axes)
style_video_axis(ax_video)


scene_limit = cfg.SUN_DISTANCE + cfg.SUN_RADIUS + 1
style_scene_axis(ax_scene, scene_limit)
ax_scene.add_patch(
    Circle((0, 0), cfg.ATMOSPHERE_RADIUS, fill=False, edgecolor=SCENE_RING_COLOR, ls="--", lw=1.4, zorder=5)
)
ax_scene.add_patch(
    Circle((0, 0), cfg.SUN_DISTANCE, fill=False, edgecolor=SCENE_ORBIT_COLOR, ls=":", lw=1.0, zorder=1)
)
viewer_dot, = ax_scene.plot(0, 0, "o", ms=8, color=TEXT_COLOR, zorder=11)

sun = Sun(ax_scene)
sun_after = Sun(ax_scene)
sun_after.outer_sun.set_alpha(0.35)
sun_after.outer_sun.set_zorder(6)

color_canvas = ColorCanvas(ax_preview_before, ax_preview_after, ax_video, BACKGROUND_COLOR, TEXT_COLOR)
ray_plots = []
current_animation = None
widget_axes = []
widget_texts = []


ax_animation_controls.set_facecolor(PANEL_COLOR)
animation_button_ax = create_button_axis(fig, ax_animation_controls)
widget_axes.append(animation_button_ax)

sliders = {}
sliders.update(
    add_slider_group(
        fig,
        ax_atmosphere_controls,
        {
            "title": "Atmosphere",
            "items": [
                {"label": "Sun Angle", "min": 0, "max": 360, "init": 0.0},
                {"label": "Enhancement", "min": 0.1, "max": cfg.MAX_COLOR_ENHANCEMENT, "init": 33.0},
                {"label": "Density Falloff", "min": -200, "max": 0, "init": cfg.EXPONENTIAL_COEFFICIENT},
            ],
        },
        widget_axes,
        widget_texts,
    )
)
sliders.update(
    add_slider_group(
        fig,
        ax_line_controls,
        {
            "title": "Sampling Line",
            "items": [
                {"label": "Start Point", "min": 0, "max": 180, "init": 45.0},
                {"label": "End Point", "min": 0, "max": 180, "init": 135.0},
                {"label": "Line Bend", "min": -1, "max": 1, "init": -1.0},
                {"label": "Sampling Radius", "min": 0.1, "max": cfg.ATMOSPHERE_RADIUS, "init": cfg.ATMOSPHERE_RADIUS},
            ],
        },
        widget_axes,
        widget_texts,
    )
)
sliders.update(
    add_slider_group(
        fig,
        ax_animation_controls,
        {
            "title": "Animation",
            "top_y": ANIMATION_SLIDER_TOP_Y,
            "bottom_y": ANIMATION_SLIDER_BOTTOM_Y,
            "slider_height": ANIMATION_SLIDER_HEIGHT,
            "items": [
                {"label": "Rotation", "min": -360, "max": 360, "init": 90.0},
                {"label": "Duration", "min": 1, "max": 20, "init": 8.0, "step": 1},
                {"label": "FPS", "min": 10, "max": 60, "init": 24.0, "step": 1},
            ],
        },
        widget_axes,
        widget_texts,
    )
)

generate_button = Button(animation_button_ax, "Generate animation", color=ACCENT_COLOR, hovercolor=ACCENT_HOVER_COLOR)
generate_button.label.set_color(BUTTON_TEXT_COLOR)


def get_blended_colors(sun_obj, enhancement, coefficient):
    scattered, direct = simulate_scattering_middle(enhancement, coefficient)
    color_num = sample_colors_along_line(sun_obj.sampling_line, sun_obj.angle_rad)

    scattered_colors = [scattered[i] for i in color_num]
    transmitted_colors = [direct[i] for i in color_num]
    blended_colors = [
        np.clip((1 - np.mean(transmitted)) * transmitted + np.mean(transmitted) * scattered_color, 0, 1)
        for scattered_color, transmitted in zip(scattered_colors, transmitted_colors)
    ]
    return scattered, direct, scattered_colors, transmitted_colors, blended_colors


def update_sun_draw_color(sun_obj, enhancement, coefficient):
    if sun_obj.line_pos is None:
        sun_obj.set_color(cfg.SUN_COLOR)
        return

    _, sun_color_segments = simulate_scattering_middle(enhancement, coefficient, cfg.GODRAY)
    dist, _, _ = calculate_distance_to_edge(sun_obj.line_pos[0], sun_obj.line_pos[1], sun_obj.angle_rad)
    num_seg = int(dist // cfg.SEGMENT_LENGTH)
    num_seg = min(num_seg, len(sun_color_segments) - 1)
    sun_obj.set_color(sun_color_segments[num_seg])


def update_preview_panels(before_colors, after_colors, before_metrics, after_metrics, before_sun_color, after_sun_color):
    color_canvas.draw_preview(ax_preview_before, before_colors, "Before", before_metrics, before_sun_color)
    color_canvas.draw_preview(ax_preview_after, after_colors, "After", after_metrics, after_sun_color)


def update_scene(scattered, direct):
    global ray_plots

    while ray_plots:
        ray_plots.pop().remove()

    sun.ray_dir = -sun.position / np.linalg.norm(sun.position)

    if sun.line_pos is not None:
        dist, x, y = calculate_distance_to_edge(sun.line_pos[0], sun.line_pos[1], sun.angle_rad)
        _, sun_color_segments = simulate_scattering_middle(
            sliders["Enhancement"].val,
            sliders["Density Falloff"].val,
            cfg.GODRAY,
        )
        num_seg = int(dist // cfg.SEGMENT_LENGTH)
        num_seg = min(num_seg, len(sun_color_segments) - 1)

        for i in range(num_seg):
            x_end = x + sun.ray_dir[0] * cfg.SEGMENT_LENGTH
            y_end = y + sun.ray_dir[1] * cfg.SEGMENT_LENGTH
            ray_plots.append(ax_scene.plot((x, x_end), (y, y_end), color=sun_color_segments[i], lw=5, zorder=10)[0])
            x = x_end
            y = y_end
    else:
        sun.set_color(cfg.SUN_COLOR)

    ray_plots.append(ax_scene.plot(sun.sampling_line[:, 0], sun.sampling_line[:, 1], color=ACCENT_COLOR, lw=2.5, zorder=15)[0])

    a, b = get_perpendicular_from_origin(sun.position, cfg.ATMOSPHERE_RADIUS)
    ray_plots.append(ax_scene.plot((a[0], b[0]), (a[1], b[1]), color=SCENE_EDGE_COLOR, lw=2.0, zorder=8)[0])

    x_coords, y_coords, ray_length = get_closest_half_circle(sun.angle_rad, cfg.ATMOSPHERE_RADIUS)
    ray_plots.append(ax_scene.plot(x_coords, y_coords, color=SCENE_RAY_GUIDE_COLOR, linewidth=3.0, alpha=0.75, zorder=4)[0])

    n_segs = (ray_length // cfg.SEGMENT_LENGTH).astype(int)
    rest_length = np.where(
        n_segs > 0,
        ray_length - (n_segs * cfg.SEGMENT_LENGTH),
        ray_length,
    )

    for n in range(cfg.NUM_RAYS):
        x = x_coords[n]
        y = y_coords[n]

        if n_segs[n]:
            for i in range(n_segs[n]):
                x_end = x + sun.ray_dir[0] * cfg.SEGMENT_LENGTH
                y_end = y + sun.ray_dir[1] * cfg.SEGMENT_LENGTH
                ray_plots.append(ax_scene.plot((x, x_end), (y, y_end), color=direct[i], lw=2.8, zorder=2)[0])
                x = x_end
                y = y_end

        if rest_length[n] > 0:
            seg_index = min(n_segs[n], len(direct) - 1)
            x_end = x + sun.ray_dir[0] * rest_length[n]
            y_end = y + sun.ray_dir[1] * rest_length[n]
            ray_plots.append(ax_scene.plot((x, x_end), (y, y_end), color=direct[seg_index], lw=2.8, zorder=2)[0])


def update(_):
    sun.rotate_deg(sliders["Sun Angle"].val)
    sun.update_sampling_line(
        sliders["Start Point"].val,
        sliders["End Point"].val,
        sliders["Line Bend"].val,
        sliders["Sampling Radius"].val,
    )
    sun.calculate_sun_metrics()

    end_angle = sliders["Sun Angle"].val + sliders["Rotation"].val
    sun_after.rotate_deg(end_angle)
    sun_after.update_sampling_line(
        sliders["Start Point"].val,
        sliders["End Point"].val,
        sliders["Line Bend"].val,
        sliders["Sampling Radius"].val,
    )
    sun_after.calculate_sun_metrics()

    sun_after.outer_sun.set_edgecolor(AFTER_SUN_EDGE_COLOR)
    sun_after.outer_sun.set_facecolor(np.clip(cfg.SUN_COLOR, 0, 1))

    update_sun_draw_color(sun, sliders["Enhancement"].val, sliders["Density Falloff"].val)
    update_sun_draw_color(sun_after, sliders["Enhancement"].val, sliders["Density Falloff"].val)

    _, direct, _, _, before_blended = get_blended_colors(
        sun,
        sliders["Enhancement"].val,
        sliders["Density Falloff"].val,
    )
    _, _, _, _, after_blended = get_blended_colors(
        sun_after,
        sliders["Enhancement"].val,
        sliders["Density Falloff"].val,
    )

    update_preview_panels(
        before_blended,
        after_blended,
        sun.canvas_y,
        sun_after.canvas_y,
        sun.color,
        sun_after.color,
    )
    update_scene(*simulate_scattering_middle(sliders["Enhancement"].val, sliders["Density Falloff"].val))
    style_scene_axis(ax_scene, scene_limit)
    fig.canvas.draw_idle()


def btn_video(_event):
    global current_animation
    if current_animation is not None and current_animation.event_source is not None:
        current_animation.event_source.stop()
    current_animation = create_video(
        sliders["Enhancement"].val,
        sliders["Density Falloff"].val,
        sun,
        sliders["Rotation"].val,
        int(sliders["Duration"].val),
        fps=int(sliders["FPS"].val),
        ax_v=ax_video,
    )
    fig.canvas.draw_idle()


generate_button.on_clicked(btn_video)
for slider in sliders.values():
    slider.on_changed(update)

update(None)
plt.show()
