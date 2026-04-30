import copy

from matplotlib import patches
import matplotlib.animation as animation
import numpy as np
import SunClass
import config as cfg

from canvas import calculate_distance_to_edge, simulate_scattering_middle, sample_colors_along_line


animations = []


def create_video(enhancement, coefficient, sun_orig: SunClass.Sun, rotation_deg, vid_length, fps=30, ax_v=None):
    if ax_v is None:
        raise ValueError("create_video requires a target axis")

    width = 1500
    height = 800

    ax_v.clear()
    ax_v.set_facecolor("#101418")
    ax_v.set_title("Generated animation", color="#f0f0f0", fontsize=13)
    ax_v.axis("off")

    fig_v = ax_v.figure

    scattered, direct = simulate_scattering_middle(enhancement, coefficient)
    _, sun_color_segments = simulate_scattering_middle(enhancement, coefficient, cfg.GODRAY)

    num_frames = max(2, int(round(vid_length * fps)))
    frame_interval_ms = 1000.0 / fps

    sun = copy.deepcopy(sun_orig)
    start_deg = sun.angle_deg
    frame_states = []

    for angle_deg in np.linspace(start_deg, rotation_deg + start_deg, num_frames):
        sun.rotate_deg(angle_deg)
        color_num = sample_colors_along_line(sun.sampling_line, sun.angle_rad)

        scattered_colors = [scattered[c] for c in color_num]
        direct_colors = [direct[c] for c in color_num]
        
        blended_colors = [
            np.clip((1 - np.mean(transmitted)) * transmitted + np.mean(transmitted) * scattered_color, 0, 1)
            for scattered_color, transmitted in zip(scattered_colors, direct_colors)
        ]
        #For further implementation, multiple different blending methods can be implemented and tested
        #blended_colors = [scattered_colors + direct_colors * 0.5 for scattered_colors, direct_colors in zip(scattered_colors, direct_colors)]

        img_data = np.repeat(
            np.array(blended_colors).reshape((len(blended_colors), 1, 3)),
            cfg.GRADIENT_WIDTH_MULTIPLIER,
            axis=1,
        )

        sun.calculate_sun_metrics()
        sun_visible = sun.canvas_y is not None and sun.line_pos is not None
        sun_y = 0.0
        sun_color = np.clip(cfg.SUN_COLOR, 0, 1)

        if sun_visible:
            dist, _, _ = calculate_distance_to_edge(sun.line_pos[0], sun.line_pos[1], sun.angle_rad)
            num_seg = int(dist // cfg.SEGMENT_LENGTH)
            num_seg = min(num_seg, len(sun_color_segments) - 1)
            sun_color = np.clip(sun_color_segments[num_seg], 0, 1)
            sun_y = sun.canvas_y * height

        frame_states.append((img_data, sun_visible, sun_y, sun_color))

    first_img, first_visible, first_y, first_color = frame_states[0]
    image_artist = ax_v.imshow(
        first_img,
        aspect="equal",
        extent=[0, width, 0, height],
        origin="lower",
        interpolation="antialiased",
        animated=True,
    )
    sun_artist = patches.Circle(
        (width * 0.5, first_y),
        radius=10,
        color=first_color,
        linewidth=0,
        visible=first_visible,
        animated=True,
        zorder=10,
    )
    ax_v.add_patch(sun_artist)

    def update_frame(frame_index):
        img_data, sun_visible, sun_y, sun_color = frame_states[frame_index]
        image_artist.set_data(img_data)
        sun_artist.set_visible(sun_visible)
        if sun_visible:
            sun_artist.center = (width * 0.5, sun_y)
            sun_artist.set_facecolor(sun_color)
        return image_artist, sun_artist

    ani = animation.FuncAnimation(
        fig_v,
        update_frame,
        frames=len(frame_states),
        interval=frame_interval_ms,
        blit=True,
        repeat=True,
        repeat_delay=0,
    )
    animations.append(ani)
    return ani
