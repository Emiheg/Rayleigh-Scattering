import copy

from matplotlib import patches
import matplotlib.animation as animation
import matplotlib.image as mpimg
import numpy as np
import SunClass
import config as cfg

from canvas import calculate_distance_to_edge, simulate_scattering_middle, sample_colors_along_line


animations = []


def _fit_image_size(image, max_height, max_width):
    source_height, source_width = image.shape[:2]
    scale = min(max_width / source_width, max_height / source_height, 1.0)
    height = max(1, int(round(source_height * scale)))
    width = max(1, int(round(source_width * scale)))
    return height, width


def _resize_image_nearest(image, height, width):
    source_height, source_width = image.shape[:2]
    y_indices = np.linspace(0, source_height - 1, height).astype(int)
    x_indices = np.linspace(0, source_width - 1, width).astype(int)
    return image[y_indices][:, x_indices]


def get_centered_image_extent(image_height, image_width, canvas_height, canvas_width):
    x_padding = (canvas_width - image_width) * 0.5
    y_padding = (canvas_height - image_height) * 0.5
    return [
        x_padding,
        x_padding + image_width,
        y_padding,
        y_padding + image_height,
    ]


def load_grayscale_alpha_map(image_path, max_height, max_width):
    image = mpimg.imread(image_path)
    height, width = _fit_image_size(image, max_height, max_width)
    image = _resize_image_nearest(image, height, width)

    if image.dtype.kind in {"u", "i"}:
        image = image.astype(np.float32) / np.iinfo(image.dtype).max
    else:
        image = image.astype(np.float32)

    if image.ndim == 2:
        grayscale = image
    else:
        rgb = image[:, :, :3]
        grayscale = (
            0.2126 * rgb[:, :, 0]
            + 0.7152 * rgb[:, :, 1]
            + 0.0722 * rgb[:, :, 2]
        )
    alpha = image[:, :, 3] if image.ndim == 3 and image.shape[2] >= 4 else np.ones_like(grayscale)

    return np.clip(grayscale, 0.0, 1.0), np.clip(alpha, 0.0, 1.0)


def load_grayscale_value_map(image_path, max_height, max_width):
    grayscale, _ = load_grayscale_alpha_map(image_path, max_height, max_width)
    return grayscale


def replace_grayscale_with_colors(grayscale_values, colors, alpha_values=None):
    """
    Replace grayscale values with equally spaced colors from black (top) to white (bottom).
    """
    if len(colors) == 0:
        channel_count = 4 if alpha_values is not None else 3
        return np.zeros((*grayscale_values.shape, channel_count), dtype=np.float32)

    color_array = np.clip(np.asarray(colors, dtype=np.float32), 0.0, 1.0)
    color_count = len(color_array)
    color_indices = np.floor((1.0 - grayscale_values) * color_count).astype(int)
    color_indices = np.clip(color_indices, 0, color_count - 1)

    image_data = color_array[color_indices]
    if alpha_values is not None:
        image_data = np.dstack((image_data, np.clip(alpha_values, 0.0, 1.0)))

    return image_data


def create_video(enhancement, coefficient, sun_orig: SunClass.Sun, rotation_deg, vid_length, fps=30, ax_v=None, image_path=None):
    if ax_v is None:
        raise ValueError("create_video requires a target axis")

    width = 1500
    height = 800
    grayscale_values = None
    alpha_values = None
    if image_path:
        grayscale_values, alpha_values = load_grayscale_alpha_map(image_path, height, width)
    image_extent = (
        get_centered_image_extent(grayscale_values.shape[0], grayscale_values.shape[1], height, width)
        if grayscale_values is not None
        else [0, width, 0, height]
    )
    image_origin = "upper" if grayscale_values is not None else "lower"

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

        frame_states.append((blended_colors, sun_visible, sun_y, sun_color))

    def build_frame_image(colors):
        if grayscale_values is not None:
            return replace_grayscale_with_colors(grayscale_values, colors, alpha_values)

        return np.repeat(
            np.array(colors, dtype=np.float32).reshape((len(colors), 1, 3)),
            cfg.GRADIENT_WIDTH_MULTIPLIER,
            axis=1,
        )

    first_colors, first_visible, first_y, first_color = frame_states[0]
    image_artist = ax_v.imshow(
        build_frame_image(first_colors),
        aspect="equal",
        extent=image_extent,
        origin=image_origin,
        interpolation="antialiased",
        animated=True,
    )
    ax_v.set_xlim(0, width)
    ax_v.set_ylim(0, height)

    
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
        colors, sun_visible, sun_y, sun_color = frame_states[frame_index]
        image_artist.set_data(build_frame_image(colors))
        sun_artist.set_visible(sun_visible) if image_path is None else sun_artist.set_visible(False)
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
