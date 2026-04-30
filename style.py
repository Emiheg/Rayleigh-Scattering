"""
UI style constants and layout helpers for the scattering dashboard.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Slider


BACKGROUND_COLOR = "#101418"
PANEL_COLOR = "#182129"
TEXT_COLOR = "#f4f1e8"
MUTED_TEXT_COLOR = "#c8bcae"
ACCENT_COLOR = "#e48d53"
ACCENT_HOVER_COLOR = "#f0a066"
BUTTON_TEXT_COLOR = "#1b1b1b"
SLIDER_COLOR = "#c1643c"
TRACK_COLOR = "#31414c"
SLIDER_TRACK_FILL_COLOR = "#24323b"
SCENE_RING_COLOR = "#d9d2c2"
SCENE_ORBIT_COLOR = "#667887"
SCENE_RAY_GUIDE_COLOR = "#5f97cf"
SCENE_EDGE_COLOR = "#f4f1e8"
AFTER_SUN_EDGE_COLOR = "#f7c88a"

FIGURE_SIZE = (16, 9)
WINDOW_TITLE = "Rayleigh Scattering Studio"
FIGURE_LEFT = 0.03
FIGURE_RIGHT = 0.98
FIGURE_TOP = 0.95
FIGURE_BOTTOM = 0.05
OUTER_WSPACE = 0.08
SIDE_PANEL_RATIO = 1
MAIN_PANEL_RATIO = 2

SIDE_SECTION_HEIGHTS = [1.0, 1.55, 1.0, 1.0, 1.0]
SIDE_SECTION_HSPACE = 0.35
PREVIEW_WSPACE = 0.15

VIDEO_TITLE = "Generated animation"
VIDEO_TITLE_SIZE = 15
VIDEO_TITLE_PAD = 16
VIDEO_PLACEHOLDER_TEXT = "Generate an animation to preview it here"
VIDEO_PLACEHOLDER_SIZE = 18
VIDEO_PLACEHOLDER_X = 0.5
VIDEO_PLACEHOLDER_Y = 0.5

SCENE_TITLE = "Atmosphere view"
SCENE_TITLE_SIZE = 12
SCENE_TITLE_PAD = 12
CONTROL_TITLE_SIZE = 12
CONTROL_TITLE_X = 0.03
CONTROL_TITLE_Y = 0.9
CONTROL_LABEL_SIZE = 10

SLIDER_DEFAULT_TOP_Y = 0.66
SLIDER_DEFAULT_BOTTOM_Y = 0.12
SLIDER_DEFAULT_HEIGHT = 0.11
SLIDER_DEFAULT_LABEL_X = 0.03
SLIDER_DEFAULT_X = 0.33
SLIDER_DEFAULT_WIDTH = 0.60
SLIDER_AXIS_ZORDER = 20
SLIDER_LABEL_VERTICAL_CENTER = 0.5

BUTTON_X = 0.04
BUTTON_Y = 0.04
BUTTON_WIDTH = 0.92
BUTTON_HEIGHT = 0.22
BUTTON_ZORDER = 20

ANIMATION_SLIDER_TOP_Y = 0.68
ANIMATION_SLIDER_BOTTOM_Y = 0.34
ANIMATION_SLIDER_HEIGHT = 0.10


def create_figure():
    fig = plt.figure(figsize=FIGURE_SIZE, constrained_layout=False)
    fig.canvas.manager.set_window_title(WINDOW_TITLE)
    fig.set_facecolor(BACKGROUND_COLOR)
    return fig


def build_layout(fig):
    outer = gridspec.GridSpec(
        1,
        2,
        figure=fig,
        width_ratios=[SIDE_PANEL_RATIO, MAIN_PANEL_RATIO],
        left=FIGURE_LEFT,
        right=FIGURE_RIGHT,
        top=FIGURE_TOP,
        bottom=FIGURE_BOTTOM,
        wspace=OUTER_WSPACE,
    )

    side = outer[0, 0].subgridspec(5, 1, height_ratios=SIDE_SECTION_HEIGHTS, hspace=SIDE_SECTION_HSPACE)
    preview_grid = side[0].subgridspec(1, 2, wspace=PREVIEW_WSPACE)

    axes = {
        "preview_before": fig.add_subplot(preview_grid[0, 0]),
        "preview_after": fig.add_subplot(preview_grid[0, 1]),
        "scene": fig.add_subplot(side[1]),
        "atmosphere_controls": fig.add_subplot(side[2]),
        "line_controls": fig.add_subplot(side[3]),
        "animation_controls": fig.add_subplot(side[4]),
        "video": fig.add_subplot(outer[0, 1]),
    }
    return axes


def apply_panel_facecolors(axes):
    for axis in axes:
        axis.set_facecolor(PANEL_COLOR)


def style_video_axis(axis):
    axis.set_title(VIDEO_TITLE, color=TEXT_COLOR, fontsize=VIDEO_TITLE_SIZE, loc="left", pad=VIDEO_TITLE_PAD)
    axis.text(
        VIDEO_PLACEHOLDER_X,
        VIDEO_PLACEHOLDER_Y,
        VIDEO_PLACEHOLDER_TEXT,
        color=MUTED_TEXT_COLOR,
        fontsize=VIDEO_PLACEHOLDER_SIZE,
        ha="center",
        va="center",
        transform=axis.transAxes,
    )
    axis.set_xticks([])
    axis.set_yticks([])


def style_scene_axis(axis, scene_limit):
    axis.set_aspect("equal")
    axis.set_xlim(-scene_limit, scene_limit)
    axis.set_ylim(-scene_limit, scene_limit)
    axis.axis("off")
    axis.set_title(SCENE_TITLE, color=TEXT_COLOR, fontsize=SCENE_TITLE_SIZE, loc="left", pad=SCENE_TITLE_PAD)


def style_control_panel(axis, title):
    axis.set_xticks([])
    axis.set_yticks([])
    axis.set_axis_off()
    axis.set_navigate(False)
    axis.patch.set_alpha(0)
    for spine in axis.spines.values():
        spine.set_visible(False)
    axis.text(
        CONTROL_TITLE_X,
        CONTROL_TITLE_Y,
        title,
        color=TEXT_COLOR,
        fontsize=CONTROL_TITLE_SIZE,
        fontweight="bold",
        ha="left",
        va="center",
        transform=axis.transAxes,
    )


def create_button_axis(fig, host_ax):
    bbox = host_ax.get_position()
    return fig.add_axes(
        [
            bbox.x0 + bbox.width * BUTTON_X,
            bbox.y0 + bbox.height * BUTTON_Y,
            bbox.width * BUTTON_WIDTH,
            bbox.height * BUTTON_HEIGHT,
        ],
        zorder=BUTTON_ZORDER,
    )


def add_slider_group(fig, host_ax, defs, widget_axes, widget_texts):
    style_control_panel(host_ax, defs["title"])
    sliders = {}
    bbox = host_ax.get_position()
    top_y = defs.get("top_y", SLIDER_DEFAULT_TOP_Y)
    bottom_y = defs.get("bottom_y", SLIDER_DEFAULT_BOTTOM_Y)
    slider_height = defs.get("slider_height", SLIDER_DEFAULT_HEIGHT)
    label_x = defs.get("label_x", SLIDER_DEFAULT_LABEL_X)
    slider_x = defs.get("slider_x", SLIDER_DEFAULT_X)
    slider_width = defs.get("slider_width", SLIDER_DEFAULT_WIDTH)

    y_positions = np.linspace(top_y, bottom_y, len(defs["items"]))
    for y, item in zip(y_positions, defs["items"]):
        label = fig.text(
            bbox.x0 + bbox.width * label_x,
            bbox.y0 + bbox.height * (y + slider_height * SLIDER_LABEL_VERTICAL_CENTER),
            item["label"],
            color=TEXT_COLOR,
            fontsize=CONTROL_LABEL_SIZE,
            ha="left",
            va="center",
        )
        widget_texts.append(label)

        slider_ax = fig.add_axes(
            [
                bbox.x0 + bbox.width * slider_x,
                bbox.y0 + bbox.height * y,
                bbox.width * slider_width,
                bbox.height * slider_height,
            ],
            zorder=SLIDER_AXIS_ZORDER,
        )
        slider_ax.set_facecolor(TRACK_COLOR)
        widget_axes.append(slider_ax)

        slider = Slider(
            ax=slider_ax,
            label="",
            valmin=item["min"],
            valmax=item["max"],
            valinit=item["init"],
            valstep=item.get("step"),
            color=SLIDER_COLOR,
        )
        slider.label.set_color(TEXT_COLOR)
        slider.valtext.set_color(TEXT_COLOR)
        if hasattr(slider, "track"):
            slider.track.set_facecolor(SLIDER_TRACK_FILL_COLOR)
        if hasattr(slider, "poly"):
            slider.poly.set_facecolor(SLIDER_COLOR)
        sliders[item["label"]] = slider

    return sliders
