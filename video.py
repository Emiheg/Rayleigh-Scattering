import copy

from matplotlib import patches
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import SunClass
import config as cfg

from canvas import calculate_distance_to_edge, simulate_scattering_middle, sample_colors_along_line

animations=[]

def create_video(enhancement, coefficient, sun_orig:SunClass.Sun, rotation_deg, vid_length, fps=30):
    
    #   Window settings
    width, height = 1000, 700
    fig_v = plt.figure()
    ax_v = fig_v.add_subplot()
    plt.show()
    # Simulate scattering for the given parameters to get the color segments for the sun and the rays
    scattered, direct = simulate_scattering_middle(enhancement, coefficient)
    _, sun_color_segments = simulate_scattering_middle(enhancement, coefficient, cfg.GODRAY)

    num_frames = vid_length * fps
    sun = copy.deepcopy(sun_orig)                # Create a copy of the sun object to manipulate its position without affecting the original
    sun.ax = ax_v                                # Redirect the sun's drawing to the new figure
    start_deg = sun.angle_deg 
    frames = []                                  # for storing the generated images
    # Create a loop for each frame with rotation radpf
    for i in np.linspace(start_deg, rotation_deg + start_deg, num_frames):
        sun.rotate_deg(i)    # Rotate the sun by the current angle
        color_num = sample_colors_along_line(sun.sampling_line, sun.angle_rad)
        # Draw Horizon
        scattered_colors, direct_colors = [], []
        for c in color_num: scattered_colors.append(scattered[c]), direct_colors.append(direct[c])

        blended_colors = [np.clip((1-np.mean(t)) * t + 
                                  np.mean(t) * s, 0, 1) for s, t in zip(scattered_colors, direct_colors)]
        color_array = np.array(blended_colors)          # Reshape the list of blended colors into a 2D array for imshow
        img_data = color_array.reshape((len(blended_colors), 1, 3)) # Reshape to (num_colors, 1, 3) for imshow

        # Use imshow to display the array as a canvas with the correct aspect ratio and extent
        img = ax_v.imshow(img_data, 
                        aspect='equal', 
                        extent=[0, width, 0, height], 
                        origin='lower',
                        animated=True)

        # Draw sun on canvas
        sun.calculate_sun_metrics()
        if (sun.canvas_y and sun.line_pos)is not None: 
            dist, _, _ = calculate_distance_to_edge(sun.line_pos[0], sun.line_pos[1], sun.angle_rad)


            
            num_seg = (dist//cfg.SEGMENT_LENGTH).astype(int)
            sun.set_color(sun_color_segments[num_seg])
            y_coord = sun.canvas_y * height
            display_color = np.clip(sun.color, 0, 1)
            sun_on_canvas = patches.Circle((width * 0.5, y_coord), radius=10, color=display_color, zorder=10)
            ax_v.add_patch(sun_on_canvas)
            frames.append([img, sun_on_canvas])
        else:
            frames.append([img])


    # Create the animation
    ani = animation.ArtistAnimation(fig_v, frames, interval=30, blit=True, repeat_delay=0)
    animations.append(ani)
    # Save the animation as a video file (e.g., MP4). 
    # Note: You might need a separate encoder like ffmpeg installed on your system.
    #ani.save('movie.mp4')
