import imageio
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, Video, Image

def create_video(arrays, labels, x_axis=None, file_name='animation.mp4', fps=100, dpi=100, output_format='MP4'):
    """
    Creates an MP4 video or GIF showing the evaluation of a system over time, given by data in an
    array where the time scale is along the first dimension.

    Parameters
    ----------
    arrays : list of ndarray
        List of numpy arrays containing the data to be plotted.
    labels : list of str
        List of labels for the data series, used in the legend of the plot, corresponding to the
        list of arrays.
    x_axis : ndarray, optional
        x-axis values. If provided, data will be plotted against these x-axis values. If None,
        data will be plotted against the array indices.
    file_name : str, optional
        Name of the output animation file. Defaults to 'animation.mp4'.
    fps : int, optional
        Frames per second for the animation. Defaults to 10.
    dpi : int, optional
        Dots per inch for the plot's resolution. Defaults to 100.
    output_format : str, optional
        Output format for the animation. Can be 'MP4' or 'GIF'.
        Defaults to 'MP4'.

    Returns
    -------
    Video or None
        Returns a Video object if the output_format is 'MP4'. If output_format is 'GIF',
        the GIF is displayed and None is returned.
    """

    fig, ax = plt.subplots(figsize=(7.04, 4), dpi=dpi)
    colors = [(0,0,0),(0,0.4,1),(1,0.7,0.3),(0.2,0.7,0.2),(0.8,0,0.2),(0.5,0.3,.9)][:len(arrays)]
    lower_value = min(min(np.min(data) for data in arrays), -2)
    upper_value = max(max(np.max(data) for data in arrays), 2*np.max(arrays[0]))
    limit_value = max(np.abs(lower_value), np.abs(upper_value))
    ax.set_ylim(-limit_value, limit_value)

    with imageio.get_writer(file_name, mode='I', fps=fps) as writer:
        for frame in range(arrays[0].shape[0]):
            lines = []
            for (data, color, label) in zip(arrays, colors, labels):
                if x_axis is not None:
                    line, = ax.plot(x_axis, data[frame, :], color=color, label=label)
                else:
                    line, = ax.plot(data[frame, :], color=color, label=label)
                lines.append(line)
            ax.legend(loc='upper right')
            ax.set_xlabel('$x$', fontsize=12)
            ax.set_ylabel('$u$', fontsize=12)
            fig.canvas.draw()
            image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
            image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            writer.append_data(image)
            ax.clear()
            ax.set_ylim(-0.25, 0.25)
        plt.close(fig)
    
    if output_format == 'MP4':
        return Video(file_name)
    elif output_format == 'GIF':
        with open(file_name,'rb') as f:
            display(Image(data=f.read(), format='gif'))
        return None