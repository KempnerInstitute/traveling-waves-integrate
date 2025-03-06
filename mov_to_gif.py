from moviepy import VideoFileClip

def mov_to_gif(input_path, output_path, fps=10, start=None, end=None):
    """
    Convert a .mov video to an animated .gif using MoviePy.

    :param input_path: Path to the input .mov file
    :param output_path: Path for the output .gif file
    :param fps: Frames per second for the GIF output
    :param start: Start time (in seconds) for trimming (optional)
    :param end: End time (in seconds) for trimming (optional)
    """
    # Load the video
    clip = VideoFileClip(input_path)

    # Optionally trim the clip
    if start is not None and end is not None:
        clip = clip.subclip(start, end)

    # Write the GIF
    clip.write_gif(output_path, fps=fps)

if __name__ == "__main__":
    mov_to_gif("Screen Recording 2025-02-05 at 7.45.47 PM.mov", "gifs/polygons1.gif", fps=5)
    mov_to_gif("Screen Recording 2025-02-05 at 7.48.07 PM.mov", "gifs/polygons2.gif", fps=5)