from moviepy.editor import ImageSequenceClip
import shutil

# To create a video from the images
def make_video(fps, path, video_file):
    print("Creating video {}".format(video_file))
    clip = ImageSequenceClip(path, fps = fps)
    clip.write_videofile(video_file)
    # shutil.rmtree(path)