from moviepy.editor import ImageSequenceClip
import shutil

def make_video(fps, path, video_file):
    """
    To make video from the images stored in an entire folder
    
    Inputs:
    fps : frame per second for the video
    path: folder path in which sequence of images are stored
    video_file: name of the video top be saved

    """
    print("****************************************")
    print("Creating video {}, FPS={}".format(video_file, fps))
    clip = ImageSequenceClip(path, fps = fps)
    clip.write_videofile(video_file)
    # shutil.rmtree(path)