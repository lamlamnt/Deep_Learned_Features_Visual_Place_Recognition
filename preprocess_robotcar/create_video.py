import os
import cv2

def create_video(run_number):
    # Path to the folder containing the images
    run_str = str(run_number).zfill(6)
    image_folder = "/Volumes/scratchdata/lamlam/processed_data/robotcar_rtk_full/run_" + run_str + "/images/left"
    output_video_path = "/home/lamlam/data/video/rtk_run_" + run_str + ".mp4"

    # Get the list of image filenames in the folder
    images = [img for img in os.listdir(image_folder)]

    # Sort the images based on their filenames
    images.sort()

    # Create a VideoWriter object to save the video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Specify the codec
    video = cv2.VideoWriter(output_video_path, fourcc, 1, (512, 384))

    # Loop through the images and write them to the video
    for image in images:
        img_path = os.path.join(image_folder, image)
        frame = cv2.imread(img_path)
        video.write(frame)

    # Release the VideoWriter and close the video file
    video.release()

#Trace out the gps path
def trace_gps():
    pass