import os
import cv2
from tqdm import tqdm

#image_folder = "/Volumes/scratchdata/lamlam/processed_data/robotcar_rtk/run_000011/images/left" 
#output_video = "/home/lamlam/data/video/run_000011.mp4"
image_folder = "/Volumes/scratchdata/lamlam/2014-12-02-15-30-08/stereo/left"
output_video = "/home/lamlam/data/video/run_000011_full.mp4"

images = [img for img in os.listdir(image_folder)]  
images.sort()
frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can change the codec if needed
video = cv2.VideoWriter(output_video, fourcc, 1, (width, height))
for image in tqdm(images):
    video.write(cv2.imread(os.path.join(image_folder, image)))

cv2.destroyAllWindows()
video.release()