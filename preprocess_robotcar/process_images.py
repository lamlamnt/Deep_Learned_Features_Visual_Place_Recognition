import sys
from PIL import Image
from colour_demosaicing import demosaicing_CFA_Bayer_bilinear as demosaic
import numpy as np
import os

def convert_bayer_to_rgb(bayer, full_image_path):
    rgb_image = demosaic(bayer, 'gbrg')
    rgb_image = np.array(rgb_image).astype(np.uint8)
    rgb_image = Image.fromarray(rgb_image)
    #Delete the old image and save the new image
    os.remove(full_image_path)
    rgb_image.save(full_image_path)

#Reduce resolution of images
def reduce_resolution(image, full_image_path):
    resized_image = image.resize((new_width,new_height),Image.Resampling.LANCZOS)
    os.remove(full_image_path)
    resized_image.save(full_image_path)

#Iterate through each image in the folder (both rtk and seasons), convert from Bayer to RGB OR reduce resolution, delete old image, and save new RGB image
def process_in_bulk(root_dir, process):
    sub_folder = os.listdir(root_dir)
    for folder in sub_folder:
        #folder is a string of the folder (robotcar_rtk or robotcar_seasons)
        runs = sorted(os.listdir(os.path.join(root_dir,folder)))
        for run in runs:
            print(run)
            path_folder = sorted(os.listdir(os.path.join(root_dir,folder,run,"images")))
            for camera in path_folder:
                full_path = os.path.join(root_dir,folder,run,"images",camera)
                images_list = sorted(os.listdir(full_path))
                for image in images_list:
                    full_image_path = os.path.join(root_dir,folder,run,"images",camera,image)
                    old_image = Image.open(full_image_path)
                    if(process == "bayer_to_rgb"):
                        convert_bayer_to_rgb(old_image, full_image_path)
                    if(process == "reduce_resolution"):
                        reduce_resolution(old_image,full_image_path)

if __name__ == '__main__':
    #process_in_bulk("/Volumes/scratchdata/lamlam/processed_data", "bayer_to_rgb")
    new_width = 512
    new_height = 384
    process_in_bulk("/Volumes/scratchdata/lamlam/backup_coloured_full_resolution/processed_data","reduce_resolution")
