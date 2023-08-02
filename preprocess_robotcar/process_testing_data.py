import numpy as np
import os
import shutil

def get_gps_list(path):
    gps_timestamps = []
    gps_lat = []
    gps_long = []
    with open(path,"r") as file:
        file.readline()
        for line in file:
            data = line.strip().split(",")
            gps_timestamps.append(int(data[0]))
            gps_lat.append(data[1])
            gps_long.append(data[2])
    return gps_timestamps,gps_lat,gps_long


if __name__ == '__main__':
    #Set parameters and path
    threshold = 2000
    runs = ["2014-11-14-16-34-33","2014-11-18-13-20-12","2014-12-02-15-30-08","2014-12-16-18-44-24","2015-08-28-09-50-22","2015-10-29-12-18-17"]
    runs_numbering = {"2014-11-14-16-34-33":9,"2014-11-18-13-20-12":10,"2014-12-02-15-30-08":11,"2014-12-16-18-44-24":12,"2015-08-28-09-50-22":13,"2015-10-29-12-18-17":14}
    root_dir = "/Volumes/scratchdata/lamlam"
    rtk_dir = "/Volumes/scratchdata/lamlam/rtk"
    for run in runs:
        run_path = os.path.join(root_dir,run,"stereo/left")
        run_timestamps = sorted(os.listdir(run_path))
        run_timestamps = [int(file_name.replace(".png", "")) for file_name in run_timestamps]
        gps_path = os.path.join(rtk_dir,run,"rtk.csv")
        gps_timestamps, gps_lat, gps_long = get_gps_list(gps_path)
        chosen_frames = []
        chosen_lat = []
        chosen_long = []
        for idx,time in enumerate(gps_timestamps):
            #closest frame is a frame in run_timestamps
            closest_frame = min(run_timestamps, key=lambda x:abs(x-time))
            if(abs(closest_frame - time) < threshold):
                chosen_frames.append(closest_frame)
                chosen_lat.append(gps_lat[idx])
                chosen_long.append(gps_long[idx])
        #Create folder and Move the chosen frames into a folder
        new_folder_path_left = os.path.join(root_dir,"processed_data/robotcar_rtk",run,"images/left")
        new_folder_path_right = os.path.join(root_dir,"processed_data/robotcar_rtk",run,"images/right")
        new_folder_path_centre = os.path.join(root_dir,"processed_data/robotcar_rtk",run,"images/centre")
        if not os.path.exists(new_folder_path_left):
            os.makedirs(new_folder_path_left)
        if not os.path.exists(new_folder_path_right):
            os.makedirs(new_folder_path_right)
        if not os.path.exists(new_folder_path_centre):
            os.makedirs(new_folder_path_centre)  
        for frame in chosen_frames:
            image_path_left = os.path.join(root_dir,run,"stereo/left", str(frame) + ".png")
            image_path_right= os.path.join(root_dir,run,"stereo/right", str(frame) + ".png")
            image_path_centre = os.path.join(root_dir,run,"stereo/centre", str(frame) + ".png")
            shutil.copy(image_path_left, new_folder_path_left)
            shutil.copy(image_path_right, new_folder_path_right)
            shutil.copy(image_path_centre, new_folder_path_centre)
        #Edit rtk data to have the correct frames
        new_rtk_path = os.path.join(root_dir,"processed_data/robotcar_rtk",run,"gps.txt")
        with open(new_rtk_path,"w") as file:
            for idx,frame in enumerate(chosen_frames):
                file.write(str(runs_numbering[run]) + "," + str(chosen_frames[idx]) + "," + chosen_lat[idx] + "," + chosen_long[idx] + "\n")
    


