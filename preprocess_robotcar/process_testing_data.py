import numpy as np
import os
import shutil
import re

def get_gps_list(path):
    gps_timestamps = []
    gps_lat = []
    gps_long = []
    northing = []
    easting = []
    down = []
    roll = []
    pitch = []
    yaw = []
    with open(path,"r") as file:
        file.readline()
        for line in file:
            data = line.strip().split(",")
            gps_timestamps.append(int(data[0]))
            gps_lat.append(data[1])
            gps_long.append(data[2])
            northing.append(data[4])
            easting.append(data[5])
            down.append(data[6])
            roll.append(data[11])
            pitch.append(data[12])
            yaw.append(data[13])
    return gps_timestamps,gps_lat,gps_long, northing, easting, down, roll, pitch, yaw


if __name__ == '__main__':
    #Set parameters and path
    threshold = 8000
    runs = ["2014-11-14-16-34-33","2014-11-18-13-20-12","2014-12-02-15-30-08","2014-12-16-18-44-24","2015-08-28-09-50-22","2015-10-29-12-18-17"]
    runs_numbering = {"2014-11-14-16-34-33":9,"2014-11-18-13-20-12":10,"2014-12-02-15-30-08":11,"2014-12-16-18-44-24":12,"2015-08-28-09-50-22":13,"2015-10-29-12-18-17":14}
    root_dir = "/Volumes/scratchdata/lamlam"
    rtk_dir = "/Volumes/scratchdata/lamlam/rtk"
    for run in runs:
        print(run)
        run_path = os.path.join(root_dir,run,"stereo/left")
        run_timestamps = sorted(os.listdir(run_path))
        run_timestamps = [int(file_name.replace(".png", "")) for file_name in run_timestamps]
        gps_path = os.path.join(rtk_dir,run,"rtk.csv")
        gps_timestamps, gps_lat, gps_long, northing, easting, down, roll, pitch, yaw = get_gps_list(gps_path)
        chosen_frames = []
        chosen_lat = []
        chosen_long = []
        chosen_northing = []
        chosen_easting = []
        chosen_down = []
        chosen_roll = []
        chosen_pitch = []
        chosen_yaw = []
        for idx,time in enumerate(gps_timestamps):
            #closest frame is a frame in run_timestamps
            closest_frame = min(run_timestamps, key=lambda x:abs(x-time))
            if(abs(closest_frame - time) < threshold):
                chosen_frames.append(closest_frame)
                chosen_lat.append(gps_lat[idx])
                chosen_long.append(gps_long[idx])
                chosen_northing.append(northing[idx])
                chosen_easting.append(easting[idx])
                chosen_down.append(down[idx])
                chosen_roll.append(roll[idx])
                chosen_pitch.append(pitch[idx])
                chosen_yaw.append(yaw[idx])
        #Create folder and Move the chosen frames into a folder
        run_name = "run_" + str(runs_numbering[run]).zfill(6)
        new_folder_path_left = os.path.join(root_dir,"processed_data/robotcar_rtk_full",run_name,"images/left")
        new_folder_path_right = os.path.join(root_dir,"processed_data/robotcar_rtk_full",run_name,"images/right")
        new_folder_path_centre = os.path.join(root_dir,"processed_data/robotcar_rtk_full",run_name,"images/centre")
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
        new_rtk_path = os.path.join(root_dir,"processed_data/robotcar_rtk_full",run_name,"gps.txt")
        with open(new_rtk_path,"w") as file:
            for idx,frame in enumerate(chosen_frames):
                file.write(str(runs_numbering[run]) + "," + str(chosen_frames[idx]) + "," + chosen_lat[idx] + "," + chosen_long[idx] +"," + chosen_northing[idx] + "," + chosen_easting[idx]+"," + chosen_down[idx] + "," + chosen_roll[idx]+"," + chosen_pitch[idx]+ "," + chosen_yaw[idx] + "\n")


