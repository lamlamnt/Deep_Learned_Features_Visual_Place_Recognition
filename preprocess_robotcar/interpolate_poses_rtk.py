import sys
sys.path.append("/home/lamlam/downloads/robotcar-dataset-sdk/python")
import interpolate_poses
import numpy as np

def get_interpolated_poses_rtk():
    for key,value in runs_numbering.items():
        #Replace se3_rtk.txt
        ins_path = "/Volumes/scratchdata/lamlam/rtk/" + key + "/rtk.csv"

        #Get pose_timestamps from the file se3_rtk
        run_str = str(value).zfill(6)
        timestamp_path = "/Volumes/scratchdata/lamlam/processed_data/robotcar_rtk_full/run_" + run_str + "/se3_rtk.txt"
        pose_timestamp = []
        with open(timestamp_path,"r") as input_file:
            for line in input_file:
                data = line.strip().split()
                pose_timestamp.append(int(data[0]))
        #Origin_timestamp. 
        #Note: The first line in each rtk.csv file is added using a point in reference run (dummy point) because we want all frames to be 
        #relative to the same location and not just to the first frame in each run, which will be slightly different locations
        origin_timestamp = pose_timestamp[0]
        se3_list = interpolate_poses.interpolate_ins_poses(ins_path,pose_timestamp,origin_timestamp,use_rtk=True)
        #Remove the first element from the list (extra)
        pose_timestamp.pop(0)

        #Write the se3_list to a file
        #Convert to numpy arrays and convert to vehicle frame and save to file
        se3_file_path = "/Volumes/scratchdata/lamlam/processed_data/robotcar_rtk_full/run_" + run_str + "/se3_rtk_interpolated.txt"
        with open(se3_file_path,"w") as output_file:
            for idx,pose in enumerate(se3_list):
                #Convert from Robotcar to UTIAS vehicle frame
                se3_final = np.array(pose)@T_R_V
                str_list = ["{:.6f}".format(element) for element in se3_final.flatten().tolist()]
                content = " ".join(str_list)
                output_file.write(str(pose_timestamp[idx]) + " " + content + "\n")

if __name__ == '__main__':
    T_R_V = np.array([[1,0,0,0],
                    [0,-1,0,0],
                    [0,0,-1,0],
                    [0,0,0,1]])
    runs_numbering = {"2014-11-14-16-34-33":9,"2014-11-18-13-20-12":0,"2014-12-02-15-30-08":11,"2014-12-16-18-44-24":12,"2015-08-28-09-50-22":13,"2015-10-29-12-18-17":14}
    get_interpolated_poses_rtk()
    