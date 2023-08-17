import numpy as np
import sys
sys.path.append("/home/lamlam/downloads/robotcar-dataset-sdk/python")
import transform
import os
import transform_tools

#Get se3 matrix in rtk frame
def create_se3_rtk_file():
    for run in runs:
        run_name =  "run_" + str(run).zfill(6)
        rtk_path = os.path.join(root_dir,run_name,"gps.txt")
        se3_path = os.path.join(root_dir,run_name,rtk_file_name)
        with open(rtk_path,"r") as input_file, open(se3_path,"w") as output_file:
            for line in input_file:
                data = line.strip().split(",")
                r = float(data[7])
                p = float(data[8])
                y = float(data[9])
                #Get the 3x3 rotation matrix
                so3 = np.array(transform.euler_to_so3([r,p,y]))
                #Concatenate with translation element
                translation = np.vstack([float(data[5]),float(data[4]),float(data[6])])
                buffer = np.array([0.0,0.0,0.0,1.0])
                se3 = np.hstack((so3,translation))
                se3_final = np.vstack((se3,buffer))

                #Convert from Robotcar to UTIAS vehicle frame
                se3_final = se3_final@T_R_V

                #Write to se3_rtk file (space between elements)
                str_list = ["{:.6f}".format(element) for element in se3_final.flatten().tolist()]
                content = " ".join(str_list)
                output_file.write(data[1] + " " + content + "\n")

#Get T_R2_R1
def create_relative_file(reference_run, query_runs):
        #Reference run is the identifying name of the reference run (usually by tag). Query runs is a dictionary of runs in the format name: time-stamp
        reference_file_path = os.path.join(root_dir,"run_" + str(reference_run).zfill(6), rtk_file_name)
        #Get the ref se3 matrices in a list
        list_se3_ref = []
        list_timestamps_ref = []
        with open(reference_file_path,"r") as reference_file:
                for line_ref in reference_file:
                        se3_ref = line_ref.strip().split()
                        list_timestamps_ref.append(se3_ref[0])
                        se3_ref = se3_ref[1:]
                        se3_ref = [float(element) for element in se3_ref]
                        se3_ref = np.reshape(se3_ref,(4,4))
                        list_se3_ref.append(se3_ref)
        for value in query_runs:
                stereo_file_path = os.path.join(root_dir,"run_" + str(value).zfill(6),rtk_file_name)
                relative_file_path = os.path.join(root_dir,"run_" + str(value).zfill(6),relative_file_name)
                with open(stereo_file_path,"r") as query_file,open(relative_file_path,"w") as relative_file:
                        for line in query_file:
                                #Get the se3 matrix
                                se3_query = line.strip().split()
                                se3_query_timestamp = se3_query[0]
                                se3_query_matrix = se3_query[1:]
                                se3_query_matrix = [float(element) for element in se3_query_matrix]
                                se3_query_matrix = np.reshape(se3_query_matrix,(4,4))

                                #Iterate through all frames in reference run to find the closest frame in reference run 
                                #to the current frame in query run (many different methods to do this)
                                index_of_closest = transform_tools.find_closest_se3_index(se3_query_matrix,list_se3_ref)

                                #Calculate the relative se3 transformation. Np array
                                relative_se3 = np.linalg.inv(list_se3_ref[index_of_closest])@se3_query_matrix
                                
                                #Write relative se3 to file and format properly for training
                                str_list = ["{:.6f}".format(element) for element in relative_se3.flatten().tolist()]
                                content = ",".join(str_list)
                                #Want time stamp of reference frame, and not just index
                                #if(abs(float(str_list[3])) < threshold_translate and float(str_list[0]) > threshold_rotate): 
                                relative_file.write(str(value) + "," + se3_query_timestamp + "," + str(reference_run) + "," + list_timestamps_ref[index_of_closest] + "," + content + "\n")

def create_transform_temporal(reference_run):
        reference_file_path = os.path.join(root_dir,"run_" + str(reference_run).zfill(6), rtk_file_name)
        #Get the ref se3 matrices in a list
        list_se3_ref = []
        list_timestamps_ref = []
        with open(reference_file_path,"r") as reference_file:
                for line_ref in reference_file:
                        se3_ref = line_ref.strip().split()
                        list_timestamps_ref.append(se3_ref[0])
                        se3_ref = se3_ref[1:]
                        se3_ref = [float(element) for element in se3_ref]
                        se3_ref = np.reshape(se3_ref,(4,4))
                        list_se3_ref.append(se3_ref)
        temporal_file_path = os.path.join(root_dir,"run_" + str(reference_run).zfill(6),temporal_file_name)
        #Transformation matrix from previous to next
        with open(temporal_file_path,"w") as output_file:
                for i in range(len(list_se3_ref)-1):
                        #i is the previous index
                        relative_se3 = np.linalg.inv(list_se3_ref[i+1])@list_se3_ref[i]
                        #Convert from bumblee frame to vehicle frame - due to no rotation part - this commented out line doesn't do anything
                        #relative_se3 = np.linalg.inv(T_s_v)@(relative_se3@T_s_v)
                        str_list = ["{:.6f}".format(element) for element in relative_se3.flatten().tolist()]
                        content = ",".join(str_list)
                        #Want time stamp of reference frame, and not just index 
                        #if(abs(float(str_list[3])) < threshold_translate and float(str_list[0]) > threshold_rotate): 
                        output_file.write(str(reference_run) + "," + list_timestamps_ref[i] + "," + str(reference_run) + "," + list_timestamps_ref[i+1] + "," + content + "\n")

if __name__ == '__main__':
    root_dir = "/Volumes/scratchdata/lamlam/processed_data/robotcar_rtk_full"
    #rtk_file_name = "se3_rtk.txt"
    rtk_file_name = "se3_rtk_interpolated.txt"
    runs = [9,0,11,12,13,14]
    #runs = [9,0]
    threshold_translate = 5.0
    threshold_rotate = -0.5

#The inverse of this is also the same. Switch y and z to negative
    T_R_V = np.array([[1,0,0,0],
                [0,-1,0,0],
                [0,0,-1,0],
                [0,0,0,1]])

    #create_se3_rtk_file()
    
    reference_run = 0
    del runs[1]
    #relative_file_name = "transforms_spatial.txt"
    relative_file_name = "transforms_spatial_interpolated.txt"
    create_relative_file(reference_run,runs)

    #temporal_file_name = "transforms_temporal.txt"
    temporal_file_name = "transforms_temporal_interpolated.txt"
    create_transform_temporal(reference_run)


    