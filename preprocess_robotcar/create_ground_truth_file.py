import os 
import transform_tools
import numpy as np
import sys
sys.path.append("/home/lamlam/downloads/robotcar-dataset-sdk/python")
import transform

#Iterate through each se3_grasshopper.txt and create a .txt file that contains the se3 in stereo
def convert_gh_to_stereo(runs):
        for value in runs:
                gh_file_path = os.path.join(root_dir,"run_" + str(value).zfill(6),gh_file_name)
                stereo_file_path = os.path.join(root_dir,"run_" + str(value).zfill(6),stereo_file_name)
                #store the entire file to be written to another file
                #Overwrite the file if it already exists 
                with open(gh_file_path,"r") as input_file,open(stereo_file_path,"w") as output_file:
                        for line in input_file:
                                #Includes the timestamp as the first element
                                se3_full = line.strip().split()
                                se3_gh = se3_full[1:]
                                se3_gh = [float(element) for element in se3_gh]
                                #Reshape the list of 16 into a numpy array of 4x4
                                se3_gh_np = np.reshape(se3_gh,(4,4))
                                #Do the transformation here!!!!
                                se3_stereo = se3_gh_np@rear_extrinsic_seasons
                                str_list = [str(element) for element in se3_stereo.flatten().tolist()]
                                content = " ".join(str_list)
                                output_file.write(se3_full[0] + " " + content + "\n")

#Find the relative se3 and put that into another .txt file
#transforms_spatial.txt in the form run_name, frame_timestamp, run_name, frame_timestamp, se3
def create_relative_file(reference_run, query_runs):
        #Reference run is the identifying name of the reference run (usually by tag). Query runs is a dictionary of runs in the format name: time-stamp
        reference_file_path = os.path.join(root_dir,"run_" + str(reference_run).zfill(6), stereo_file_name)
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
                stereo_file_path = os.path.join(root_dir,"run_" + str(value).zfill(6),stereo_file_name)
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
                                #Convert from bumblee frame to vehicle frame - due to no rotation part - this commented out line doesn't do anything
                                #relative_se3 = np.linalg.inv(T_s_v)@(relative_se3@T_s_v)
                                #Write relative se3 to file and format properly for training
                                str_list = [str(element) for element in relative_se3.flatten().tolist()]
                                content = ",".join(str_list)
                                #Want time stamp of reference frame, and not just index
                                if(abs(float(str_list[11])) < threshold_translate and float(str_list[0]) > threshold_rotate): 
                                        relative_file.write(str(value) + "," + se3_query_timestamp + "," + str(reference_run) + "," + list_timestamps_ref[index_of_closest] + "," + content + "\n")

def create_transform_temporal(reference_run):
        reference_file_path = os.path.join(root_dir,"run_" + str(reference_run).zfill(6), stereo_file_name)
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
                        str_list = [str(element) for element in relative_se3.flatten().tolist()]
                        content = ",".join(str_list)
                        #Want time stamp of reference frame, and not just index 
                        if(abs(float(str_list[11])) < threshold_translate): 
                                output_file.write(str(reference_run) + "," + list_timestamps_ref[i] + "," + str(reference_run) + "," + list_timestamps_ref[i+1] + "," + content + "\n")

if __name__ == '__main__':
        #Set parameters
        root_dir = "/Volumes/scratchdata/lamlam/processed_data/robotcar_seasons"
        runs = [0,1,2,3,4,5,6,7,8]
        gh_file_name = "se3_grasshopper_2.txt"
        stereo_file_name = "se3_stereo.txt"
        threshold_translate = 2.0
        threshold_rotate = -0.8

        #Using the data from sdk instead of from seasons
        xyzrpy = np.array([-2.0582, 0.0894, 0.3675, -0.0119, -0.2498, 3.1283])
        rear_extrinsic_seasons = np.asarray(transform.build_se3_transform(xyzrpy))
        """
        rear_extrinsic_seasons = np.array([[-0.999802, -0.011530, -0.016233, 0.060209],
                                   [-0.015184, 0.968893, 0.247013, 0.153691],
                                    [0.012880, 0.247210, -0.968876, -2.086142],
                                    [0.000000, 0.000000, 0.000000, 1.000000]])
        """
        #Transformation matrix that transforms a point from vehicle frame to sensor frame
        T_s_v = np.array([[1,0,0,0],
                          [0,1,0,0],
                          [0,0,1,1.52],
                          [0,0,0,1]])

        convert_gh_to_stereo(runs)

        reference_run = 0
        
        relative_file_name = "transforms_spatial.txt"
        del runs[0]
        
        create_relative_file(reference_run,runs)

        temporal_file_name = "transforms_temporal.txt"
        create_transform_temporal(reference_run)


