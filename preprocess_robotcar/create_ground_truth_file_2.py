import numpy as np
import os

def query_transform_gh_to_stereo(isQuery, runs):
    #For query runs: Iterate through each se3_grasshopper file to get the matrices into np arrays
    for key, value in runs.items():
        gh_file_path = os.path.join(root_dir,key + "_" + value,gh_file_name)
        if(isQuery is True):
            intermediate_file_path = os.path.join(root_dir,key + "_" + value,intermediate_file_query)
        else:
            intermediate_file_path = os.path.join(root_dir,key + "_" + value,intermediate_file_reference)
        #store the entire file to be written to another file
        #Overwrite the file if it already exists 
        with open(gh_file_path,"r") as input_file,open(intermediate_file_path,"w") as output_file:
            for line in input_file:
                #Includes the timestamp as the first element
                se3_full = line.strip().split()
                se3_gh = se3_full[1:]
                se3_gh = [float(element) for element in se3_gh]
                #Reshape the list of 16 into a numpy array of 4x4
                se3_gh_np = np.reshape(se3_gh,(4,4))
                #Do the transformation here (T_PG1)
                if(isQuery is True):
                    se3_stereo = se3_gh_np@rear_extrinsic_seasons
                else:
                    se3_stereo = inv_rear_extrinsic@np.linalg.inv(se3_gh_np)
                str_list = [str(element) for element in se3_stereo.flatten().tolist()]
                content = " ".join(str_list)
                output_file.write(se3_full[0] + " " + content + "\n")

def find_relative(reference_run,query_runs):
    #Uses the associative property of matrix multiplication
    pass

if __name__ == '__main__':
        #Set parameters (might put these into config file)
        root_dir = "/Volumes/scratchdata/lamlam/processed_data/robotcar_seasons"
        combined_runs = {"dawn":"2014-12-16-09-14-09", "dusk":"2015-02-20-16-34-06", "night":"2014-12-10-18-10-50", 
        "night-rain":"2014-12-17-18-18-43", "overcast-summer":"2015-05-22-11-14-30", "overcast-winter":"2015-11-13-10-28-08", 
        "rain":"2014-11-25-09-18-32", "snow":"2015-02-03-08-45-10","snow":"2015-02-03-08-45-10"}
        query_runs = {"dawn":"2014-12-16-09-14-09", "dusk":"2015-02-20-16-34-06", "night":"2014-12-10-18-10-50", 
        "night-rain":"2014-12-17-18-18-43", "overcast-summer":"2015-05-22-11-14-30", "overcast-winter":"2015-11-13-10-28-08", 
        "rain":"2014-11-25-09-18-32", "snow":"2015-02-03-08-45-10"}
        reference_run = {"sun":"2015-03-10-14-18-10"}
        gh_file_name = "se3_grasshopper.txt"
        #File that stores the final ground truth (relative poses between Bumblee) - matrix that transforms from query to reference
        final_file = "transform_spatial.txt"
        intermediate_file_query = "se3_bumble_query.txt"
        intermediate_file_reference = "se3_bumble_reference.txt"

        #Extrinsic matrix that transforms from Bumblee Left to Grasshopper Rear (T_G1B1 = T_G2B2)
        rear_extrinsic_seasons = np.array([[-0.999802, -0.011530, -0.016233, 0.060209],
                                   [-0.015184, 0.968893, 0.247013, 0.153691],
                                    [0.012880, 0.247210, -0.968876, -2.086142],
                                    [0.000000, 0.000000, 0.000000, 1.000000]])
        inv_rear_extrinsic = np.linalg.inv(rear_extrinsic_seasons)

        #Call functions
        query_transform_gh_to_stereo(True,combined_runs)
        query_transform_gh_to_stereo(False,combined_runs)

        #If want to change the reference run to another run, need to rerun just the find_relative() function