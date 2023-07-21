import os 
import transform_tools
import numpy as np

#Iterate through each se3_grasshopper.txt and create a .txt file that contains the se3 in stereo
def convert_gh_to_stereo(runs):
        for key, value in runs.items():
                gh_file_path = os.path.join(root_dir,key + "_" + value,gh_file_name)
                stereo_file_path = os.path.join(root_dir,key + "_" + value,stereo_file_name)
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
                                se3_stereo = transform_tools.gh_to_stereo(se3_gh_np)
                                str_list = [str(element) for element in se3_stereo.flatten().tolist()]
                                content = " ".join(str_list)
                                output_file.write(se3_full[0] + " " + content + "\n")

#Find the relative se3 and put that into another .txt file
#transforms_spatial.txt in the form run_name, frame_timestamp, run_name, frame_timestamp, se3
def create_relative_file(reference_run, reference_run_timestamps,query_runs):
        #Reference run is the identifying name of the reference run (usually by tag). Query runs is a dictionary of runs in the format name: time-stamp
        reference_file_path = os.path.join(root_dir,reference_run + "_" + reference_run_timestamps, stereo_file_name)
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
        for key, value in query_runs.items():
                stereo_file_path = os.path.join(root_dir,key + "_" + value,stereo_file_name)
                relative_file_path = os.path.join(root_dir,key + "_" + value,relative_file_name)
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
                                relative_se3 = transform_tools.get_relative(se3_query_matrix,list_se3_ref[index_of_closest])
                                #Write relative se3 to file and format properly for training
                                str_list = [str(element) for element in relative_se3.flatten().tolist()]
                                content = ",".join(str_list)
                                #Want time stamp of reference frame, and not just index 
                                relative_file.write(value + "," + se3_query_timestamp + "," + reference_run_timestamp + "," + list_timestamps_ref[index_of_closest] + "," + content + "\n")

if __name__ == '__main__':
        #Set parameters
        root_dir = "/Volumes/scratchdata/lamlam/processed_data"
        #runs = {"dawn":"2014-12-16-09-14-09", "dusk":"2015-02-20-16-34-06", "night":"2014-12-10-18-10-50", 
        #"night-rain":"2014-12-17-18-18-43", "overcast-summer":"2015-05-22-11-14-30", "overcast-winter":"2015-11-13-10-28-08", 
        #"rain":"2014-11-25-09-18-32", "snow":"2015-02-03-08-45-10", "sun":"2015-03-10-14-18-10"}
        runs = {"dawn":"2014-12-16-09-14-09","sun":"2015-03-10-14-18-10"}
        gh_file_name = "se3_grasshopper.txt"
        stereo_file_name = "se3_stereo.txt"

        #convert_gh_to_stereo(runs)

        reference_run = "sun"
        reference_run_timestamp = runs[reference_run]
        
        relative_file_name = "se3_relative_to_" + reference_run + "_" + runs[reference_run] + ".txt"
        del runs[reference_run]
        
        create_relative_file(reference_run,reference_run_timestamp,runs)


