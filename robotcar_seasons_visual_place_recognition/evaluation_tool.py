import numpy as np
from haversine import haversine
import matplotlib.pyplot as plt

#Get GPS ground truth between 2 runs
def gps_ground_truth(reference_run, query_run, reference_len, query_len, increment_ref, increment_que):
    #The smaller number of frames of the two runs is used
    #Vertical axis is reference run. Horizontal axis is query run.
    ref_name = ""
    if(reference_run < 10):
        ref_name = "0" + str(reference_run)
    else:
        ref_name = str(reference_run)
    que_name = ""
    if(query_run < 10):
        que_name = "0" + str(query_run)
    else:
        que_name = str(query_run)
    reference_path ="/Volumes/oridatastore09/ThirdPartyData/utias/inthedark/run_0000" + ref_name + "/gps.txt"
    query_path = "/Volumes/oridatastore09/ThirdPartyData/utias/inthedark/run_0000" + que_name + "/gps.txt"
    gps_distance = np.zeros((reference_len, query_len),dtype=float)
    ref_gps = np.zeros((reference_len,3),dtype=float)
    query_gps = np.zeros((query_len,3),dtype=float)
    
    #Get gps data
    with open(reference_path, "r") as file:
        for i in range(reference_len):
            line = file.readline().strip()  # Read the first line and remove leading/trailing whitespace
            numbers = line.split(",")  # Split the line by comma
            ref_gps[i,0] = numbers[2] #lat
            ref_gps[i,1] = numbers[3] #lon
            ref_gps[i,2] = numbers[4]  #rotation
            for j in range(increment_ref-1):
                file.readline()
    with open(query_path,"r") as file:
        for i in range(query_len):
            line = file.readline().strip()  # Read the first line and remove leading/trailing whitespace
            numbers = line.split(",")  # Split the line by comma
            query_gps[i,0] = numbers[2] #lat
            query_gps[i,1] = numbers[3] #lon
            query_gps[i,2] = numbers[4] #rotation
            for j in range(increment_que-1):
                file.readline()
    #Calculate distance 
    for row1,frame1 in enumerate(ref_gps):
        for row2,frame2 in enumerate(query_gps):
            distance = haversine((ref_gps[row1,0],ref_gps[row1,1]),(query_gps[row2,0],query_gps[row2,1]),unit="m")
            gps_distance[row1,row2] = distance

    #Plot the gps difference between frames
    plt.figure()
    #Max distance is around 90m
    plt.title("GPS ground truth for reference run " + str(reference_run) + " and query run " + str(query_run))
    #cmap: twilight
    gps_plot = plt.imshow(gps_distance, cmap='viridis', interpolation='nearest')
    colorbar = plt.colorbar(gps_plot)
    colorbar.set_label("In meters")
    plt.xlabel("Query run frame number")
    plt.ylabel("Reference run frame number")
    plot_name = "gps_ground_truth for runs " + str(reference_run)+ " and " + str(query_run) + ".png"
    plt.savefig("/home/lamlam/code/visual_place_recognition/plots/" + plot_name)

    """
    #Plot the gps actual values to check for accuracy
    plt.figure()
    plt.title("GPS latitude and longitude for reference run " + str(reference_run) + " and query run " + str(query_run))
    plt.scatter(ref_gps[:,0],ref_gps[:,1],s=5,marker = ".")
    plt.scatter(query_gps[:,0],query_gps[:,1],s=5,marker = ".")
    plt.xlabel("Latitude")
    plt.ylabel("Longitude")
    plot_name = "gps for runs " + str(reference_run)+ " and " + str(query_run) + ".png"
    plt.savefig("plots/" + plot_name)
    """
    #Get the max distance difference:
    #np.savetxt("/home/lamlam/code/visual_place_recognition/clustering/query_gps.txt",query_gps)
    #np.savetxt("/home/lamlam/code/visual_place_recognition/clustering/ref_gps.txt",ref_gps)
    return gps_distance, ref_gps, query_gps

def plot_similarity(similarity_run, reference_run, query_run, sampling_method):
    plt.figure()
    plt.title("Difference in descriptors between runs " + str(reference_run) + " and " + str(query_run) + " - " + sampling_method)
    similarity_plot = plt.imshow(similarity_run, cmap='viridis', interpolation='nearest')
    colorbar = plt.colorbar(similarity_plot)
    colorbar.set_label("Cosine similarity")
    plt.xlabel("Query run frame number")
    plt.ylabel("Reference run frame number")
    plot_name = "Descriptor map for runs " + str(reference_run)+ " and " + str(query_run) + " - " + sampling_method + ".png"
    plt.savefig("/home/lamlam/code/visual_place_recognition/plots" + plot_name)

def plot_similarity_clustering(similarity_run, reference_run, query_run, method):
    plt.figure()
    plt.title("Similarity between runs " + str(reference_run) + " and " + str(query_run))
    similarity_plot = plt.imshow(similarity_run, cmap='viridis', interpolation='nearest')
    colorbar = plt.colorbar(similarity_plot)
    colorbar.set_label(method)
    plt.xlabel("Query run frame number")
    plt.ylabel("Reference run frame number")
    plot_name = "Similarity clustering map for runs " + str(reference_run)+ " and " + str(query_run) + ".png"
    plt.savefig("/home/lamlam/code/visual_place_recognition/plots/" + plot_name)

def plot_distance(distance_frame, reference_run, query_run):
    plt.figure()
    plt.title("Distance error for runs " + str(reference_run) + " and " + str(query_run))
    plt.plot(distance_frame)
    plt.xlabel("Query run frame number")
    plt.ylabel("Meters")
    plot_name = "Distance_error_" + str(reference_run) + "_" + str(query_run) + ".png"
    plt.savefig("/home/lamlam/code/visual_place_recognition/clustering/plot_visualize_cluster/" + plot_name)

def calculate_success_rate_list(max_similarity_idx, ref_gps, query_gps, threshold_list, reference_run, query_run, incre_ref,incre_que):
    success = np.zeros((len(max_similarity_idx),len(threshold_list)),dtype=int)
    sum = 0
    distance_frame = []
    for index,value in enumerate(max_similarity_idx):
        distance = haversine((ref_gps[value,0], ref_gps[value,1]),(query_gps[index,0], query_gps[index,1]),unit="m")
        distance_frame.append(distance)
        print("Query frame num: " + str(index) + " Reference frame num: " + str(value))
        print("Distance is " + str(distance))
        for threshold_idx,threshold in enumerate(threshold_list):
            if(distance <= threshold):
                success[index,threshold_idx] = 1
        sum += distance
    #Returns the success rate and the average distance error
    if(query_run != 0):
        plot_distance(distance_frame,reference_run, query_run)
        #Localized frame to run 0 
        localized_frame = get_localized_frame(query_run,len(query_gps),incre_ref,incre_que)
        plot_chosen_frame(max_similarity_idx,localized_frame,reference_run,query_run)
    return np.sum(success,axis=0)/len(max_similarity_idx),float(sum/len(max_similarity_idx))

def plot_scores(scores):
    plt.figure()
    plt.title("Scores Histogram")
    plt.hist(scores,bins=20)
    plt.xlabel("Scores")
    plt.ylabel("Frequency")
    plt.savefig("plots/" + "scores_histogram_final")

def plot_chosen_frame(indices, localized_frame, reference_run, query_run):
    #Plot the chosen frames 
    plt.figure()
    plt.title("Chosen reference frame for runs " + str(reference_run) + " and " + str(query_run))
    plt.plot(indices)
    if(reference_run == 0):
        plt.plot(localized_frame)
    plt.xlabel("Query run")
    plt.ylabel("Reference run")
    plot_name = "Localisation_" + str(reference_run) + "_" + str(query_run) + ".png"
    plt.savefig("/home/lamlam/code/visual_place_recognition/clustering/plot_visualize_cluster/" + plot_name)

def plot_singular_values(S):
    plt.title("Magnitude of singular values")
    plt.plot(S)
    plt.xlabel("Nth singular value in descending order")
    plt.ylabel("Magnitude of singular value")
    plot_name = "Magnitude_Singular_Values.png"
    plt.savefig("/home/lamlam/code/visual_place_recognition/clustering/plot_visualize_cluster/" + plot_name)

def get_localized_frame(query_run,query_len, incre_ref,incre_que):
    if(query_run < 10):
        name = "0" + str(query_run)
    else:
        name = str(query_run)
    transform_path = "/Volumes/oridatastore09/ThirdPartyData/utias/inthedark/run_0000" + name + "/transforms_spatial.txt"
    localized_frames = np.zeros(query_len)
    with open(transform_path, 'r') as file:
        for i in range(query_len):
            line = file.readline().strip()  # Read the first line and remove leading/trailing whitespace
            numbers = line.split(",")  # Split the line by comma
            localized_frames[i] = numbers[3]
            for j in range(incre_que-1):
                file.readline()
    localized_frames= localized_frames/incre_ref
    localized_frames = np.round(localized_frames).astype(int)
    np.savetxt("/home/lamlam/code/visual_place_recognition/clustering/localized_frames.txt",localized_frames)
    return localized_frames


