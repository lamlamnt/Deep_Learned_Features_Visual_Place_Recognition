import numpy as np
from haversine import haversine
import matplotlib.pyplot as plt

#Get GPS ground truth between 2 runs
def gps_ground_truth(reference_run, query_run, reference_len, query_len):
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
    with open(query_path,"r") as file:
        for i in range(query_len):
            line = file.readline().strip()  # Read the first line and remove leading/trailing whitespace
            numbers = line.split(",")  # Split the line by comma
            query_gps[i,0] = numbers[2] #lat
            query_gps[i,1] = numbers[3] #lon
            query_gps[i,2] = numbers[4] #rotation
    
    #Calculate distance 
    for row1,frame1 in enumerate(ref_gps):
        for row2,frame2 in enumerate(query_gps):
            distance = haversine((ref_gps[row1,0],ref_gps[row1,1]),(query_gps[row2,0],query_gps[row2,1]),unit="m")
            gps_distance[row1,row2] = distance

    #Plot
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
    plt.savefig("plots/" + plot_name)
    return gps_distance

def plot_similarity(similarity_run, reference_run, query_run):
    plt.figure()
    plt.title("Difference in descriptors between runs " + str(reference_run) + " and " + str(query_run))
    similarity_plot = plt.imshow(similarity_run, cmap='viridis', interpolation='nearest')
    colorbar = plt.colorbar(similarity_plot)
    colorbar.set_label("Cosine similarity")
    plt.xlabel("Query run frame number")
    plt.ylabel("Reference run frame number")
    plot_name = "Descriptor map for runs " + str(reference_run)+ " and " + str(query_run) + ".png"
    plt.savefig("plots/" + plot_name)

#use gps data to find the total translational error
def rmse_error(gps_ground_truth,max_similarity_idx):
    pass

#Based on localisation data
def calculate_success_rate(max_similarity_idx, reference_run, query_run, query_len):
    que_name = ""
    if(query_run < 10):
        que_name = "0" + str(query_run)
    else:
        que_name = str(query_run)
    path = "/Volumes/oridatastore09/ThirdPartyData/utias/inthedark/run_0000" + que_name + "/transforms_spatial.txt"
    counter = 0
    with open(path,"r") as file:
        for i in range(query_len):
            line = file.readline().strip()  # Read the first line and remove leading/trailing whitespace
            numbers = line.split(",")
            if(max_similarity_idx[i] == numbers[3]):
                counter +=1
    return float(counter/query_len)
