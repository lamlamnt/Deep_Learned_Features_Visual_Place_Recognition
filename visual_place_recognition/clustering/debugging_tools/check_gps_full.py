import numpy as np
from haversine import haversine
import matplotlib.pyplot as plt
#Get gps of run 0 and dark run and localized frames
#Find distance error and plot
#14-19
def get_gps(run):
    name = ""
    if(run < 10):
        name = "0" + str(run)
    else:
        name = str(run)
    path ="/Volumes/oridatastore09/ThirdPartyData/utias/inthedark/run_0000" + name + "/gps.txt"
    #Get line number
    with open(path, 'r') as file:
        line_count = sum(1 for _ in file)
    gps = np.zeros((line_count,3),dtype=float)
    with open(path, "r") as file:
        for i in range(line_count):
            line = file.readline().strip()  # Read the first line and remove leading/trailing whitespace
            numbers = line.split(",")  # Split the line by comma
            gps[i,0] = numbers[2] #lat
            gps[i,1] = numbers[3] #lon
            gps[i,2] = numbers[4]
    print(line_count)
    return gps, line_count

def get_localized_frames(query_run, query_len):
    if(query_run < 10):
        name = "0" + str(query_run)
    else:
        name = str(query_run)
    transform_path = "/Volumes/oridatastore09/ThirdPartyData/utias/inthedark/run_0000" + name + "/transforms_spatial.txt"
    localized_frames = np.zeros((query_len),dtype=int)
    with open(transform_path, 'r') as file:
        for i in range(query_len):
            line = file.readline().strip()  # Read the first line and remove leading/trailing whitespace
            numbers = line.split(",")  # Split the line by comma
            localized_frames[i] = numbers[3]
    return localized_frames

def compare_distance(gps_0,gps_query, localized_frames):
    distance_full = np.zeros(len(gps_query))
    for i in range(len(localized_frames)):
        distance = haversine((gps_0[localized_frames[i],0], gps_0[localized_frames[i],1]),(gps_query[i,0], gps_query[i,1]),unit="m")
        distance_full[i] = distance
    average_distance = np.mean(distance_full)
    maximum = np.max(distance_full)
    return distance_full, average_distance,maximum

def plot_distance(distance, query_run):
    plt.title("Distance error for runs 0 and " + str(query_run) + " (using localized frames)")
    plt.plot(distance)
    plt.xlabel("Query run frame number")
    plt.ylabel("Meters")
    plot_name = "Distance_error_localized_frames_" + str(query_run) + ".png"
    plt.savefig("/home/lamlam/code/visual_place_recognition/clustering/plot_check_gps_accuracy/" + plot_name)

runs = [1,2,5,6,9,10,11,12,13,14,15,16,17,18,19,25,26,27,28,29,30,31]
for run in runs:
    query_run = run
    gps_0, len_0 = get_gps(0)
    gps_query, len_query = get_gps(query_run)
    localized_frames = get_localized_frames(query_run, len_query)
    distance_full,average_distance,maximum = compare_distance(gps_0,gps_query,localized_frames)
    plot_distance(distance_full, query_run)
    print(str(run) + ": " + str(average_distance) + " " + str(maximum))

