import numpy as np
from haversine import haversine
import matplotlib.pyplot as plt

#Given gps of run 0 and localized frames AND gps of query run and chosen frames, calculate success rate and average distance error
gps_0 = np.loadtxt("/home/lamlam/code/visual_place_recognition/clustering/ref_gps.txt")
gps_query = np.loadtxt("/home/lamlam/code/visual_place_recognition/clustering/query_gps.txt")
localized_frames = np.loadtxt("/home/lamlam/code/visual_place_recognition/clustering/localized_frames.txt",dtype=int)
chosen_frames = np.loadtxt("/home/lamlam/code/visual_place_recognition/clustering/chosen_frames.txt",dtype=int)

def calculate_success_rate_list(gps_0,localized_frames,chosen_frames, threshold_list):
    success = np.zeros((len(localized_frames),len(threshold_list)),dtype=int)
    sum = 0
    distance_frame = []
    localized_frames = np.where(localized_frames == 558, 557, localized_frames)
    for i in range(len(localized_frames)):
        distance = haversine((gps_0[chosen_frames[i],0], gps_0[chosen_frames[i],1]),(gps_0[localized_frames[i],0], gps_0[localized_frames[i],1]),unit="m")
        distance_frame.append(distance)
        print("Query frame num: " + str(i) + " Localized frame num: " + str(localized_frames[i]) + " Chosen frame num: " + str(chosen_frames[i]))
        print("Distance is " + str(distance))
        for threshold_idx,threshold in enumerate(threshold_list):
            if(distance <= threshold):
                success[i,threshold_idx] = 1
        sum += distance
    plot_distance(distance_frame)
    #Localized frame to run 0 
    return np.sum(success,axis=0)/len(chosen_frames),float(sum/len(chosen_frames))

def plot_distance(distance_frame):
    plt.figure()
    plt.title("Distance error for runs localized 0 and 16")
    plt.plot(distance_frame)
    plt.xlabel("Query run frame number")
    plt.ylabel("Meters")
    plot_name = "Distance_error_localized_gps.png"
    plt.savefig("/home/lamlam/code/visual_place_recognition/clustering/plot_visualize_cluster/" + plot_name)

threshold_list = [5,10]
success_rate, average_error = calculate_success_rate_list(gps_0,localized_frames,chosen_frames, threshold_list)
for idx,rate in enumerate(success_rate):
    print("Success rate at threshold " + str(threshold_list[idx]) + "m is " + str(rate))
print("Average error in meters: " + str(average_error))
