import numpy as np
from haversine import haversine
import matplotlib.pyplot as plt

#Run 0 and 16
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
    
    #Get the max distance difference:
    return ref_gps, query_gps

def plot(gps_0,gps_16,gps_2,gps_11):
    #Plot the gps actual values to check for accuracy
    plt.figure()
    plt.title("GPS latitude and longitude")
    colors_0 = np.arange(558)
    colors_16 = np.arange(543)
    colors_2 = np.arange(544)
    colors_11 = np.arange(439)
    plt.scatter(gps_0[:,0],gps_0[:,1],s=5,marker = ".", c=colors_0,cmap='viridis')
    plt.scatter(gps_16[:,0],gps_16[:,1],s=5,marker = ".", c=colors_16,cmap='viridis')
    plt.scatter(gps_2[:,0],gps_2[:,1],s=5,marker = ".", c=colors_2,cmap='viridis')
    plt.scatter(gps_11[:,0],gps_11[:,1],s=5,marker = ".", c=colors_11,cmap='viridis')
    plt.scatter(gps_0[238,0],gps_0[238,1],s=30,marker = "*",c = "red")
    plt.scatter(gps_16[300,0],gps_16[300,1],s=30,marker = "*", c= "green")
    plt.scatter(gps_2[220,0],gps_2[220,1],s=30,marker = "*", c= "blue")
    plt.scatter(gps_11[155,0],gps_11[155,1],s=30,marker = "*", c= "orange")
    cbar = plt.colorbar()
    cbar.set_label('Index')
    plt.xlabel("Latitude")
    plt.ylabel("Longitude")
    plot_name = "check_gps for many runs" + ".png"
    plt.savefig("/home/lamlam/code/visual_place_recognition/clustering/plot_visualize_cluster/" + plot_name)

#gps_0, gps_16 = gps_ground_truth(0, 16, 558, 543, 2, 6)
#gps_2, gps_11 = gps_ground_truth(2, 11, 544, 439, 2, 2)
#plot(gps_0,gps_16,gps_2,gps_11)

#Plot error in distance for run 0 and query run
#Already have the localized frames (before dividing by incre) -> get gps (using line number) and get distance
query_gps = np.loadtxt("/home/lamlam/code/visual_place_recognition/clustering/query_gps.txt",dtype=float)
localized_frames = np.loadtxt("/home/lamlam/code/visual_place_recognition/clustering/localized_frames.txt",dtype=int)
ref_gps = np.loadtxt("/home/lamlam/code/visual_place_recognition/clustering/ref_gps.txt",dtype=float)
distance_error = np.zeros(len(query_gps))
localized_frames = np.where(localized_frames == 558, 557, localized_frames)
for i in range(len(localized_frames)):
    distance = haversine((ref_gps[localized_frames[i],0], ref_gps[localized_frames[i],1]),(query_gps[i,0], query_gps[i,1]),unit="m")
    distance_error[i] = distance
    if(i == 400):
        print(localized_frames[i])
        print(distance)
plt.title("Distance error for runs 0 and 16 (using localized frames)")
plt.plot(distance_error)
plt.xlabel("Query run frame number")
plt.ylabel("Meters")
plot_name = "Distance_error_localized_frames_0_16.png"
plt.savefig("/home/lamlam/code/visual_place_recognition/clustering/plot_visualize_cluster/" + plot_name)
