from haversine import haversine
import numpy as np
import torch
from scipy.stats import wasserstein_distance

def calculate_success_rate(max_similarity_idx, ref_gps, query_gps, threshold):
    #Get the gps data of the the max_similarity_idx in the reference run
    success = np.zeros(len(max_similarity_idx))
    sum = 0
    for index,value in enumerate(max_similarity_idx):
        print(str(ref_gps[value,0]) + " " + str(ref_gps[value,1]))
        print(str(query_gps[index,0]) + " " + str(query_gps[index,1]))
        distance = haversine((ref_gps[value,0], ref_gps[value,1]),(query_gps[index,0], query_gps[index,1]))
        if(distance <= threshold):
            success[index] = 1
        sum += distance
    #Returns the success rate and the average distance error
    return float(np.sum(success)/len(max_similarity_idx)),float(sum/len(max_similarity_idx))

max_similarity_idx = np.array([1,0])
ref_gps = np.array([[43.782160,-79.465760,147.52],[43.781800,-79.464763,153.99]])
query_gps = np.array([[43.782160,-79.465760,147.52],[43.781800,-79.464763,153.99]])
#success_rate, error = calculate_success_rate(max_similarity_idx,ref_gps,query_gps,1)
#print(success_rate)
#print(error)

#tensor1 = [[[1,2,3],[4,5,6]],[[7,8,9],[10,11,12]]]
#tensor2 = torch.arrange(0,3)
#torch.mul(tensor1,tensor2)

a = [1,2,3,4,5]
b = [0,0,0,0,0]
print(wasserstein_distance(a,b))