import numpy as np
from sklearn import metrics
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
    
def cluster_dbscan(descriptor_list, config):
    db = DBSCAN(eps=config["dbscan_eps"], min_samples=config["dbscan_min_samples"]).fit(descriptor_list)
    labels = db.labels_
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    print("Estimated number of clusters: %d" % n_clusters_)    

    unique_labels = set(labels) - {-1}
    representative_elements = []
    for label in unique_labels:
        #A list of {992} tensors
        descriptor_label = [value for idx, value in enumerate(descriptor_list) if labels[idx] == label]
        descriptor_label = torch.tensor(descriptor_label)
        #Tensors of size {n,992}
        print(descriptor_label.size())

        #Set some kind of threshold to eliminate the clusters with lots of elements 
        if(descriptor_label.size()[0] < config["threshold_to_eliminate_clusters"]):
            #Average
            average_descriptor = torch.mean(descriptor_label,0)
            print(average_descriptor.size())
            #Do L2 normalization
            #normalized_descriptor = F.normalize(average_descriptor.squeeze(), p=2, dim=0)
            representative_elements.append(average_descriptor)
    
    #List of tensors
    return representative_elements

def cluster_kmeans(descriptor_list, config):
    Kmean = KMeans(n_clusters=config["kmeans_num_clusters"], n_init = 10, max_iter = 400, random_state = 42).fit(descriptor_list)
    #Labels has shape (24,000,). Representative_elements has shape (n,992)
    representative_elements = Kmean.cluster_centers_

    return representative_elements

    
    





    


