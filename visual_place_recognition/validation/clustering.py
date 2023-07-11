import numpy as np
from sklearn import metrics
from sklearn.cluster import DBSCAN
import torch
import torch.nn as nn
import torch.nn.functional as F

def cluster(descriptors):
    #Convert from {1,496,n} to {n,496}
    descriptors = descriptors.squeeze()
    descriptors = descriptors.transpose(0,1)
    descriptors_cpu = descriptors.detach().cpu()
    
    #Plot distance between each descriptor with other descriptors to help choose eps

    db = DBSCAN(eps=20, min_samples=5).fit(descriptors_cpu)
    labels = db.labels_
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)
    print("Estimated number of clusters: %d" % n_clusters_)
    print("Estimated number of noise points: %d" % n_noise_)

def cluster2(descriptor_list):
    #Descriptor_list size {662,496}
    db = DBSCAN(eps=20, min_samples=5).fit(descriptor_list)
    labels = db.labels_
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    print("Estimated number of clusters: %d" % n_clusters_)

    #Select the first core point of each cluster OR max-pooled 
    #Iterate through labels
    unique_labels = set(labels) - {-1}
    representative_elements = []
    for label in unique_labels:
        #A list of {496} tensors
        descriptor_label = [value for idx, value in enumerate(descriptor_list) if labels[idx] == label]
        #{Tensor of size (496,n)}
        descriptor_label = torch.stack(descriptor_label,dim=1)
        #Tensor of size {496,1}
        descriptors_maxpool = F.max_pool1d(descriptor_label, kernel_size=descriptor_label.shape[-1])

        #Do L2 normalization
        normalized_descriptor = F.normalize(descriptors_maxpool.squeeze(), p=2, dim=0)
        representative_elements.append(normalized_descriptor)
    return representative_elements

def cluster_dbscan(descriptor_list):
    db = DBSCAN(eps=15, min_samples=40).fit(descriptor_list)
    labels = db.labels_
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    print("Estimated number of clusters: %d" % n_clusters_)    

    unique_labels = set(labels) - {-1}
    representative_elements = []
    for label in unique_labels:
        #A list of {496} tensors
        descriptor_label = [value for idx, value in enumerate(descriptor_list) if labels[idx] == label]
        descriptor_label = torch.tensor(descriptor_label)
        #Tensors of size {n,992}

        #Set some kind of threshold to eliminate the clusters with lots of elements

        #Average
        average_descriptor = torch.mean(descriptor_label,0)

        #Do L2 normalization
        normalized_descriptor = F.normalize(average_descriptor.squeeze(), p=2, dim=0)
        representative_elements.append(normalized_descriptor)

    #List of tensors
    return representative_elements

    #Need to store around 100 descriptors of size 992


    


