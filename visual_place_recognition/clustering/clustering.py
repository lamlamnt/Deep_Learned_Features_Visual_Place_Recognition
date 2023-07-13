import numpy as np
from sklearn import metrics
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


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
    
    #print("Number of clusters chosen: " + str(len(representative_elements)))
    #List of tensors
    return representative_elements

def cluster_kmeans(descriptor_list, config):
    Kmean = KMeans(n_clusters=config["kmeans_num_clusters"]).fit(descriptor_list)
    #Labels has shape (24,000,). Representative_elements has shape (n,992)
    representative_elements = Kmean.cluster_centers_
    
    return representative_elements

    #Choosing the optimal number of k clusters
    """
    sse = []
    for k in range(5,45,5):
        kmeans = KMeans(n_clusters=k).fit(descriptor_list)
        sse.append(kmeans.inertia_)

    x_values = list(range(5,45,5))
    #visualize results#
    plt.figure()
    plt.title("Finding optimal number of k clusters")
    plt.plot(x_values,sse)
    plt.xlabel("Number of Clusters")
    plt.ylabel("SSE")
    plt.savefig("/home/lamlam/code/visual_place_recognition/clustering/plot_visualize_cluster/k_clustering_detailed.png")
    """

    """
    # Count the number of elements in each cluster
    labels = Kmean.labels_
    cluster_counts = np.bincount(labels)
    for cluster, count in enumerate(cluster_counts):
        print(f"Cluster {cluster}: {count} elements")
    """

    





    


