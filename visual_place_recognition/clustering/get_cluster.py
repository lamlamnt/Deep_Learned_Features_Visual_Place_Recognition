import glob
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from scipy.spatial.distance import cdist
import unet_extraction_cluster
from unet_extraction_cluster import LearnedFeatureDetector
import argparse
import json
import clustering
from sklearn.manifold import TSNE
from umap import UMAP
import pandas as pd
import matplotlib.pyplot as plt
import path_processing
from scipy.stats import wasserstein_distance
from scipy.stats import entropy
import sys
sys.path.append("/home/lamlam/code/visual_place_recognition")
import evaluation_tool
from tqdm import tqdm

if __name__ == '__main__':
    #Load parameters:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=None, type=str,
                      help='config file path (default: None)')
    args = parser.parse_args()
    with open(args.config) as f:
        config = json.load(f)
    network_path = config['network_path']
    cuda = True
    learned_feature_detector = LearnedFeatureDetector(n_channels=3, 
                                                      layer_size=config['layer_size'], 
                                                      window_height=16, 
                                                      window_width=16, 
                                                      image_height=384, 
                                                      image_width=512,
                                                      checkpoint_path=network_path,
                                                      cuda=cuda)
    
    #Get clusters from validation run
    descriptors_list = []
    images_path = path_processing.path_processing_validate(config["validation_run"], config)
    #images_path_dark = path_processing.path_processing_validate(config["validation_run_dark"], config)
    #images_path = images_path + images_path_dark
    for frame_path in images_path:
        #List of descriptors of size {996}
        descriptors = learned_feature_detector.run_window_normalize(path_processing.pre_process_image(frame_path),config["score_threshold_to_be_chosen"])
        descriptors_list = descriptors_list + descriptors #list of list

    #Do clustering -> list_descriptors is a list of around 100 tensors of size {992}
    if(config["clustering_method"] == "dbscan"):
        print("Using dbscan")
        list_descriptors = clustering.cluster_dbscan(descriptors_list, config)
        #Convert this to list of tensors to tensors
        list_descriptors = torch.stack(list_descriptors)
    else:
        nparray_kmeans = clustering.cluster_kmeans(descriptors_list,config)
        print("Finished creating k means clusters")
        #Convert numpy array to tensors
        list_descriptors = torch.tensor(nparray_kmeans)

    print("Finish finding clusters from validation run")
    
    #Get histogram from reference run
    reference_run = config['reference_run']
    query_run = config['query_run']
    reference_path, ref_length, incre_ref = path_processing.path_process_ref_que_accurate(reference_run, config)
    query_path, query_length, incre_query = path_processing.path_process_ref_que_accurate(query_run, config)

    max_similarity_run_index = np.zeros(query_length,dtype=int)
    ref_count = []
    similarity_run = np.zeros((ref_length,query_length),dtype=float)

    print("Start doing inference on reference images")
    #print(reference_path)
    #print(query_path)
    #Must be done in order -> otherwise the appending will be messed up. Can initialize the ref_count first if don't want to do in order
    for i in tqdm(range(len(reference_path))):
        #List of tensors
        descriptors = learned_feature_detector.run_window_normalize(path_processing.pre_process_image(reference_path[i]),config["score_threshold_to_be_chosen"])
        descriptors = torch.tensor(descriptors)  #size {768,992}

        count = np.zeros(len(list_descriptors),dtype=int)
        #Organize the chosen descriptors into the clusters found above
        for idx,value in enumerate(descriptors):
            similarities = 1.0 - cdist(value[None, :], list_descriptors, metric="cosine")
            label = np.argmax(similarities)
            count[label] +=1
        ref_count.append(count)

    print("Start doing inference on query run")
    #Only necessary to do visual ambiguity matrix method
    histogram_matrix_B = []  #List of numpy arrays
    for frame_index,frame_path in tqdm(enumerate(query_path)):
        #List of tensors
        descriptors = learned_feature_detector.run_window_normalize(path_processing.pre_process_image(frame_path),config["score_threshold_to_be_chosen"])
        descriptors = torch.tensor(descriptors)  #size {664,992}

        count = np.zeros(len(list_descriptors),dtype=int)
        #Organize the chosen descriptors into the clusters found above
        for idx,value in enumerate(descriptors):
            similarities = 1.0 - cdist(value[None, :], list_descriptors, metric="cosine")
            label = np.argmax(similarities)
            count[label] +=1

        histogram_matrix_B.append(count)

        #Compare histograms of reference and query runs
        #Cannot do vectorized operation because wassterstein_distance doesn't support that
        min_distance = 1000
        min_index = 0
        for index,value in enumerate(ref_count):
            if(config["histogram_comparison_method"] == "EMD"):
                #Normalize as well for completeness
                distance = wasserstein_distance(count/np.sum(count),value/np.sum(count))
            else:
                #Normalize histograms and do cross entropy. Cross entropy doesn't work if there are elements with zero probability.
                #So add a small epsilon value
                epsilon = 1e-8  # Small epsilon value
                distance = entropy((count+epsilon)/np.sum(count),(value+epsilon)/np.sum(count))
            
            similarity_run[index,frame_index] = distance

    #Use compression to hopefully filter out noise
    if(config["compressed"] == "svd"):
        num_removed = config["num_singular_values_removed"]
        U, S, VT = np.linalg.svd(similarity_run, full_matrices=False)  #Already sorted in descending order
        similarity_run = U[:, :-num_removed] @ np.diag(S[:-num_removed]) @ VT[:-num_removed, :]
        #Assume the eigenvalues are positive and real -> might need to plot magnitude instead
        evaluation_tool.plot_singular_values(S)
    max_similarity_run_index = np.argmin(similarity_run,axis=0)

    #Need similarity matrix for A vs A and B vs B
    similarity_A = np.zeros((ref_length,ref_length),dtype=float)
    for idx1,value1 in enumerate(ref_count):
        for idx2,value2 in enumerate(ref_count):
            if(config["histogram_comparison_method"] == "EMD"):
                distance = wasserstein_distance(value1/np.sum(value1),value2/np.sum(value2))
            else:
                epsilon = 1e-8  # Small epsilon value
                distance = entropy((value1+epsilon)/np.sum(value1),(value2+epsilon)/np.sum(value2))
            similarity_A[idx1,idx2] = distance

    similarity_B = np.zeros((query_length,query_length),dtype=float)
    for idx1,value1 in enumerate(histogram_matrix_B):
        for idx2,value2 in enumerate(histogram_matrix_B):
            if(config["histogram_comparison_method"] == "EMD"):
                distance = wasserstein_distance(value1/np.sum(value1),value2/np.sum(value2))
            else:
                epsilon = 1e-8  # Small epsilon value
                distance = entropy((value1+epsilon)/np.sum(value1),(value2+epsilon)/np.sum(value2))
            similarity_B[idx1,idx2] = distance
    
    np.savetxt("/home/lamlam/code/visual_place_recognition/clustering/A.txt",similarity_A)
    np.savetxt("/home/lamlam/code/visual_place_recognition/clustering/B.txt",similarity_B)
    np.savetxt("/home/lamlam/code/visual_place_recognition/clustering/similarity_matrix.txt",similarity_run)

    print("Start evaluation")
    gps_distance, ref_gps, query_gps = evaluation_tool.gps_ground_truth(reference_run,query_run,ref_length,query_length, incre_ref, incre_query)
    print(ref_gps[270])
    print(query_gps[310])
    evaluation_tool.plot_similarity_clustering(similarity_run, reference_run, query_run,config["histogram_comparison_method"])
    
    threshold_list = config["success_threshold_in_m"]
    success_rate, average_error = evaluation_tool.calculate_success_rate_list(max_similarity_run_index,ref_gps,query_gps,threshold_list, reference_run, query_run, incre_ref,incre_query)
    for idx,rate in enumerate(success_rate):
        print("Success rate at threshold " + str(config["success_threshold_in_m"][idx]) + "m is " + str(rate))
    
    print("Average error in meters: " + str(average_error))
    print("Finish evaluation")
