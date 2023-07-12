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
import sys
sys.path.append("/home/lamlam/code/visual_place_recognition")
import evaluation_tool

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
        descriptors = learned_feature_detector.run(path_processing.pre_process_image(frame_path),config["score_threshold_to_be_chosen"])
        descriptors_list = descriptors_list + descriptors #list of list

    #Do clustering -> list_descriptors is a list of around 100 tensors of size {992}
    list_descriptors = clustering.cluster_dbscan(descriptors_list, config)

    print("K-means")
    nparray_kmeans = clustering.cluster_kmeans(descriptors_list,config)
    
    #Convert this to list of tensors to tensors
    list_descriptors = torch.stack(list_descriptors)

    print("Finish finding clusters from validation run")
    
    #Get histogram from reference run
    reference_run = config['reference_run']
    query_run = config['query_run']
    reference_path, ref_length, incre_ref = path_processing.path_process_ref_que(reference_run, config)
    query_path, query_length, incre_query = path_processing.path_process_ref_que(query_run, config)
    max_similarity_run_index = np.zeros(query_length,dtype=int)
    ref_count = []
    similarity_run = np.zeros((ref_length,query_length),dtype=float)

    print("Start doing inference on reference images")
    #Must be done in order -> otherwise the appending will be messed up. Can initialize the ref_count first if don't want to do in order
    for i in range (len(reference_path)):
        #List of tensors
        descriptors = learned_feature_detector.run(path_processing.pre_process_image(reference_path[i]),config["score_threshold_to_be_chosen"])
        descriptors = torch.tensor(descriptors)  #size {664,992}

        count = np.zeros(len(list_descriptors),dtype=int)
        max_sim = 0
        label = 0
        #Organize the chosen descriptors into the clusters found above
        for idx,value in enumerate(descriptors):
            #1d np array of size around 100 and the values are the counts
            """
            for center_label, center in enumerate(list_descriptors):
                #Find cosine similarity between the descriptor and each of the cluster's center
                similarity = 1.0-cdist(value.unsqueeze(0), center.unsqueeze(0), metric = "cosine")
                if(similarity > max_sim):
                    max_sim = similarity
                    label = center_label
            """
            similarities = 1.0 - cdist(value[None, :], list_descriptors, metric="cosine")
            label = np.argmax(similarities)
            count[label] +=1
        ref_count.append(count)
    print(ref_count)

    print("Start doing inference on query run")
    for frame_index,frame_path in enumerate(query_path):
        #List of tensors
        descriptors = learned_feature_detector.run(path_processing.pre_process_image(frame_path),config["score_threshold_to_be_chosen"])
        descriptors = torch.tensor(descriptors)  #size {664,992}

        count = np.zeros(len(list_descriptors),dtype=int)
        max_sim = 0
        label = 0
        #Organize the chosen descriptors into the clusters found above
        for idx,value in enumerate(descriptors):
            #1d np array of size around 100 and the values are the counts
            """
            for center_label, center in enumerate(list_descriptors):
                #Find cosine similarity between the descriptor and each of the cluster's center
                similarity = 1.0-cdist(value.unsqueeze(0), center.unsqueeze(0), metric = "cosine")
                if(similarity > max_sim):
                    max_sim = similarity
                    label = center_label
            """
            similarities = 1.0 - cdist(value[None, :], list_descriptors, metric="cosine")
            label = np.argmax(similarities)
            count[label] +=1

        #Compare histograms of reference and query runs
        #Cannot do vectorized operation because wassterstein_distance doesn't support that
        min_distance = 1000
        min_index = 0
        for index,value in enumerate(ref_count):
            distance = wasserstein_distance(count,value)
            similarity_run[index,frame_index] = distance
            if(distance < min_distance):
                min_distance = distance
                min_index = index
        max_similarity_run_index[frame_index] = min_index
        
    print("Start evaluation")
    gps_distance, ref_gps, query_gps = evaluation_tool.gps_ground_truth(reference_run,query_run,ref_length,query_length, incre_ref, incre_query)
    evaluation_tool.plot_similarity_wasserstein(similarity_run, reference_run, query_run)
    
    threshold_list = config["success_threshold_in_m"]
    success_rate, average_error = evaluation_tool.calculate_success_rate_list(max_similarity_run_index,ref_gps,query_gps,threshold_list)
    for idx,rate in enumerate(success_rate):
        print("Success rate at threshold " + str(config["success_threshold_in_m"][idx]) + "m is " + str(rate))
    
    print("Average error in meters: " + str(average_error))
    print("Finish evaluation")