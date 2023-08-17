import glob
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from scipy.spatial.distance import cdist
import unet_extraction
from unet_extraction import LearnedFeatureDetector
import evaluation_tool
import argparse
import json
import sys
sys.path.append("/home/lamlam/code/robotcar_visual_place_recognition/clustering")
import path_processing
from tqdm import tqdm

def pre_process_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    tensor = transform(image)
    return tensor[None,:,:,:]

if __name__ == '__main__':
    #Load parameters:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=None, type=str,
                      help='config file path (default: None)')
    args = parser.parse_args()
    with open(args.config) as f:
        config = json.load(f)

    cuda = True
    reference_run = config['reference_run']
    query_run = config['query_run']
    #Trained for the task of pose estimation
    network_path = config['network_path']
    
    learned_feature_detector = LearnedFeatureDetector(n_channels=3, 
                                                      layer_size=config['layer_size'], 
                                                      window_height=16, 
                                                      window_width=16, 
                                                      image_height=384, 
                                                      image_width=512,
                                                      checkpoint_path=network_path,
                                                      cuda=cuda)
    #Reference path is a list 
    reference_path, ref_length, incre_ref = path_processing.path_process_ref_que_accurate(reference_run, config)
    query_path, query_length, incre_query = path_processing.path_process_ref_que_accurate(query_run, config)
    transform = transforms.Compose([transforms.ToTensor()])
    reference_descriptors = []
    max_similarity_run = np.zeros(query_length,dtype=float)
    max_similarity_run_index = np.zeros(query_length,dtype=int)
    similarity_run = np.zeros((ref_length,query_length),dtype=float)

    #Extract keypoints, scores, and descriptors (max pooling) from the reference run and query run
    print("Start doing inference on reference images")

    for i in tqdm(range(len(reference_path))):
        descriptors = learned_feature_detector.run5(pre_process_image(reference_path[i]))
        reference_descriptors.append(descriptors)
    print("Finish processing reference frames")

    for idx,frame_path in tqdm(enumerate(query_path)):
        descriptors_maxpool= learned_feature_detector.run5(pre_process_image(frame_path))
        max_similarity = 0
        max_index = 0
        for ref_idx,ref_descriptor in enumerate(reference_descriptors):
            #Use cosine similarity 
            similarity = 1.0-cdist(descriptors_maxpool.reshape(1,-1), ref_descriptor.reshape(1,-1), metric = config['descriptor_difference_method'])
            #Exactly similar -> similarity = 1. 
            similarity = float(similarity[0,0].astype(float))
            similarity_run[ref_idx,idx] = similarity
            if(similarity > max_similarity):
                max_similarity = similarity
                max_index = ref_idx
            max_similarity_run[idx] = max_similarity
            max_similarity_run_index[idx] = max_index
    print("Finish calculating similarity")

    #Evaluation
    gps_distance, ref_gps, query_gps = evaluation_tool.gps_ground_truth(ref_length,query_length, incre_ref, incre_query, config)
    print("Finish calculating gps ground truth")
    
    evaluation_tool.plot_similarity(similarity_run, config)
    
    threshold_list = config["success_threshold_in_m"]
    success_rate, average_error = evaluation_tool.calculate_success_rate_list(max_similarity_run_index,ref_gps,query_gps,threshold_list, config)
    for idx,rate in enumerate(success_rate):
        print("Success rate at threshold " + str(config["success_threshold_in_m"][idx]) + "m is " + str(rate))
    
    print("Average error in meters: " + str(average_error))
    print("Finish evaluation")

    #Write to a file for record keeping
    file_path = "results/" + str(reference_run) + "_" + str(query_run) + ".txt"
    file = open(file_path,"w+")
    for key, value in config.items():
        file.write(f"{key}: {value}\n")
    for idx,rate in enumerate(success_rate):
        file.write("Success rate at threshold " + str(config["success_threshold_in_m"][idx]) + "m is " + str(rate) + "\n")
    file.write("Average error in meters: " + str(average_error))
    file.close()


