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

#Use the pre-trained weights from many runs -> Assume the provided .pth is from the runs listed as training runs

def path_processing(run):
    name = ""
    if(run < 10):
        name = "0" + str(run)
    else:
        name = str(run)
    #This is a list
    images_path = glob.glob("/Volumes/oridatastore09/ThirdPartyData/utias/inthedark/run_0000" + name + "/images/left/*.png")
    return images_path, len(images_path)

def pre_process_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    tensor = transform(image)
    return tensor[None,:,:,:]

if __name__ == '__main__':
    #Set parameters:
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
    reference_path, ref_length = path_processing(reference_run)
    query_path, query_length = path_processing(query_run)
    transform = transforms.Compose([transforms.ToTensor()])
    reference_descriptors = []
    max_similarity_run = np.zeros(query_length,dtype=float)
    max_similarity_run_index = np.zeros(query_length,dtype=int)
    similarity_run = np.zeros((ref_length,query_length),dtype=float)

    #Extract keypoints, scores, and descriptors (max pooling) from the reference run and query run
    #Using this means it's not done in order
    for frame_path in reference_path:
        #Normalized. #descriptors has size {1,992,768}. Sparse (with one descriptor for each keypoint)
        keypoints, descriptors, scores = learned_feature_detector.run(pre_process_image(frame_path))
        #Use maxpooling
        descriptors_maxpool = descriptors.squeeze()
        descriptors_maxpool = F.max_pool1d(descriptors_maxpool, kernel_size=descriptors.shape[-1])
        descriptors_maxpool = descriptors_maxpool.squeeze(1)
        reference_descriptors.append(descriptors_maxpool)

    for idx,frame_path in enumerate(query_path):
        keypoints, descriptors, scores = learned_feature_detector.run(pre_process_image(frame_path))
        descriptors_maxpool = descriptors.squeeze()
        descriptors_maxpool = F.max_pool1d(descriptors_maxpool, kernel_size=descriptors.shape[-1])
        descriptors_maxpool = descriptors_maxpool.squeeze(1)
        max_similarity = 0
        max_index = 0
        for ref_idx,ref_descriptor in enumerate(reference_descriptors):
            #Use cosine similarity - use the cdist function
            #similarity = F.cosine_similarity(descriptors_maxpool, ref_descriptor,dim=0)
            similarity = 1.0-cdist(descriptors_maxpool.reshape(1,-1), ref_descriptor.reshape(1,-1), metric = config['descriptor_difference_method'])
            #similarity2 = cdist(descriptors_maxpool.reshape(1,-1), ref_descriptor.reshape(1,-1), metric = "euclidean")
            #Exactly similar -> similarity = 1. 
            similarity = float(similarity[0,0].astype(float))
            #if(similarity < 0.98):
                #similarity = 0.9
            similarity_run[ref_idx,idx] = similarity
            if(similarity > max_similarity):
                max_similarity = similarity
                max_index = ref_idx
            max_similarity_run[idx] = max_similarity
            max_similarity_run_index[idx] = max_index

    #Evaluation
    gps_distance = evaluation_tool.gps_ground_truth(reference_run,query_run,ref_length,query_length)
    evaluation_tool.plot_similarity(similarity_run, reference_run, query_run)
    evaluation_tool.rmse_error(gps_distance,max_similarity_run_index)
    success_rate = evaluation_tool.calculate_success_rate(max_similarity_run_index,reference_run,query_run,query_length)
    print(success_rate)
