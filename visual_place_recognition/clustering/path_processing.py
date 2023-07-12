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

def path_processing_validate(run, config):
    name = ""
    if(run < 10):
        name = "0" + str(run)
    else:
        name = str(run)
    images_path = glob.glob("/Volumes/oridatastore09/ThirdPartyData/utias/inthedark/run_0000" + name + "/images/left/*.png")
    gps_path ="/Volumes/oridatastore09/ThirdPartyData/utias/inthedark/run_0000" + name + "/gps.txt"
    with open(gps_path, 'r') as file:
        line_count = 0
        for line in file:
            line_count += 1
    #Some images at the end are missing gps data
    difference = len(images_path) - line_count
    if(difference > 0):
        images_path = images_path[:-difference]
    
    incre = int(round(len(images_path)/config["sample_rate_num_frames"]))
    images_path = sorted(images_path)[::incre]
    return images_path

def pre_process_image(image_path):
    transform = transforms.Compose([transforms.ToTensor()])
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    tensor = transform(image)
    return tensor[None,:,:,:]

def path_process_ref_que(run,config):
    name = ""
    if(run < 10):
        name = "0" + str(run)
    else:
        name = str(run)
    images_path = glob.glob("/Volumes/oridatastore09/ThirdPartyData/utias/inthedark/run_0000" + name + "/images/left/*.png")
    gps_path ="/Volumes/oridatastore09/ThirdPartyData/utias/inthedark/run_0000" + name + "/gps.txt"
    with open(gps_path, 'r') as file:
        line_count = 0
        for line in file:
            line_count += 1
    #Some images at the end are missing gps data
    difference = len(images_path) - line_count
    if(difference > 0):
        images_path = images_path[:-difference]
    #Downsampled
    incre = int(round(line_count*config["number_meter_per_frame"]/config["path_length"]))        
    images_path = sorted(images_path)[::incre]
    return images_path, len(images_path), incre