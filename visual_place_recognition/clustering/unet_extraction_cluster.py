import sys
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import cv2
import matplotlib.pyplot as plt

sys.path.append("/home/lamlam/code/deep_learned_visual_features")
from src.model.unet import UNet
from src.model.keypoint_block import KeypointBlock
from src.utils.keypoint_tools import normalize_coords, get_norm_descriptors, get_scores
from src.dataset import Dataset

class LearnedFeatureDetector(nn.Module):
    """ 
        Class to detect learned features.
    """
    def __init__(self, n_channels, layer_size, window_height, window_width, image_height, image_width, checkpoint_path, cuda):
        """
            Set the variables needed to initialize the network.

            Args:
                num_channels (int): number of channels in the input image (we use 3 for one RGB image).
                layer_size (int): size of the first layer if the encoder. The size of the following layers are
                                  determined from this.
                window_height (int): height of window, inside which we detect one keypoint.
                window_width (int): width of window, inside which we detect one keypoint.
                image_height (int): height of the image.
                image_width (int): width of the image.
                checkpoint_path (string): path to where the network weights are stored.
                cuda (bool): true if using the GPU.
        """
        super(LearnedFeatureDetector, self).__init__()

        self.cuda = cuda
        self.n_classes = 1
        self.n_channels = n_channels
        self.layer_size = layer_size
        self.window_h = window_height
        self.window_w = window_width
        self.height = image_height
        self.width = image_width

        # Load the network weights from a checkpoint.
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
        else:
            raise RuntimeError(f'The specified checkpoint path does not exists: {checkpoint_path}')

        self.net = UNet(self.n_channels, self.n_classes, self.layer_size)
        # self.net = UNet(self.n_channels, self.n_classes, self.layer_size, self.height, self.width, checkpoint)
        self.net.load_state_dict(checkpoint['model_state_dict'])
        self.keypoint_block = KeypointBlock(self.window_h, self.window_w, self.height, self.width)
        self.sigmoid = nn.Sigmoid()

        if cuda:
            self.net.cuda()
            self.keypoint_block.cuda()

        self.net.eval()
    
    def run(self,image_tensor, score_threshold):
        if self.cuda:
            image_tensor = image_tensor.cuda()
        detector_scores, scores, descriptors = self.net(image_tensor)

        #Eliminate descriptors with scores less than a threshold
        descriptors_eliminated = descriptors.view(descriptors.size(1),-1)
        scores = scores.view(scores.size(2)*scores.size(3))
        chosen_descriptors = descriptors_eliminated[:, scores > score_threshold].detach().cpu().t().tolist()

        #List of list
        return chosen_descriptors
    
    def run_L1_normalize(self,image_tensor,score_threshold):
        #Do L1 normalization
        if self.cuda:
            image_tensor = image_tensor.cuda()
        detector_scores, scores, descriptors = self.net(image_tensor)

        #Eliminate descriptors with scores less than a threshold
        descriptors_eliminated = descriptors.view(descriptors.size(1),-1)
        scores = scores.view(scores.size(2)*scores.size(3))
        chosen_descriptors = descriptors_eliminated[:, scores > score_threshold].detach().cpu().t()
        print(chosen_descriptors.size())

        normalized_descriptor = F.normalize(chosen_descriptors, p=1, dim=0)
        print(normalized_descriptor.size())
        
    
    
        
