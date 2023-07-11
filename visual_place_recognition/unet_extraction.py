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


def get_keypoint_info(kpt_2D, scores_map, descriptors_map):
    """
        Gather information we need associated with each detected keypoint. Compute the normalized 
        descriptor and the score for each keypoint.

        Args:
            kpt_2D (torch.tensor): keypoint 2D image coordinates, (Bx2xN).
            scores_map (torch.tensor): scores for each pixel, (Bx1xHxW).
            descriptors_map (torch.tensor): descriptors for each pixel, (BxCxHxW).

        Returns:
            kpt_desc_norm (torch.tensor): Normalized descriptor for each keypoint, (BxCxN).
            kpt_scores (torch.tensor): score for each keypoint, (Bx1xN).

    """
    batch_size, _, height, width = scores_map.size()
    kpt_2D_norm = normalize_coords(kpt_2D, batch_size, height, width).unsqueeze(1)  # Bx1xNx2
    kpt_desc_norm = get_norm_descriptors(descriptors_map, True, kpt_2D_norm)
    kpt_scores = get_scores(scores_map, kpt_2D_norm)
    return kpt_desc_norm, kpt_scores


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

    #Uses L2 Normalization
    def run(self, image_tensor):
        """
            Forward pass of network to get keypoint detector values, descriptors and, scores
            Args:
                image_tensor (torch.tensor, Bx3xHxW): RGB images to input to the network.
            Returns:
                keypoints (torch.tensor, Bx2xN): the detected keypoints, N=number of keypoints.
                descriptors (torch.tensor, BxCxN): descriptors for each keypoint, C=496 is length of descriptor.
                scores (torch.tensor, Bx1xN): an importance score for each keypoint.
        """
        if self.cuda:
            image_tensor = image_tensor.cuda()
        detector_scores, scores, descriptors = self.net(image_tensor)

        descriptor_reshaped = descriptors.view(descriptors.size(0), -1)  # (batch_size, height * width * channels)
        normalized_descriptor = F.normalize(descriptor_reshaped, p=2, dim=1)
        normalized_descriptor = normalized_descriptor.view(descriptors.size())
        descriptors_maxpool =F.max_pool2d(normalized_descriptor, kernel_size=(normalized_descriptor.shape[-2], normalized_descriptor.shape[-1]))
        descriptors_maxpool = descriptors_maxpool.squeeze()

        return descriptors_maxpool.detach().cpu()

    #Uses channel normalization
    def run2(self, image_tensor):
        if self.cuda:
            image_tensor = image_tensor.cuda()
        detector_scores, scores, descriptors = self.net(image_tensor)
        descriptor_reshaped = descriptors.view(descriptors.size(0), -1)  # (batch_size, height * width * channels)

        #Normalization part 
        descriptors_mean = torch.mean(descriptor_reshaped, dim=1, keepdim=True)           # Bx1x(N or HW)
        descriptors_std = torch.std(descriptor_reshaped, dim=1, keepdim=True)             # Bx1x(N or HW)
        normalized_descriptor = (descriptor_reshaped - descriptors_mean) / descriptors_std 

        normalized_descriptor = normalized_descriptor.view(descriptors.size())
        descriptors_maxpool =F.max_pool2d(normalized_descriptor, kernel_size=(normalized_descriptor.shape[-2], normalized_descriptor.shape[-1]))
        descriptors_maxpool = descriptors_maxpool.squeeze()
        return descriptors_maxpool.detach().cpu()
    
    #Uses L2 normalization and window max-pooling
    def run3(self,image_tensor):
        if self.cuda:
            image_tensor = image_tensor.cuda()
        detector_scores, scores, descriptors = self.net(image_tensor)
        scores = self.sigmoid(scores)
        keypoints = self.keypoint_block(detector_scores)
        point_descriptors_norm, point_scores = get_keypoint_info(keypoints, scores, descriptors)
        descriptors_maxpool = point_descriptors_norm.squeeze()
        descriptors_maxpool = F.max_pool1d(descriptors_maxpool, kernel_size=descriptors.shape[-1])
        descriptors_maxpool = descriptors_maxpool.squeeze(1)
        return descriptors_maxpool.detach().cpu()
    
    #Uses channel normalization and window max-pooling (original - like Mona)
    def run4(self,image_tensor):
        if self.cuda:
            image_tensor = image_tensor.cuda()
        detector_scores, scores, descriptors = self.net(image_tensor)
        scores = self.sigmoid(scores)
        keypoints = self.keypoint_block(detector_scores)
        descriptors_norm, point_scores = get_keypoint_info(keypoints, scores, descriptors)

        descriptors_maxpool = descriptors_norm.squeeze()
        descriptors_maxpool = F.max_pool1d(descriptors_maxpool, kernel_size=descriptors.shape[-1])
        descriptors_maxpool = descriptors_maxpool.squeeze(1)
        return descriptors_maxpool.detach().cpu()
    
    #multiply by scores before max-pooling
    def run5(self,image_tensor):
        if self.cuda:
            image_tensor = image_tensor.cuda()
        detector_scores, scores, descriptors = self.net(image_tensor)

        descriptors = torch.mul(descriptors,scores)
        
        #L2 normalization
        descriptor_reshaped = descriptors.view(descriptors.size(0), -1)  # (batch_size, height * width * channels)
        normalized_descriptor = F.normalize(descriptor_reshaped, p=2, dim=1)
        normalized_descriptor = normalized_descriptor.view(descriptors.size())
        descriptors_maxpool =F.max_pool2d(normalized_descriptor, kernel_size=(normalized_descriptor.shape[-2], normalized_descriptor.shape[-1]))
        descriptors_maxpool = descriptors_maxpool.squeeze()

        return descriptors_maxpool.detach().cpu()
    
    #Also returns scores
    def run6(self,image_tensor):
        if self.cuda:
            image_tensor = image_tensor.cuda()
        detector_scores, scores, descriptors = self.net(image_tensor)
        keypoints = self.keypoint_block(detector_scores)
        descriptors_norm, point_scores = get_keypoint_info(keypoints, scores, descriptors)
        
        clustering.cluster(descriptors_norm)
        #Plot point scores
        plt.figure()
        plt.title("Scores Histogram (768)")
        plt.hist(point_scores.squeeze().detach().cpu(),bins=20)
        plt.xlabel("Scores")
        plt.ylabel("Frequency")
        plt.savefig("plots/" + "scores_histogram_768.png")

        descriptors_maxpool = descriptors_norm.squeeze()
        descriptors_maxpool = F.max_pool1d(descriptors_maxpool, kernel_size=descriptors.shape[-1])
        descriptors_maxpool = descriptors_maxpool.squeeze(1)
        return descriptors_maxpool.detach().cpu()
    
    def run7(self,image_tensor):
        if self.cuda:
            image_tensor = image_tensor.cuda()
        detector_scores, scores, descriptors = self.net(image_tensor)

        #Eliminate descriptors with scores less than 0.1
        descriptors_eliminated = descriptors.view(descriptors.size(1),-1)
        scores = scores.view(scores.size(2)*scores.size(3))
        #chosen_descriptors = descriptors_eliminated[:, scores > 0.4].detach().cpu().t().tolist()
        chosen_descriptors = descriptors_eliminated[:, scores > 0.4].detach().cpu().t()
        """
        chosen_descriptors = []
        for idx,value in enumerate(scores):
            if(scores[idx] > 0.4):
                chosen_descriptors.append(descriptors_eliminated[:,idx].detach().cpu())
        """
        #Do clustering with the chosen descriptors 
        #clustering.cluster2(chosen_descriptors)

        #L2 normalization
        descriptor_reshaped = descriptors.view(descriptors.size(0), -1)   # (batch_size, height * width * channels)
        normalized_descriptor = F.normalize(descriptor_reshaped, p=2, dim=1)
        normalized_descriptor = normalized_descriptor.view(descriptors.size())
        descriptors_maxpool =F.max_pool2d(normalized_descriptor, kernel_size=(normalized_descriptor.shape[-2], normalized_descriptor.shape[-1]))
        descriptors_maxpool = descriptors_maxpool.squeeze()

        return descriptors_maxpool.detach().cpu()

