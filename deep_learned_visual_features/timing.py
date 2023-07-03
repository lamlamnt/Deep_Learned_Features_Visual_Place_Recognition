'''A script that shows how to pass an image to the network to get keypoints, descriptors and scrores. '''

import sys
import os

import argparse
import json
import os
import pickle
import re
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import cv2

from src.model.unet import UNet
from src.model.keypoint_block import KeypointBlock
from src.model.matcher_block import MatcherBlock
from src.model.ransac_block import RANSACBlock
from src.model.svd_block import SVDBlock
from src.model.unet import UNet
from src.model.weight_block import WeightBlock
from src.utils.keypoint_tools import normalize_coords, get_norm_descriptors, get_scores
from src.utils.stereo_camera_model import StereoCameraModel
from src.dataset import Dataset
import time

def get_disparity(left_img, right_img):
        """
            Create the disparity image using functions from OpenCV.

            Args:
                left_img (numpy.uint8): left stereo image.
                right_img (numpy.uint8): right stereo image.
        """
        stereo = cv2.StereoSGBM_create(minDisparity = 0,
                                       numDisparities = 48, 
                                       blockSize = 5, 
                                       preFilterCap = 30, 
                                       uniquenessRatio = 20, 
                                       P1 = 200, 
                                       P2 = 800, 
                                       speckleWindowSize = 200, 
                                       speckleRange = 1, 
                                       disp12MaxDiff = -1)

        disp = stereo.compute(left_img, right_img)
        disp  = disp.astype(np.float32) / 16.0

        # Adjust values close to or equal to zero, which would cause problems for depth calculation.
        disp[(disp < 1e-4) & (disp >= 0.0)] = 1e-4

        return torch.from_numpy(disp)

def get_keypoint_info(kpt_2D, scores_map, descriptors_map, disparity, stereo_cam):
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

    kpt_3D, valid = stereo_cam.inverse_camera_model(kpt_2D, disparity)

    return kpt_3D, valid, kpt_desc_norm, kpt_scores


class LearnedFeatureDetector(nn.Module):
    """ 
        Class to detect learned features.
    """
    def __init__(self, config,n_channels, layer_size, window_height, window_width, image_height, image_width, checkpoint_path, cuda):
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
        self.batch_size = 1

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

        #Set up other blocks
        self.config = config
        # Transform from sensor to vehicle.
        T_s_v = torch.tensor([[0.000796327, -1.0, 0.0, 0.119873],
                              [-0.330472, -0.000263164, -0.943816, 1.49473],
                              [0.943815, 0.000751586, -0.330472, 0.354804],
                              [0.0, 0.0, 0.0, 1.0]])
        self.register_buffer('T_s_v', T_s_v)
        self.matcher_block = MatcherBlock()
        self.weight_block = WeightBlock()
        self.svd_block = SVDBlock(self.T_s_v)
        self.ransac_block = RANSACBlock(config, self.T_s_v)

        self.stereo_cam = StereoCameraModel(config['stereo']['cu'], config['stereo']['cv'],
                                            config['stereo']['f'], config['stereo']['b'])
        
        v_coord, u_coord = torch.meshgrid([torch.arange(0, self.height), torch.arange(0, self.width)])
        v_coord = v_coord.reshape(self.height * self.width).float()  # HW
        u_coord = u_coord.reshape(self.height * self.width).float()
        image_coords = torch.stack((u_coord, v_coord), dim=0)  # 2xHW
        self.register_buffer('image_coords', image_coords)

        if cuda:
            self.net.cuda()
            self.keypoint_block.cuda()
            self.matcher_block.cuda()
            self.ransac_block.cuda()
            self.svd_block.cuda()
            self.weight_block.cuda()

        self.net.eval()

    def run(self, image_tensor, target_tensor, disparity_tensor, target_disparity_tensor):
        """
            Forward pass of network to get keypoint detector values, descriptors and, scores

            Args:
                image_tensor (torch.tensor, Bx3xHxW): RGB images to input to the network.

            Returns:
                keypoints (torch.tensor, Bx2xN): the detected keypoints, N=number of keypoints.
                descriptors (torch.tensor, BxCxN): descriptors for each keypoint, C=496 is length of descriptor.
                scores (torch.tensor, Bx1xN): an importance score for each keypoint.
        """
        start_time_1 = time.time()
        if self.cuda:
            image_tensor = image_tensor.cuda()
            target_tensor = target_tensor.cuda()
            disparity_tensor = disparity_tensor.cuda()
            target_disparity_tensor = target_disparity_tensor.cuda()

        detector_scores_src, scores_src, descriptors_src = self.net(image_tensor)
        scores_src = self.sigmoid(scores_src)

        # Get 2D keypoint coordinates from detector scores, Bx2xN
        kpt_2D_src = self.keypoint_block(detector_scores_src)

        # Get one descriptor and scrore per keypoint, BxCxN, Bx1xN, C=496.
        kpt_3D_src, kpt_valid_src, kpt_desc_norm_src, kpt_scores_src = get_keypoint_info(kpt_2D_src,scores_src,descriptors_src,disparity_tensor,self.stereo_cam)
        
        detector_scores_trg, scores_trg, descriptors_trg = self.net(target_tensor)
        scores_trg = self.sigmoid(scores_trg)
        kpt_2D_trg = self.keypoint_block(detector_scores_trg)
        kpt_3D_trg, kpt_valid_trg, kpt_desc_norm_trg, kpt_scores_trg = get_keypoint_info(kpt_2D_trg,scores_trg,descriptors_trg, target_disparity_tensor,self.stereo_cam)
        time1 = time.time()-start_time_1 


        start_time_2 = time.time()
        #Match keypoints from source and target
        if self.config['pipeline']['dense_matching']:
            # Match against descriptors for each pixel in the target.
            desc_norm_trg_dense = get_norm_descriptors(descriptors_trg)
            kpt_2D_trg_dense = self.image_coords.unsqueeze(0).expand(self.batch_size, 2, self.height * self.width).cuda()
            # Compute the coordinates of the matched keypoints in the target frame, which we refer to as pseudo points.
            kpt_2D_pseudo = self.matcher_block(kpt_2D_src, kpt_2D_trg_dense, kpt_desc_norm_src, desc_norm_trg_dense)
        else:
            # Match only against descriptors associated with detected keypoints in the target frame.
            kpt_2D_pseudo = self.matcher_block(kpt_2D_src, kpt_2D_trg, kpt_desc_norm_src, kpt_desc_norm_trg)

        # Get 3D point coordinates, normalized descriptors, and scores associated with each individual matched pseudo
        # point in the target frame (Bx4xN, BxCxN, Bx1xN).
        kpt_3D_pseudo, kpt_valid_pseudo, kpt_desc_norm_pseudo, kpt_scores_pseudo = get_keypoint_info(kpt_2D_pseudo,scores_trg,descriptors_trg,target_disparity_tensor,self.stereo_cam)
        # Compute the weight associated with each matched point pair. They will be used when computing the pose.
        weights = self.weight_block(kpt_desc_norm_src, kpt_desc_norm_pseudo, kpt_scores_src, kpt_scores_pseudo)
        time2 = time.time() - start_time_2

        #Outlier rejection
        # Find the inliers by using RANSAC (inference).
        start_time_3 = time.time()
        valid_inliers = torch.ones(kpt_valid_src.size()).type_as(kpt_valid_src)
        ransac_inliers = self.ransac_block(kpt_3D_src,kpt_3D_pseudo,kpt_2D_pseudo, kpt_valid_src,kpt_valid_pseudo,weights,self.config['outlier_rejection']['dim'][0])
        valid_inliers = ransac_inliers.unsqueeze(1)

        #  Check that we have enough inliers for all example sin the bach to compute pose.
        valid = kpt_valid_src & kpt_valid_pseudo & valid_inliers
        num_inliers = torch.sum(valid.squeeze(1), dim=1)[0]
        #if torch.any(num_inliers < 6):
            #raise RuntimeError('Too few inliers to compute pose: {}'.format(num_inliers))
        time3 = time.time() - start_time_3

        #Compute pose
        weights[valid == 0] = 0.0
        #T_trg_src = self.svd_block(kpt_3D_src, kpt_3D_pseudo, weights)

        #return T_trg_src
        return time1, time2, time3
        
def get_image(run, run_node):
    left_image_path = "/Volumes/oridatastore09/ThirdPartyData/utias/multiseason/run_000" + run + "/images/left/000" + run_node + ".png"
    right_image_path = "/Volumes/oridatastore09/ThirdPartyData/utias/multiseason/run_000" + run + "/images/right/000" + run_node + ".png"
    
    transform = transforms.Compose([transforms.ToTensor()])
    # Read the image
    left_image = cv2.imread(left_image_path)
    # Convert BGR image to RGB image
    left_image = cv2.cvtColor(left_image, cv2.COLOR_BGR2RGB)
    # Convert the image to Torch tensor (not normalized)
    left_tensor = transform(left_image) 
    left_tensor = left_tensor[None,:,:,:]
    #Get disparity
    right_image = cv2.imread(right_image_path)
    disparity = get_disparity(left_image,right_image)
    disparity = disparity[None,:,:]
    return left_tensor, disparity

def get_mean_column(timing, column):
    sum = 0
    for i in range(len(timing)):
        sum += timing[i][column]
    return str(sum/len(timing))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=None, type=str,
                      help='config file path (default: None)')

    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    cuda = True
    checkpoint = '/home/lamlam/data/networks/network_multiseason_inthedark_layer16.pth'
    learned_feature_detector = LearnedFeatureDetector(config,n_channels=3, 
                                                      layer_size=16, 
                                                      window_height=16, 
                                                      window_width=16, 
                                                      image_height=384, 
                                                      image_width=512,
                                                      checkpoint_path=checkpoint,
                                                      cuda=cuda)
    #Read from transforms_spatial.txt
    #600 frames
    #Some runs have svd non-convergence errors: 15, 40,31(before 10 runs are done)
    runs = ["005","007","010","013","016","025","031","039","041","060","070","075","080","090","100","105","110","120","130","134"]
    timing = [[0] * 3 for _ in range(len(runs))]
    for j in range(len(runs)):
        info_path = "/Volumes/oridatastore09/ThirdPartyData/utias/multiseason/run_000" + runs[j] + "/transforms_spatial.txt"
        with open(info_path, "r") as file:
            for i in range(30):
                line = file.readline().strip()  # Read the first line and remove leading/trailing whitespace
                numbers = line.split(",")  # Split the line by comma
                first_four = numbers[:4]  # Get the first four numbers
                #Need to pad with zeros
                if(len(first_four[1]) == 1):
                    first_four[1] = "00" + first_four[1]
                elif(len(first_four[1]) == 2):
                    first_four[1] = "0" + first_four[1]
                if(len(first_four[3]) == 1):
                    first_four[3] = "00" + first_four[3]
                elif(len(first_four[3]) == 2):
                    first_four[3] = "0" + first_four[3]
                #print(first_four[0] + " " + first_four[1] + " "+ first_four[2] + " " + first_four[3])
                test_image,disparity_tensor = get_image(runs[j], first_four[1])
                test_target,target_disparity_tensor = get_image("000",first_four[3])
                time1, time2, time3 = learned_feature_detector.run(test_image,test_target, disparity_tensor, target_disparity_tensor)
                timing[j][0] = time1
                timing[j][1] = time2
                timing[j][2] = time3
    print("Average timing to get keypoints and associated info: " + get_mean_column(timing,0))
    print("Average timing to do matching: " + get_mean_column(timing,1))
    print("Average timing to do outlier rejection: " + get_mean_column(timing,2))
  
