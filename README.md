<br> __1. Deep Learned Features__:
<br> Code: https://github.com/utiasASRL/deep_learned_visual_features <span>
<br> Dataset (multiseason and inthedark): http://asrl.utias.utoronto.ca/datasets/2020-vtr-dataset/
<br> On the fargons, these 2 datasets are currently in /Volumes/oridatastore09/ThirdPartyData/utias. Multiseason runs do not have GPS data, and only some inthedark runs have GPS data.
<br> Information about training: 
<br> *One batch of multiseason and inthedark uses 6.6 GB and 6.2 GB respectively of GPU memory usage for training. A checkpoint (.pth file) is stored every time validation loss decreases, and early-stopping counter is incremented for every consecutive increase in validation loss. Using the 24GB RAM GPU, we can do up to batch size = 3. 
<br> * __Step 1: Build dataset__ -> 
'''
python3 -m data.build_train_test_dataset_loc --config config/data.json
'''
A spatio-temporal graph using ground truth poses is built and stored in a .pickle file. Ground truth for training is relative poses (in the form of 4x4 se3 transformation matrix) in the vehicle frame between one frame in a query run and one frame in the reference run. These 2 frames are very close to each other in geographical location. 
<br> * __Step 2: Train__ 
'''
python3 -m src.train --config config/train.json
'''
<br> ![File Structure of Output From Building Datasets and Training](/images/file_structure_of_results.png)
<br>
<br> Timing (see deep_learned_visual_features/timing.py) for one image using PyTorch. This was tested on 600 frames from 20 different runs of the multiseason dataset 
<br> CPU: 
<br> Average timing to get keypoints and associated info: 0.608 s
<br> Average timing to do matching: 0.786 s
<br> Average timing to do outlier rejection: 0.0182 s
<br> GPU:
<br> Average timing to get keypoints and associated info: 0.0485 s
<br> Average timing to do matching: 0.0345 s
<br> Average timing to do outlier rejection: 0.168 s
<br> Some changes need to be made to some lines for it to run on gpu or cpu (i.e. add or remove .cuda() at the end of some lines): stereo_camera_model.py: line 214-215 (Q) and line 143 (M), timing.py: line 203 - kpt_2d_trg_dense, ransac_block.py: line 82 (inliers) -> can just add if statement in the future to prevent having to do this
<br> 
<br> Conclusion: GPU is faster at CPU with the neural network and matching, but slower at outlier rejection.
<br>
<br> __2. Inference using TensorRT on GPU__ 
<br> Installation requirements: need CUDA Toolkit (fafnir already has it), cuDNN, and CUDA-supported version of opencv 
<br> 
<br> Installing CUDA-supported version of opencv (normal installation methods will probably install the non-cuda-supported version): 
<br> Step 1: git clone opencv and opencv_contrib.git
<br> Step 2: cmake -D CMAKE_BUILD_TYPE=Release -D CMAKE_INSTALL_PREFIX=/home/lamlam/miniconda3/envs/opencv-cuda-env -D WITH_CUDA=ON -D CUDA_ARCH_BIN=8.6 -D -D ARCH=sm_86 -D gencode=arch=compute_86,code=sm_86 -DBUILD_PERF_TESTS=OFF -DBUILD_TESTS=OFF -D OPENCV_EXTRA_MODULES_PATH=opencv_contrib/modules opencv-4.7.0
<br> Step 3: make install
<br> 
<br> Installing cuDNN: download the right TAR file and unzip 
<br> 
<br> gpu_inference/exportnetwork.py exports the .pth file into an ONNX model and then from ONNX model to TensorRT engine (there are probably other better ways to do this conversion, like a direct conversion for example instead of through 3 different frameworks)
<br>
<br> gpu_inference/cpp_gpu/src contains the code to do inference (just the Unet part, not including the subsequent operations like matching or outlier rejection) -> For batch size 1. 
<br> Currently, Input: {1,3,384,512}. 
<br> Outputs: Keypoints: torch.Size([1, 2, 768]), Descriptors: torch.Size([1, 496, 768]), Scores: torch.Size([1, 1, 768])
<br> Based on: https://github.com/cyrusbehr/tensorrt-cpp-api and https://learnopencv.com/how-to-run-inference-using-tensorrt-c-api/
<br> __Note the need to convert from NHWC (opencv) to NCHW (tensorRT)__
<br>
<br> Accuracy issues probably in the Pytorch -> ONNX -> TensorRT engine. Probably better to find a way to convert directly from PyTorch to TensorRT rather than going through ONNX. The inference part is indeed faster using TensorRT. However, the current code is slower than the Python code due to the time taken to initialize big vectors to store the data for every frame inferred. -> Probably a faster/more efficient way to do this in C++.
<br>
<br> __3. Inference using Torch JIT and subsequent tensor operations for the other blocks in the pipeline using LibTorch on CPU (there's also a GPU version of this library)__
<br> cpp_binding/exportnetwork_to_pt.py exports .pth to .pt file and cpp_binding/test.cpp uses Torch Jit to load the .pt file and performs inference
<br>
<br> __4. Visual place recognition on UTIAS inthedark data__: see folder robotcar_visual_place_recognition 
<br> Each frame is inferenced through a UNET, and normalized descriptors are extracted. 
<br> Using max-pooling: Maxpool the normalized descriptors and then use cosine similarity to measure the similarity between frames
<br> To run: main.py --config config.json
<br> Using BoW/clustering approach: Validation run (downsampled)'s descriptors are used to create clusters (options:dbscan or kmeans, not much of a difference in terms of results but kmeans is much faster) ->  Reference run's and query run's descriptors are put into the created clusters. The histogram of each of the query run's frame is compared to each of the reference run's frame using cross entropy or EMD (wasserstein distance) (cross entropy seems to do better).
<br> 
<br> To run: cd clustering and then main.py --config config_cluster.json
<br> ![Recall@1 Graph for inthedark dataset](/images/recall_plot_inthedark.png)
<br>
<br> Issues with GPS data for dark runs: All of the dark runs's gps is not accurate. See /table/inthedark_results.xlsx. One way to still use these runs (for example because we want to see how well a dark query run does with a bright reference run) is to use the transform_spatial.txt files, which localize each frame in all the other runs to a frame in run_000000. We can replace the gps data for the dark runs' frames with the gps data of the frames in run_000000 that they were localized to.
<br>
<br> __5. Pre-process Robotcar and Robotcar Seasons data for training__: see folder preprocess_robotcar
<br> - Robotcar Seasons data is a subset of the original Robotcar.
<br> - /Volumes/scratchdata/lamlam contains the full folder for stereo_left and stereo_right images as well as vo and gps data in the folders with their dates and times (ex. 2014-11-14-16-34-33) for all 10 Robotcar Seasons runs and 6 Robotcar runs with RTK data (also includes stereo_centre images for these). Of the 10 Robotcar Seasons runs, there's one reference run and 9 query runs. 
<br> - /Volumes/scratchdata/lamlam/processed_data/robotcar_seasons contains Robotcar Seasons data that has been processed for training. <br> - /Volumes/scratchdata/lamlam/processed_data/robotcar_rtk contains 6 Robotcar runs with RTK data (not the same runs as Robotcar Seasons) and is heavily downsampled (around 1300 frames/run) 
<br> - /Volumes/scratchdata/lamlam/processed_data/robotcar_rtk_full contains the same 6 Robotcar runs with RTK data but is sampled at a higher frequency (around 5000 frames/run)
<br> 
<br> - Using Robotcar Seasons robotcar_v2_train.txt: Only poses for frames from the 9 query runs are given in robotcar_v2_train.txt. The reference frame's poses are given in a different format (in COLMAP/NVM models). So, I only used the 9 runs in robotcar_v2_train.txt, chose one of them (specifically 2015-03-10-14-18-10, which has the tag "sun") as the reference run, and used the other 8 as query runs. Each of the run only has about 200 frames each, which are divided up into different areas. So, for example, the first 15 (or whatever number) frames will be close to each other, and the next 15 frames will be close to each other but far from the first 15 frames. For training using this pose estimation model, it's not great to use frames from different runs that are metrically far from each other because they will most likely be rejected as outliers anyways. 
<br> - Transformation: The se3 transformation matrices given in robotcar_v2_train.txt transform points from the Grasshopper coordinate system where the z-axis is forward, x-axis points to the right, and y-axis points downwards to the world coordinate system, where x points forward. The poses are all relative to a same stationary global origin. The UTIAS model requires the ground truth poses to be in the vehicle frame with x pointing forward, z upwards, and y to the left. The UTIAS model uses images taken using the Bumblee stereo images while Robotcar Seasons uses images taken using the Grasshopper camera. 
<br> ![Transformation formula](/images/Transformation.jpg)
<br> However, none of the many transformations I tried results in reasonable-looking z-direction translation elements (the z-direction translation elements in the relative poses between one frame in the reference run and one frame in query run should not exceed 5 meters for Oxford). 
<br>
<br> file_rearrange.py: contains functions for preparing Robotcar Seasons data for training (like copying the images with the closest timestamps to the right folder)
<br> create_ground_truth_file.py: does transformations 
<br>
<br> Using Robotcar RTK: The main advantage of using RTK data is that there are lot more frames per run and many more runs can be used for training compared to using Robotcar Seasons data, but Robotcar Seasons might contain more accurate poses (if we can get the transformations correct that is).
<br>
<br> rtk_to_poses.py: converts rtk northing,east,down and rpy data to relative poses for training
<br> process_images.py: performs demosaicing and reduces resolution of images. The resolution of 
<br> transform_tools.py: helper functions 
<br>
<br> __6. Training Robotcar dataset__: see robotcar_deep_learned_visual_features folder
<br> Tuning hyperparameters: Besides changing the camera parameters to work with the camera used for Oxford Robotcar, I also changed the gamma rate and step size to make it converge a bit better for Robotcar and changed parameters to make outlier rejection less strict (increased plane error tolerance in config file and accepted depth range in check_valid_disparity in stereo_camera_model.py).
<br>
<br> __Future potential things to change/test for training Robotcar__:
<br> - Do not sample. Use 
<br> - For debugging/comparing poses and transformations, 
<br> - Interpolate poses
<br>
<br> __7. Visual place recognition on Robotcar Seasons data__:  
<br> Using the checkpoint provided by UTIAS (trained on inthedark dataset) to do visual place recognition (using BoW) on Robotcar dataset for reference run 2014-11-18-13-20-12 (run_000000) and query run 2014-12-02-15-30-08 (run_000011), the average error is 329m and recall@1 rate is 29.5%, 36%, and 41.3% at 5m, 10m, and 25m respectively.


