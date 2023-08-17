<br> #1. Deep Learned Features:
<br> Code: https://github.com/utiasASRL/deep_learned_visual_features <span>
<br> Dataset (multiseason and inthedark): http://asrl.utias.utoronto.ca/datasets/2020-vtr-dataset/
<br> These 2 datasets are currently in /Volumes/oridatastore09/ThirdPartyData/utias
<br> Information about training: 
<br> Markup : * One batch of multiseason and inthedark uses 6.6 GB and 6.2 GB respectively of GPU memory usage. A checkpoint is stored every time validation loss decreases, and early-stopping counter is incremented for every consecutive increase in validation loss. Using the 24GB RAM GPU, we can do up to batch size = 3. 
<br> Markup : * __Step 1: Build dataset__ -> 'python3 -m data.build_train_test_dataset_loc --config config/data.json' A spatio-temporal graph is
Ground truth for training is relative poses (in the form of 4x4 se3 transformation matrix) in the vehicle frame between one frame in a query run and one frame in the reference run. These 2 frames are very close to each other in geographical location. 
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
<br> Conclusion: 
<br>
<br> #2. Inference using TensorRT on GPU 
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
<br> Note the need to convert from NHWC (opencv) to NCHW (tensorRT).
<br>
<br> Accuracy issues probably in the Pytorch -> ONNX -> TensorRT engien 
<br>
<br> 3. Inference using Torch JIT and subsequent tensor operations for the other blocks in the pipeline using LibTorch on CPU (there's also a GPU version of this library): 
<br> cpp_binding/exportnetwork_to_pt.py exports .pth to .pt file and cpp_binding/test.cpp uses Torch Jit to load the .pt file and performs inference
<br>
<br> 4. Bag of Binary Words
<br>
<br> 5. Visual place recognition on UTIAS inthedark data
<br>
<br> 6. Pre-process robotcar seasons data
<br>
<br> 7. Visual place recognition on Robotcar Seasons data 
<br>


