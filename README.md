<br> 1. Deep Learned Features:
<br>    Code: https://github.com/utiasASRL/deep_learned_visual_features
<br>    Dataset (multiseason and inthedark): http://asrl.utias.utoronto.ca/datasets/2020-vtr-dataset/
<br>    Information about training: one batch of multiseason and inthedark uses 6.6 GB and 6.2 GB respectively of GPU memory usage. A checkpoint is stored every time validation loss decreases, and early-stopping counter is incremented for every consecutive increase in validation loss.
<br>    Timing (see deep_learned_visual_features/timing.py) for one image using PyTorch. This was tested on 600 frames from 20 different runs of the multiseason dataset 
<br>    CPU: 
<br>        Average timing to get keypoints and associated info: 0.608
<br>        Average timing to do matching: 0.786
<br>        Average timing to do outlier rejection: 0.0182
<br>    GPU:
<br>        Average timing to get keypoints and associated info: 0.0485
<br>        Average timing to do matching: 0.0345
<br>        Average timing to do outlier rejection: 0.168

<br> 2. Inference using TensorRT
<br>
<br> 3. Inference using Torch JIT on CPU


