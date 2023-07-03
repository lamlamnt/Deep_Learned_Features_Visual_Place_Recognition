#include </home/lamlam/ENTER/lib/python3.10/site-packages/torch/include/torch/csrc/api/include/torch/torch.h>
#include </home/lamlam/ENTER/lib/python3.10/site-packages/torch/include/torch/script.h>
#include </home/lamlam/downloads/libtorch/include/torch/csrc/api/include/torch/nn/functional/fold.h>
#include </home/lamlam/downloads/libtorch/include/torch/csrc/api/include/torch/nn/functional/vision.h>
#include <iostream>
using namespace std;
using namespace torch;
namespace F = torch::nn::functional;

//Normalize 2D keypoint coordinates to lie in the range [-1, 1].
Tensor normalize_coords(Tensor coords_2D, int batch_size, int height, int width) 
{
    //dim, start, end parameters for slice
    Tensor u_norm = (2*coords_2D.slice(1, 0, 1).view({batch_size, -1}) / (width - 1)) - 1;
    Tensor v_norm = (2*coords_2D.slice(1, 1, 2).view({batch_size, -1}) / (height - 1)) - 1;
    return torch::stack({u_norm, v_norm}, 2);
}

//descriptor_map is either dense with one descriptor for each image pixel (BxCxHxW), or sparse with one descriptor for each keypoint, BxCxN.
Tensor get_norm_descriptors(Tensor descriptor_map, bool sample, Tensor keypoints_norm) 
{
  if (descriptor_map.dim() == 4) {
    //The descriptor map has dense descriptors, one for each pixel.
    int batch_size = descriptor_map.size(0);
    int channels = descriptor_map.size(1);
    int height = descriptor_map.size(2);
    int width = descriptor_map.size(3);

      if (sample) {
        // Sample descriptors for the given keypoints.
        Tensor descriptors = F::grid_sample(descriptor_map, keypoints_norm, mode="bilinear");  // BxCx1xN
        descriptors = descriptors.reshape({batch_size, channels, keypoints_norm.size(1)});         // BxCxN
        return descriptors;
      } 
      else {return descriptor_map.reshape({batch_size, channels, height * width});}  // BxCxHW 
  } 
  else 
  {
    // The descriptor map has sparse descriptors, one for each keypoint.
    return descriptor_map;
  }
}

Tensor get_scores(Tensor scores_map, Tensor keypoints_norm) {
    int batch_size = keypoints_norm.size(0);
    int num_points = keypoints_norm.size(1);

    Tensor kpt_scores = F::grid_sample(scores_map, keypoints_norm, mode="bilinear");  // Bx1x1xN
    kpt_scores = kpt_scores.reshape({batch_size, 1, num_points});

    return kpt_scores;
}

Tensor get_keypoint_info(Tensor kpt_2D, Tensor scores_map, Tensor descriptors_map) 
{
    int batch_size, height, width;
    std::tie(batch_size, std::ignore, height, width) = scores_map.sizes();
    // Bx1xNx2
    Tensor kpt_2D_norm = normalize_coords(kpt_2D, batch_size, height, width).unsqueeze(1);  
    Tensor kpt_desc_norm = get_norm_descriptors(descriptors_map, true, kpt_2D_norm);
    Tensor kpt_scores = get_scores(scores_map, kpt_2D_norm);
    return make_tuple(kpt_desc_norm, kpt_scores);
}

//One keypoint is detected per window.
class KeypointBlock : public torch::nn::Module 
{
  public: 
    int win_height;
    int win_width;
    Tensor v_coords;
    Tensor u_coords;
    KeypointBlock(int window_height, int window_width, int image_height, int image_width)
    {
      win_height = window_height;
      win_width = window_width;
      v_coords = torch::arange(0, image_height).unsqueeze(0).to(torch::kFloat);  // 1 x H
      u_coords = torch::arange(0, image_width).unsqueeze(0).to(torch::kFloat);  // 1 x W
      v_coords = v_coords.expand({1, image_width}).unsqueeze(0);  // 1 x H x W
      u_coords = u_coords.expand({1, image_height}).unsqueeze(0);  // 1 x W x H

      register_buffer("v_coords", v_coords);
      register_buffer("u_coords", u_coords);
    }
    Tensor forward(Tensor detector_values);
};

// Given a tensor of detector values (same width/height as the original image), divide the tensor into
//windows and use a spatial softmax over each window. Returns the coordinates of 2d keypoints
Tensor KeypointBlock::forward(Tensor detector_values)
{
  float temperature = 1.0;
  int64_t batch_size = detector_values.size(0);
  int64_t height = detector_values.size(2);
  int64_t width = detector_values.size(3);

  //unfold parameteres: the coordinates,kernel_size, stride
  // B x n_window_elements x n_windows
  Tensor v_windows = F::unfold(v_coords.expand({batch_size, 1, height, width}), F::UnfoldFuncOptions({win_height, win_width}).stride({win_height, win_width}));
  Tensor u_windows = F::unfold(u_coords.expand({batch_size, 1, height, width}), F::UnfoldFuncOptions({win_height, win_width}).stride({win_height, win_width}));

  // B x n_wind_elements x n_windows
  Tensor detector_values_windows = F::unfold(detector_values, F::UnfoldFuncOptions({win_height, win_width}).stride({win_height, win_width}));
  Tensor softmax_attention = torch::softmax(detector_values_windows / temperature, 1);  

  // B x n_windows
  Tensor expected_v = torch::sum(v_windows * softmax_attention, 1);  
  Tensor expected_u = torch::sum(u_windows * softmax_attention, 1);

  // B x 2 x n_windows
  Tensor keypoints_2D = torch::stack({expected_u, expected_v}, 2).transpose(1, 2);  

  return keypoints_2D;
}

int main() {
  torch::jit::script::Module net = torch::jit::load("/home/lamlam/data/cpp_data/multiseason.pt");
  
  //Replace by an image tensor
  torch::Tensor x = torch::rand({1, 3, 384,512});
  
  std::vector<torch::jit::IValue> input;
  input.push_back(x);

  //Forward pass through neural network
  c10::IValue out = net.forward(input);

  // CPUFloatType{1,1,384,512}
  Tensor detector_scores = out.toTuple()->elements()[0].toTensor();
  // CPUFloatType{1,1,384,512}
  Tensor scores = out.toTuple()->elements()[1].toTensor();
  // CPUFloatType{1,496,384,512}
  Tensor descriptors = out.toTuple()->elements()[2].toTensor();
  
  //Get 2D keypoint coordinates from detector scores, Bx2xN using keypoint block
  KeypointBlock block = KeypointBlock(16,16,384,512);
  Tensor keypoints = block.forward(detector_scores);

  return 0;
}