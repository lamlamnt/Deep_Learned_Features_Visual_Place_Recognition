//#include </home/lamlam/ENTER/lib/python3.10/site-packages/torch/include/torch/csrc/api/include/torch/torch.h>
//#include <iostream>
//using namespace std;
#include "keypoint_block.h"

/*
class KeypointBlock : public torch::nn::Module 
{
    public:
        int n;
};
*/

/*
class KeypointBlock : public torch::nn::Module 
{
    public:
        //Constructor with parameters
        KeypointBlock(int window_height, int window_width, int image_height, int image_width)
            : window_height(window_height), window_width(window_width),temperature(1.0) 
        {
            v_coords = torch::arange(0, image_height).unsqueeze(0).to(torch::kFloat);  // 1 x H
            u_coords = torch::arange(0, image_width).unsqueeze(0).to(torch::kFloat);  // 1 x W
            v_coords = v_coords.expand({1, image_width}).unsqueeze(0);  // 1 x H x W
            u_coords = u_coords.expand({1, image_height}).unsqueeze(0);  // 1 x W x H

            register_buffer("v_coords", v_coords);
            register_buffer("u_coords", u_coords);
        }

        torch::Tensor forward(const torch::Tensor& detector_values) 
        {
            int64_t batch_size = detector_values.size(0);
            int64_t height = detector_values.size(2);
            int64_t width = detector_values.size(3);

            auto v_windows = torch::unfold(v_coords.expand({batch_size, 1, height, width}),
                                        torch::nn::UnfoldOptions({window_height, window_width})
                                            .stride(window_height, window_width));  // B x n_window_elements x n_windows
            auto u_windows = torch::unfold(u_coords.expand({batch_size, 1, height, width}),
                                        torch::nn::UnfoldOptions({window_height, window_width})
                                            .stride(window_height, window_width));

            auto detector_values_windows = torch::unfold(detector_values,
                                                        torch::nn::UnfoldOptions({window_height, window_width})
                                                            .stride(window_height, window_width));  // B x n_wind_elements x n_windows

            auto softmax_attention = torch::softmax(detector_values_windows / temperature, dim=1);  // B x n_wind_elements x n_windows

            auto expected_v = torch::sum(v_windows * softmax_attention, dim=1);  // B x n_windows
            auto expected_u = torch::sum(u_windows * softmax_attention, dim=1);
            auto keypoints_2D = torch::stack({expected_u, expected_v}, dim=2).transpose(1, 2);  // B x 2 x n_windows

            return keypoints_2D;
        }
};
*/

