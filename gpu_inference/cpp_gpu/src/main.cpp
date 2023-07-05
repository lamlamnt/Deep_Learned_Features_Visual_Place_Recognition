#include "engine.h"
#include </home/lamlam/miniconda3/envs/opencv-cuda-env/include/opencv4/opencv2/opencv.hpp>
#include <chrono>

typedef std::chrono::high_resolution_clock Clock;
using namespace std;
using namespace cv;

int main() 
{
    //For batch size 1 and FP16
    //Set parameters
    const string trtpath = "/home/lamlam/data/cpp_data/multiseason_layer16.trt";
    //const string trtpath = "/home/lamlam/data/cpp_data/multiseason_layer16_FP32.trt";
    //const string trtpath = "/home/lamlam/tensorrt-cpp-api/multiseason_layer_16.engine.NVIDIAGeForceRTX3090Ti.fp16.1.1.4000000000";
    cudaSetDevice(0);
    const string inputImage = "/Volumes/oridatastore09/ThirdPartyData/utias/multiseason/run_000010/images/left/000200.png";
    //Not including the first iteration
    size_t numIterations = 3;

    Engine engine;
    engine.loadNetwork(trtpath);
      
    auto cpuImg = imread(inputImage);
    cvtColor(cpuImg, cpuImg, cv::COLOR_BGR2RGB);

    //Start timing the pre-processing of the image 
    auto t1_preprocess = Clock::now();
    // Upload to GPU 
    cuda::GpuMat img;
    img.upload(cpuImg);

    //Resizing is only necessary if network and input images are different shapes
    //Flattened into a 1d array
    // Copy over the input data and perform the preprocessing
    cv::cuda::GpuMat gpu_dst(1, img.rows * img.cols, CV_8UC3);

    // OpenCV reads images into memory in NHWC format, while TensorRT expects images in NCHW format. 
    // The following code converts NHWC to NCHW.
    size_t width = img.cols * img.rows;
    std::vector<cv::cuda::GpuMat> input_channels{
        cv::cuda::GpuMat(img.rows, img.cols, CV_8U, &(gpu_dst.ptr()[0])),
        cv::cuda::GpuMat(img.rows, img.cols, CV_8U, &(gpu_dst.ptr()[width])),
        cv::cuda::GpuMat(img.rows, img.cols, CV_8U, &(gpu_dst.ptr()[width * 2]))};
    cv::cuda::split(img, input_channels);  

    cv::cuda::GpuMat mfloat;
    gpu_dst.convertTo(mfloat, CV_32FC3, 1.f / 255.f);
    //mfloat has 3 channels, 1 row, and 196608 cols

    auto t2_preprocess = Clock::now();
    double totalTime_preprocess = chrono::duration_cast<chrono::milliseconds>(t2_preprocess - t1_preprocess).count();
    cout << "Time taken to pre-process the image: " << totalTime_preprocess <<endl;

    //The input fed into this function is already the right size (flattened into 1d vectors), scaled to 0-1, and not normalized
    //Don't count timing for first inference as it takes longer
    vector<vector<float>> output; 
    engine.runInference(mfloat, output);

    //Post Process - convert outputs from 1d back to 3d vectors

    //Prints the first few elements of the outputs
    for (size_t outputNum = 0; outputNum < output.size(); ++outputNum) 
        {
            cout << "output " << outputNum << std::endl;
            int i = 0;
            for (const auto &e:  output[outputNum]) 
            {
                std::cout << e << " ";
                if (++i == 20) {
                    std::cout << "...";
                    break;
                }
            }
            std::cout << "\n" << std::endl;
        }

    
    auto t1 = Clock::now();
    // Benchmark the inference time
    for (size_t i = 0; i < numIterations; ++i) {
        output.clear();
        engine.runInference(mfloat, output);
    }
    auto t2 = Clock::now();
    double totalTime = chrono::duration_cast<chrono::milliseconds>(t2 - t1).count();
    cout << "Time per inference in ms: " << totalTime/numIterations<<endl;

    return 0;
}


