#include "engine.h"
#include </home/lamlam/miniconda3/envs/opencv-cuda-env/include/opencv4/opencv2/opencv.hpp>
#include <chrono>

typedef std::chrono::high_resolution_clock Clock;
using namespace std;
using namespace cv;


int main() 
{
    // For batch size 1 and FP16
    //Set parameters
    string trtpath = "/home/lamlam/data/cpp_data/multiseason_layer16.trt";

    //Could do proper getDeviceCount stuff later
    cudaSetDevice(0);
    Engine engine;
    engine.loadNetwork(trtpath);

    const string inputImage = "/Volumes/oridatastore09/ThirdPartyData/utias/multiseason/run_000001/images/left/000057.png";
    auto cpuImg = imread(inputImage);
    cvtColor(cpuImg, cpuImg, cv::COLOR_BGR2RGB);

    // Upload to GPU memory
    cuda::GpuMat img;
    img.upload(cpuImg);
    const auto& inputDims = engine.getInputDims(); //inputDims[0].d[0] and d[1] and d[2] are 3,384,512

    //Resize. Height and width of resized and img are the same? -> doesn't do anything
    cuda::GpuMat resized;
    cuda::resize(img, resized, Size(inputDims[0].d[2], inputDims[0].d[1]));
    
    // OpenCV reads images into memory in NHWC format, while TensorRT expects images in NCHW format. 
    //Converts NHWC to NCHW.
    cuda::GpuMat nchwImage;
    nchwImage.create(inputDims[0].d[1], inputDims[0].d[2], CV_8UC3);
    vector<cuda::GpuMat> nhwcChannels;
    cuda::split(resized, nhwcChannels);
    vector<cv::cuda::GpuMat> nchwChannels;
    for (int c = 0; c < inputDims[0].d[0]; ++c) 
    {
        cuda::GpuMat channel(inputDims[0].d[1], inputDims[0].d[2], CV_8UC1, nchwImage.ptr(0) + c *inputDims[0].d[1] * inputDims[0].d[2]);
        nchwChannels.push_back(channel);
    }
    cuda::merge(nchwChannels, nchwImage);

    //Scale between 0-1
    cuda::GpuMat scaled;
    nchwImage.convertTo(scaled, CV_32FC3, 1.f / 255.f);

    vector<cuda::GpuMat> input;
    input.emplace_back(move(scaled));
    vector<vector<vector<float>>> featureVectors;
    //The input fed into this function is already the right size and the scaled to 0-1
    // The default Engine::runInference method will normalize values between [0.f, 1.f]   
    engine.runInference(scaled, featureVectors);
    
    /*
    size_t numIterations = 3;

    // Benchmark the inference time
    auto t1 = Clock::now();
    for (size_t i = 0; i < numIterations; ++i) {
        featureVectors.clear();
        engine.runInference(scaled, featureVectors);
    }
    auto t2 = Clock::now();
    double totalTime = chrono::duration_cast<chrono::milliseconds>(t2 - t1).count();

    cout << "Average time per inference in ms: " << totalTime / numIterations<<endl;

    //Get outputs 
    */
    return 0;
}


