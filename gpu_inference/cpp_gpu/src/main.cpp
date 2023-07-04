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
    const string trtpath = "/home/lamlam/data/cpp_data/multiseason_layer16.trt";
    cudaSetDevice(0);
    const string inputImage = "/Volumes/oridatastore09/ThirdPartyData/utias/multiseason/run_000001/images/left/000057.png";
    
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

    auto t2_preprocess = Clock::now();
    double totalTime_preprocess = chrono::duration_cast<chrono::milliseconds>(t2_preprocess - t1_preprocess).count();
    cout << "Time taken to pre-process the image: " << totalTime_preprocess <<endl;
    const vector<nvinfer1::Dims>& outputDims = engine.getOutputDims();
    vector<vector<float>> outputEmpty;
    for (int32_t i = 0; i < outputDims.size(); ++i) 
    {
        vector<float> output_n(outputDims[i].d[0]*outputDims[i].d[1]*outputDims[i].d[2]*outputDims[i].d[3]);
        outputEmpty.emplace_back(output_n);
    }
    //The input fed into this function is already the right size and the scaled to 0-1
    //Don't count timing for first inference as it takes longer
    vector<vector<float>> output; 
    engine.runInference(mfloat, outputEmpty, output);

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

    //should use a tuple
    size_t numIterations = 3;
    
    auto t1 = Clock::now();
    // Benchmark the inference time
    for (size_t i = 0; i < numIterations; ++i) {
        output.clear();
        engine.runInference(mfloat, outputEmpty, output);

        /*
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
        */
    }
    auto t2 = Clock::now();
    double totalTime = chrono::duration_cast<chrono::milliseconds>(t2 - t1).count();
    cout << "Time per inference in ms: " << totalTime/numIterations<<endl;

    //Get outputs 
    return 0;
}


