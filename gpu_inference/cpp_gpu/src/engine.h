#pragma once
#include </home/lamlam/miniconda3/envs/opencv-cuda-env/include/opencv4/opencv2/opencv.hpp>
#include </home/lamlam/miniconda3/envs/opencv-cuda-env/include/opencv4/opencv2/core/cuda.hpp>
#include </home/lamlam/miniconda3/envs/opencv-cuda-env/include/opencv4/opencv2/cudawarping.hpp>
#include </home/lamlam/miniconda3/envs/opencv-cuda-env/include/opencv4/opencv2/cudaarithm.hpp>
#include "NvInfer.h"
#include <string>
using namespace std;
using namespace cv;

// Class to extend TensorRT logger
class Logger : public nvinfer1::ILogger 
{
    void log (Severity severity, const char* msg) noexcept override;
};

class Engine 
{
    public:
        void loadNetwork(string trtFilePath);
        void runInference(cuda::GpuMat input, vector<vector<vector<float>>> featureVectors);
        const std::vector<nvinfer1::Dims3>& getInputDims() const { return m_inputDims; };
    private:
        unique_ptr<nvinfer1::IRuntime> m_runtime = nullptr;
        unique_ptr<nvinfer1::ICudaEngine> m_engine = nullptr;
        unique_ptr<nvinfer1::IExecutionContext> m_context = nullptr;
        Logger m_logger;

         // Holds pointers to the input and output GPU buffers
        vector<void*> m_buffers;
        std::vector<nvinfer1::Dims3> m_inputDims;
        vector<nvinfer1::Dims> m_outputDims;

};

