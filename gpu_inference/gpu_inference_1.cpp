#include </home/lamlam/downloads/TensorRT-8.6.1.6/include/NvInfer.h>
#include <cuda_runtime_api.h>
#include <iostream>

#Load Tensor Engine from .trt file
std::ifstream engineFile("/home/lamlam/data/cpp_data/multiseason_layer16.trt", std::ios::binary);
engineFile.seekg(0, std::ios::end);
const size_t modelSize = engineFile.tellg();
engineFile.seekg(0, std::ios::beg);
std::vector<char> engineData(modelSize);
engineFile.read(engineData.data(), modelSize);
engineFile.close();

nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(gLogger);
nvinfer1::ICudaEngine* engine = runtime->deserializeCudaEngine(engineData.data(), modelSize, nullptr);

nvinfer1::IExecutionContext* context = engine->createExecutionContext();
print("Success")
