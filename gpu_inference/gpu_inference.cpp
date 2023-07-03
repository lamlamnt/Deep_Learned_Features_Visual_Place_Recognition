#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <iostream>
using namespace std;
using namespace nvinfer1;

int main()
{
    //Load Tensor Engine from .trt file
    const std::string engineFile = "/home/lamlam/data/cpp_data/multiseason_layer16.trt";

    // Create a Logger to capture TensorRT messages
    ILogger* logger = new Logger(ILogger::Severity::kINFO);

    // Create a runtime object
    IRuntime* runtime = createInferRuntime(*logger);
    cout << "Success";

/*
const int batchSize = 1;  // Specify the batch size
const int inputSize = ...;  // Specify the size of the input tensor
const int outputSize = ...;  // Specify the size of the output tensor

void* buffers[2];  // Array to hold input and output pointers
cudaMalloc(&buffers[0], batchSize * inputSize);
cudaMalloc(&buffers[1], batchSize * outputSize);

float* inputData = ...;  // Input data in CPU memory
cudaMemcpy(buffers[0], inputData, batchSize * inputSize, cudaMemcpyHostToDevice);

context->execute(batchSize, buffers);
*/
}
