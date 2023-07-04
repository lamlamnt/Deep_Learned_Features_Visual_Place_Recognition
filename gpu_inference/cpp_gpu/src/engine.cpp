#include <iostream>
#include <fstream>

#include "engine.h"
#include <NvOnnxParser.h>
#include <typeinfo>

typedef std::chrono::high_resolution_clock Clock;
using namespace nvinfer1;
using namespace std;

//TensorRT’s builder and engine require a logger
//Can change this to a different setting later + look at logger when exporting network in python too
void Logger::log(Severity severity, const char *msg) noexcept 
{
    // Only log Warnings or more important.
    if (severity <= Severity::kWARNING) {
        cout << msg << endl;
    }
}

void Engine::loadNetwork(string trtFilePath) 
{
    //Read the serialized model into an input file stream 
    ifstream file(trtFilePath, std::ios::binary | std::ios::ate);
    streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    vector<char> buffer(size);
    file.read(buffer.data(), size);

    m_runtime = unique_ptr<IRuntime> {createInferRuntime(m_logger)};
    m_engine = unique_ptr<nvinfer1::ICudaEngine>(m_runtime->deserializeCudaEngine(buffer.data(), buffer.size()));
    m_context = unique_ptr<nvinfer1::IExecutionContext>(m_engine->createExecutionContext());

    int numBindings = m_engine->getNbBindings();
    
    //Allocate GPU memory for input and output buffers
    //Could use cuda.mem_alloc instead of cudaMallocAsync and then synchronzie afterwards
    m_buffers.resize(numBindings);
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    for (int i = 0; i < numBindings; ++i) 
    {
        if (m_engine->bindingIsInput(i)) 
        {
            auto inputBindingDims = m_engine->getBindingDimensions(i);

            // Allocate memory for the input
            cudaMallocAsync(&m_buffers[i], inputBindingDims.d[1] * inputBindingDims.d[2] * inputBindingDims.d[3] * sizeof(float), stream);
            // Store the input dims for later use
            m_inputDims.emplace_back(inputBindingDims.d[1], inputBindingDims.d[2], inputBindingDims.d[3]);
        }
        else 
        {
            // The binding is an output
            auto outputDims = m_engine->getBindingDimensions(i);
            m_outputDims.push_back(outputDims);
            //Allocate memory for the output
            cudaMallocAsync(&m_buffers[i], outputDims.d[1]*outputDims.d[2]*outputDims.d[3]* sizeof(float), stream);
        }
    }
    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);
}


void Engine::runInference(cuda::GpuMat input, vector<vector<float>>& outputEmpty, vector<vector<float>>& output)
{
    cudaStream_t inferenceCudaStream;
    cudaStreamCreate(&inferenceCudaStream);

    auto *dataPointer = input.ptr<void>();
    
    //Copy from GPU to GPU
    cudaMemcpyAsync(m_buffers[0], dataPointer,input.cols * input.rows *input.channels() * sizeof(float),cudaMemcpyDeviceToDevice, inferenceCudaStream);

    //The actual inference bit
    m_context->enqueueV2(m_buffers.data(), inferenceCudaStream, nullptr);

    //Copy from GPU to CPU
    vector<vector<float>> batchOutputs{};
    for (int32_t outputBinding = 1; outputBinding < m_engine->getNbBindings(); ++outputBinding) 
    {
        int outputLenFloat = m_outputDims[outputBinding-1].d[0]*m_outputDims[outputBinding-1].d[1]*m_outputDims[outputBinding-1].d[2]*m_outputDims[outputBinding-1].d[3];
        //outputEmpty[outputBinding-1].resize(outputLenFloat);
        cudaMemcpyAsync(outputEmpty[outputBinding-1].data(), static_cast<char*>(m_buffers[outputBinding]), outputLenFloat * sizeof(float), cudaMemcpyDeviceToHost, inferenceCudaStream);
        output.push_back(outputEmpty[outputBinding-1]);
    }
    /*
    for (int32_t outputBinding = 1; outputBinding < m_engine->getNbBindings(); ++outputBinding) 
    {
        int outputLenFloat = m_outputDims[outputBinding-1].d[0]*m_outputDims[outputBinding-1].d[1]*m_outputDims[outputBinding-1].d[2]*m_outputDims[outputBinding-1].d[3];
        auto t1_d = Clock::now();
        vector<float> output(outputLenFloat);
        //output.resize(outputLenFloat);
        auto t2_d = Clock::now();
        double totalTime_move4 = chrono::duration_cast<chrono::milliseconds>(t2_d - t1_d).count();
        //cout << "Time to do resizing: " << totalTime_move4 <<endl;
        //Uses pointer arithmetic
        cudaMemcpyAsync(output.data(), static_cast<char*>(m_buffers[outputBinding]), outputLenFloat * sizeof(float), cudaMemcpyDeviceToHost, inferenceCudaStream);
        batchOutputs.emplace_back(std::move(output));
    }
    */
    
    cudaStreamSynchronize(inferenceCudaStream);
    cudaStreamDestroy(inferenceCudaStream);

    //Should this be included
    //outputEmpty.clear();
}

void Engine::checkCudaErrorCode(cudaError_t code) {
    if (code != 0) {
        std::string errMsg = "CUDA operation failed with code: " + std::to_string(code) + "(" + cudaGetErrorName(code) + "), with message: " + cudaGetErrorString(code);
        std::cout << errMsg << std::endl;
    }
}

Engine::~Engine() {
    // Free the GPU memory
    for (auto & buffer : m_buffers) {
        checkCudaErrorCode(cudaFree(buffer));
    }
    m_buffers.clear();
}


