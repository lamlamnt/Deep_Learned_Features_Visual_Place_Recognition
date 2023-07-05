import os
import torch
import torchvision
import sys
import torchvision.models as models
sys.path.append("/home/lamlam/code/deep_learned_visual_features/src/model")
from unet import UNet
import tensorrt as trt
import numpy as np
import onnx

torch.cuda.is_available_lazy = True

def build_engine(onnx_model_path, tensorrt_engine_path, engine_precision, img_size):
    # Builder
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    #network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    network = builder.create_network(1)
    profile = builder.create_optimization_profile()
    config = builder.create_builder_config()
    
    # Set FP16 
    if engine_precision == 'FP16':
        config.set_flag(trt.BuilderFlag.FP16)

    if engine_precision == 'INT8':
        config.set_flag(trt.BuilderFlag.INT8)
    
    # Onnx parser
    parser = trt.OnnxParser(network, logger)
    with open(onnx_model_path, "rb") as model:
        parser.parse(model.read())
        print("Succeeded parsing .onnx file!")
    
    # Input
    # Create an input tensor and add it to the network
    input_shape = img_size  # Example input shape
    #input_tensor = network.add_input('input', trt.float16, input_shape)

    # Create a profile and set the dynamic shape for the input tensor
    profile = builder.create_optimization_profile()
    profile.set_shape('input', input_shape, input_shape, input_shape)

    # Attach the optimization profile to the builder
    config.add_optimization_profile(profile)
    
    # Write engine
    engineString = builder.build_serialized_network(network, config)
    if engineString == None:
        print("Failed building engine!")
        exit()
    print("Succeeded building engine!")
    with open(tensorrt_engine_path, "wb") as f:
        f.write(engineString)

#Load trained network from .pth file
model = UNet(3,1,16)
file_path = "/home/lamlam/data/networks/network_multiseason_layer16.pth"
torch.save(model.state_dict(), file_path)
model.eval()
example = torch.rand(1, 3, 384, 512)

#Convert to ONNX model 
onnx_path = "/home/lamlam/data/cpp_data/multiseason_layer_16.onnx"  # Specify the path and filename for the ONNX model
torch.onnx.export(model, example, onnx_path)

#Compare ONNX and Pytorch results

#Load ONNX model
onnx_model = onnx.load(onnx_path)
tensorrt_engine_path = "/home/lamlam/data/cpp_data/multiseason_layer16.trt"
build_engine(onnx_path, tensorrt_engine_path, 'FP16', (1,3,384,512))



