import tensorrt as trt
import onnx
from onnx import shape_inference

def load_engine_and_get_input_size(engine_path):
    # Load the TensorRT engine from file
    with open(engine_path, "rb") as f:
        engine_data = f.read()
    runtime = trt.Runtime(trt.Logger())
    engine = runtime.deserialize_cuda_engine(engine_data)

    # Get the input size
    num_inputs = engine.num_bindings
    print("Number of bindings: " + str(num_inputs))
    """
    print("Input 1 size: ", engine.get_binding_shape(0))
    print("Input 2 size: ",engine.get_binding_shape(1))
    print("Output 1 size:", engine.get_binding_shape(2))
    print("Output 2 size:", engine.get_binding_shape(3))
    print("Output 3 size:", engine.get_binding_shape(4))
    """

    # Clean up
    del engine
    del runtime

def get_onnx_size(onnx_path):
    # Load the ONNX model
    model = onnx.load(onnx_path)

    # Perform shape inference
    inferred_model = shape_inference.infer_shapes(model)

    # Get the input and output shapes
    input_shape = inferred_model.graph.input[0].type.tensor_type.shape.dim
    output_shape = inferred_model.graph.output[0].type.tensor_type.shape.dim

    # Extract the sizes from the shapes
    input_size = [dim.dim_value for dim in input_shape]
    output_size = [dim.dim_value for dim in output_shape]

    # Print the sizes
    print("Input size:", input_size)
    print("Output size:", output_size)

# Provide the path to your TensorRT engine file
engine_path = "/home/lamlam/data/cpp_data/multiseason_layer16.trt"
#engine_path = "/home/lamlam/tensorrt-cpp-api/multiseason_layer_16.engine.NVIDIAGeForceRTX3090Ti.fp16.1.1.4000000000"
onnx_path = "/home/lamlam/data/cpp_data/multiseason_layer_16.onnx"

# Load the engine and get the input size
#get_onnx_size(onnx_path)
load_engine_and_get_input_size(engine_path)
