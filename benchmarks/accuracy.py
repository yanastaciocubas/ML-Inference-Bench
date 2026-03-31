import torch
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np

def measure(engine_path, pytorch_model, sample_input):
    logger = trt.Logger(trt.Logger.WARNING)
    
    # Load the engine
    with open(engine_path, "rb") as f:
        runtime = trt.Runtime(logger)
        engine = runtime.deserialize_cuda_engine(f.read())
    
    context = engine.create_execution_context()
    
    # Run TensorRT inference
    input_data = sample_input.numpy().astype(np.float32)
    output_data = np.empty_like(input_data)
    d_input = cuda.mem_alloc(input_data.nbytes)
    d_output = cuda.mem_alloc(output_data.nbytes)
    cuda.memcpy_htod(d_input, input_data)
    context.execute_v2([int(d_input), int(d_output)])
    cuda.memcpy_dtoh(output_data, d_output)
    
    # Run PyTorch FP32 inference as baseline
    with torch.no_grad():
        pytorch_output = pytorch_model(sample_input).numpy()
    
    # Compare outputs using cosine similarity
    trt_flat = output_data.flatten()
    pt_flat = pytorch_output.flatten()
    cosine_sim = float(np.dot(trt_flat, pt_flat) / (np.linalg.norm(trt_flat) * np.linalg.norm(pt_flat)))
    
    return {"cosine_similarity": round(cosine_sim, 4)}