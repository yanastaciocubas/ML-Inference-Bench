import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import pynvml

def measure(engine_path, sample_input):
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    logger = trt.Logger(trt.Logger.WARNING)
    
    # Baseline memory before loading engine
    baseline = pynvml.nvmlDeviceGetMemoryInfo(handle).used
    
    # Load the engine
    with open(engine_path, "rb") as f:
        runtime = trt.Runtime(logger)
        engine = runtime.deserialize_cuda_engine(f.read())
    
    context = engine.create_execution_context()
    
    # Allocate and run
    input_data = sample_input.numpy().astype(np.float32)
    d_input = cuda.mem_alloc(input_data.nbytes)
    d_output = cuda.mem_alloc(input_data.nbytes)
    cuda.memcpy_htod(d_input, input_data)
    context.execute_v2([int(d_input), int(d_output)])
    
    # Measure peak memory after inference
    peak = pynvml.nvmlDeviceGetMemoryInfo(handle).used
    memory_used_mb = (peak - baseline) / 1024 / 1024
    
    return {"memory_mb": round(memory_used_mb, 2)}