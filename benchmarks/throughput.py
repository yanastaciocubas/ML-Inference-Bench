import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import time

def measure(engine_path, sample_input, batch_sizes=[1, 8, 32]):
    logger = trt.Logger(trt.Logger.WARNING)
    
    # Load the engine
    with open(engine_path, "rb") as f:
        runtime = trt.Runtime(logger)
        engine = runtime.deserialize_cuda_engine(f.read())
    
    context = engine.create_execution_context()
    results = {}
    
    for batch_size in batch_sizes:
        # Prepare batched input
        input_data = np.tile(sample_input.numpy(), (batch_size, 1, 1, 1)).astype(np.float32)
        d_input = cuda.mem_alloc(input_data.nbytes)
        d_output = cuda.mem_alloc(input_data.nbytes)
        cuda.memcpy_htod(d_input, input_data)
        
        # Measure throughput over 100 runs
        start = time.time()
        for _ in range(100):
            context.execute_v2([int(d_input), int(d_output)])
        elapsed = time.time() - start
        
        images_per_second = (100 * batch_size) / elapsed
        results[f"batch_{batch_size}"] = round(images_per_second, 2)
    
    return results