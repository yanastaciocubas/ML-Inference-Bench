import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np

def measure(engine_path, sample_input, num_runs=1000, warmup=100):
    logger = trt.Logger(trt.Logger.WARNING)
    
    # Load the engine
    with open(engine_path, "rb") as f:
        runtime = trt.Runtime(logger)
        engine = runtime.deserialize_cuda_engine(f.read())
    
    context = engine.create_execution_context()
    
    # Allocate GPU memory
    input_data = sample_input.numpy().astype(np.float32)
    d_input = cuda.mem_alloc(input_data.nbytes)
    d_output = cuda.mem_alloc(input_data.nbytes)
    
    # Warmup runs
    for _ in range(warmup):
        cuda.memcpy_htod(d_input, input_data)
        context.execute_v2([int(d_input), int(d_output)])
    
    # Measure latency using CUDA events
    latencies = []
    for _ in range(num_runs):
        start = cuda.Event()
        end = cuda.Event()
        start.record()
        context.execute_v2([int(d_input), int(d_output)])
        end.record()
        end.synchronize()
        latencies.append(start.time_till(end))
    
    latencies = np.array(latencies)
    
    return {
        "p50_ms": float(np.percentile(latencies, 50)),
        "p95_ms": float(np.percentile(latencies, 95)),
        "p99_ms": float(np.percentile(latencies, 99)),
        "mean_ms": float(np.mean(latencies))
    }