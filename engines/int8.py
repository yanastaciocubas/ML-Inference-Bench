import tensorrt as trt
import numpy as np

class Int8Calibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, sample_input, cache_file="calibration.cache"):
        super().__init__()
        self.data = sample_input.numpy().astype(np.float32)
        self.cache_file = cache_file
        self.current_index = 0

    def get_batch_size(self):
        return 1

    def get_batch(self, names):
        if self.current_index >= len(self.data):
            return None
        batch = self.data[self.current_index:self.current_index + 1]
        self.current_index += 1
        return [batch.ctypes.data]

    def read_calibration_cache(self):
        try:
            with open(self.cache_file, "rb") as f:
                return f.read()
        except FileNotFoundError:
            return None

    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            f.write(cache)

def build(onnx_path, engine_path, sample_input):
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    config = builder.create_builder_config()
    config.max_workspace_size = 1 << 30  # 1GB

    # Enable INT8 and attach calibrator
    config.set_flag(trt.BuilderFlag.INT8)
    config.int8_calibrator = Int8Calibrator(sample_input)

    # Parse the ONNX model
    with open(onnx_path, "rb") as f:
        parser.parse(f.read())

    # Build and serialize the engine
    engine = builder.build_serialized_network(network, config)

    with open(engine_path, "wb") as f:
        f.write(engine)

    print(f"INT8 engine saved to {engine_path}")