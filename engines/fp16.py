import tensorrt as trt

def build(onnx_path, engine_path):
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    config = builder.create_builder_config()
    config.max_workspace_size = 1 << 30  # 1GB

    # Enable FP16 only if the GPU supports it
    if builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        print("FP16 enabled")
    else:
        print("FP16 not supported on this GPU, falling back to FP32")

    # Parse the ONNX model
    with open(onnx_path, "rb") as f:
        parser.parse(f.read())

    # Build and serialize the engine
    engine = builder.build_serialized_network(network, config)

    with open(engine_path, "wb") as f:
        f.write(engine)

    print(f"FP16 engine saved to {engine_path}")