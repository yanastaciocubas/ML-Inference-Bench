import torch
from models import efficientnet

def export(output_path="results/efficientnet.onnx"):
    model, sample_input = efficientnet.load()
    torch.onnx.export(
        model,
        sample_input,
        output_path,
        opset_version=17,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
    )
    print(f"EfficientNet exported to {output_path}")