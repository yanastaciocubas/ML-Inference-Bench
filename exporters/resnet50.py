import torch
from models import resnet50

def export(output_path="results/resnet50.onnx"):
    model, sample_input = resnet50.load()
    torch.onnx.export(
        model,
        sample_input,
        output_path,
        opset_version=17,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
    )
    print(f"ResNet50 exported to {output_path}")