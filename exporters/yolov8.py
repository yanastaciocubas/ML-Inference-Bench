import torch
from models import yolov8

def export(output_path="results/yolov8.onnx"):
    model, sample_input = yolov8.load()
    torch.onnx.export(
        model,
        sample_input,
        output_path,
        opset_version=17,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
    )
    print(f"YOLOv8 exported to {output_path}")