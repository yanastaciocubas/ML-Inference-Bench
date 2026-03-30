import torch
from models import bert

def export(output_path="results/bert.onnx"):
    model, sample_input = bert.load()
    torch.onnx.export(
        model,
        (sample_input,),
        output_path,
        opset_version=17,
        input_names=["input_ids"],
        output_names=["output"],
        dynamic_axes={"input_ids": {0: "batch_size"}, "output": {0: "batch_size"}}
    )
    print(f"BERT exported to {output_path}")