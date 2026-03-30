import torch
from models import whisper

def export(output_path="results/whisper.onnx"):
    model, sample_input = whisper.load()
    torch.onnx.export(
        model.encoder,
        sample_input,
        output_path,
        opset_version=17,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
    )
    print(f"Whisper exported to {output_path}")