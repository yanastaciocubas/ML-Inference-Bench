import torch
from transformers import WhisperModel

def load():
    model = WhisperModel.from_pretrained("openai/whisper-tiny")
    model.eval()
    sample_input = torch.randn(1, 80, 3000)
    return model, sample_input