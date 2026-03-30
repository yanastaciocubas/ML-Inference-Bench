import torch
from ultralytics import YOLO

def load():
    model = YOLO("yolov8n.pt")
    sample_input = torch.randn(1, 3, 640, 640)
    return model.model, sample_input