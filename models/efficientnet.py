import torch
import torchvision.models as models

def load():
    model = models.efficientnet_b0(weights="IMAGENET1K_V1")
    model.eval()
    sample_input = torch.randn(1, 3, 224, 224)
    return model, sample_input