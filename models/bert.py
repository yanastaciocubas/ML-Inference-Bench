import torch
from transformers import BertModel

def load():
    model = BertModel.from_pretrained("bert-base-uncased")
    model.eval()
    sample_input = torch.randint(0, 100, (1, 128))
    return model, sample_input