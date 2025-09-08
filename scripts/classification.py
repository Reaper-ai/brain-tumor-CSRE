from io import BytesIO
from PIL import Image
from typing import Dict
from torchvision import transforms
from torch import nn, max, Tensor
import torch

def preprocessor(data: BytesIO) -> Tensor:
    img = Image.open(data).convert("RGB")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),   # fixed size
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    return transform(img).unsqueeze(0)

def ClassificationPipeline(data: BytesIO, model: nn.Module, device) -> Dict[str, str| float]:
    labels = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]

    pdata  = preprocessor(data)
    pdata = pdata.to(device)
    model.eval()
    with torch.no_grad():
        pred = model(pdata)
        probs = nn.functional.softmax(pred, dim=1)
        conf, class_idx = max(probs, dim=1)


    return  {"class" : labels[class_idx.item()], "confidence" : conf.item()}

