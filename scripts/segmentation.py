from io import BytesIO
from torch import nn, Tensor
import torch
from torchvision import transforms
from typing import Dict, Tuple
from PIL import Image

def preprocess(flair: BytesIO, t1ce: BytesIO) -> Tuple[Tensor, Tensor]:

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0,0), std=(1,1))
    ])

    if flair:
        flair = Image.open(flair)
        t1 = transform(flair)
    else:
        t1 = torch.zeros((1, 256, 256))

    if t1ce:
        t1ce = Image.open(t1ce)
        t2 = transform(t1ce)
    else:
        t2 = torch.zeros((1, 256, 256))


    return t1, t2


def SegmentationPipeline2D(flair : BytesIO, t1ce: BytesIO, model: nn.Module, device) -> Dict[str , Image | float]:
    d1 , d2 = preprocess(flair, t1ce)
    d1.to(device)
    d2.to(device)

    model.eval()
    with torch.no_grad():
        pred = model(d1, d2)




def SegmentationPipeline3D(flair, t1ce):
    return None