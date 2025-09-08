from torchvision import models
from torch import load, nn
import torch
from typing import Tuple
import segmentation_models_pytorch as smp

def load_models()-> Tuple[torch.device ,nn.Module, nn.Module]:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    classifier = models.mobilenet_v2().to(device)
    classifier.load_state_dict(load("../weights/brain_tumor_classifier.pth", map_location=device))

    segmentation = smp.Unet(encoder_name='timm-efficientnet-b0', in_channels=2,classes =4).to(device)
    segmentation.load_state_dict(load("../weights/brain_tumor_segmenter.pth", map_location=device))

    return device, classifier, segmentation