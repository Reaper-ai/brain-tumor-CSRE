from torchvision import models
from torch import load, nn
import torch
from typing import Tuple
import segmentation_models_pytorch as smp


def load_models() -> Tuple[torch.device, nn.Module, nn.Module]:
    """Loads the models for classification and segmentation.

    Returns:
        Tuple containing:
            torch.device: Device to run models on (CUDA if available, else CPU)
            nn.Module: Classification model (MobileNetV2) for tumor type prediction
            nn.Module: Segmentation model (UNet) for tumor region segmentation
    """

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    classifier = models.mobilenet_v2(weights=None).to(device)
    classifier.classifier[1] = nn.Linear(classifier.last_channel, 4).to(device)
    classifier.load_state_dict(load("weights/brain_tumor_classifier.pth", map_location=device))

    segmentation = smp.Unet(
        encoder_name='timm-efficientnet-b0',
        in_channels=2,
        classes=4,
        encoder_weights=None  # important!
    ).to(device)
    segmentation.load_state_dict(load("weights/brain_tumor_segmenter.pth", map_location=device))

    return device, classifier, segmentation
