from io import BytesIO
from PIL import Image
from typing import Dict
from torchvision import transforms
from torch import nn, max, Tensor
import torch

def preprocessor(data: BytesIO) -> Tensor:
    """Preprocesses an input image for neural network classification.

    Args:
        data (BytesIO): Input image as bytes buffer.

    Returns:
        Tensor: Preprocessed image tensor of shape (1, 3, 224, 224) with normalized values.

    Processing steps:
        1. Opens and converts image to RGB
        2. Resizes to 224x224
        3. Converts to tensor
        4. Normalizes using ImageNet means and stds
        5. Adds batch dimension
    """
    img = Image.open(data).convert("RGB")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),   # fixed size
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    return transform(img).unsqueeze(0)

def ClassificationPipeline(data: BytesIO, model: nn.Module, device) -> Dict[str, str| float]:
    """Runs complete classification pipeline on brain MRI image.

    Args:
        data (BytesIO): Input image as bytes buffer
        model (nn.Module): PyTorch classification model
        device: Device to run inference on (CPU/CUDA)

    Returns:
        Dict[str, str|float]: Dictionary containing:
            - 'class': Predicted tumor class label
            - 'confidence': Confidence score for prediction

    The function preprocesses the image, runs inference using the provided model,
    and returns the predicted tumor class with confidence score.
    """
    labels = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]

    pdata  = preprocessor(data)
    pdata = pdata.to(device)
    model.eval()
    with torch.no_grad():
        pred = model(pdata)
        probs = nn.functional.softmax(pred, dim=1)
        conf, class_idx = max(probs, dim=1)
    

    return  {"class" : labels[class_idx.item()], "confidence" : conf.item()}

