import torch
from torchvision import models
from torch import nn, Tensor
from typing import List, Tuple
import numpy as np
from collections import Counter
from .config import MODEL_PATHS

def load_model() -> nn.Module:
    """Load the 2D classifier model."""
    model = models.mobilenet_v2(pretrained=True)
    model.classifier[1] = nn.Linear(model.last_channel, 4)
    
    try:
        model.load_state_dict(torch.load(MODEL_PATHS["classifier"]))
        model.eval()
        return model
    except Exception as e:
        raise RuntimeError(f"Failed to load classifier model: {str(e)}")

def classify_slices(slice_list: List[Tensor], model: nn.Module, device: str = "cpu") -> List[str]:
    """
    Classify a list of 2D slice tensors using a 2D classifier model.

    Args:
        slice_list (List[Tensor]): List of tensors, each [1,1,H,W]
        model (nn.Module): PyTorch model for classification
        device (str): 'cpu' or 'cuda'

    Returns:
        List[str]: Predicted labels for each slice
    """
    labels = ['Glioma', 'Meningioma', 'Pituitary', 'No Tumor']

    if not slice_list:
        return []

    # Stack slices into a batch tensor: [batch_size, 1, H, W]
    batch = torch.cat(slice_list, dim=0).to(device)

    with torch.no_grad():
        outputs = model(batch)                  # [batch_size, num_classes]
        preds = outputs.argmax(dim=1).cpu()    # indices of max class per slice

    # Convert indices to label strings
    return [labels[i] for i in preds]

def classify_volume(
    slice_list: List[Tensor],
    model: nn.Module,
    device: str = "cpu",
    batch_size: int = 8,
    sample_slices: int = 20
) -> Tuple[str, float]:
    """
    Classify a 3D volume by analyzing multiple slices and returning the most common prediction.

    Args:
        slice_list (List[Tensor]): List of tensors [1,1,H,W]
        model (nn.Module): Classification model
        device (str): 'cpu' or 'cuda'
        batch_size (int): Number of slices per batch
        sample_slices (int): Number of slices to sample

    Returns:
        Tuple[str, float]: (most_common_label, confidence_percentage)
    """
    labels = ['Glioma', 'Meningioma', 'Pituitary', 'No Tumor']

    if not slice_list:
        return "No Tumor", 0.0

    # Sample slices if volume is large
    if len(slice_list) > sample_slices:
        indices = np.linspace(0, len(slice_list) - 1, sample_slices, dtype=int)
        slice_list = [slice_list[i] for i in indices]

    # Batch process slices
    all_preds = []
    with torch.no_grad():
        for i in range(0, len(slice_list), batch_size):
            batch = torch.cat(slice_list[i:i+batch_size], dim=0).to(device)
            outputs = model(batch)
            preds = outputs.argmax(dim=1).cpu().tolist()
            all_preds.extend(preds)

    # Count predictions and calculate confidence
    pred_counter = Counter(all_preds)
    most_common_idx, count = pred_counter.most_common(1)[0]
    confidence = (count / len(all_preds)) * 100.0

    return labels[most_common_idx], confidence