import numpy as np
import torch
import segmentation_models_pytorch as smp
from typing import List, Optional, Union
from pathlib import Path
from .config import MODEL_PATHS

class SegmentationError(Exception):
    """Custom exception for segmentation errors"""
    pass

def load_seg_model(modality: str, device: str = "cpu") -> smp.Unet:
    """Load a pre-trained UNet model for a specific modality."""
    modality = modality.lower()
    
    if modality not in MODEL_PATHS:
        raise ValueError(f"Invalid modality: {modality}")
    
    model = smp.Unet(
        encoder_name="timm-efficientnet-b0",
        encoder_weights="imagenet",
        in_channels=1,
        classes=4
    )
    
    try:
        model.load_state_dict(torch.load(MODEL_PATHS[modality], map_location=device))
        model.eval()
        return model
    except Exception as e:
        raise RuntimeError(f"Failed to load {modality} model: {str(e)}")

def segment_image(
    slices: List[torch.Tensor],
    model: smp.Unet,
    device: str = "cpu",
    batch_size: int = 8
) -> List[torch.Tensor]:
    """
    Segment 2D slices using the provided model.

    Args:
        slices: List of image tensors [1,1,H,W]
        model: Segmentation model
        device: Device to run inference on
        batch_size: Batch size for processing

    Returns:
        List of segmentation masks [classes,H,W]
    """
    if not slices:
        raise ValueError("No slices provided for segmentation")

    pred_masks = []
    model.eval()

    with torch.no_grad():
        for i in range(0, len(slices), batch_size):
            batch = torch.cat(slices[i:i+batch_size], dim=0).to(device)
            preds = model(batch)  # [B,C,H,W]
            pred_masks.extend(preds.cpu())

    return pred_masks

def segment_volume(
    slices: List[torch.Tensor],
    model: smp.Unet,
    device: str = "cpu",
    batch_size: int = 8,
    sample_slices: Optional[int] = None
) -> List[torch.Tensor]:
    """
    Segment a 3D volume slice by slice.

    Args:
        slices: List of image tensors [1,1,H,W]
        model: Segmentation model
        device: Device to run inference on
        batch_size: Batch size for processing
        sample_slices: Optional number of slices to sample (None for all)

    Returns:
        List of segmentation masks [classes,H,W]
    """
    if not slices:
        raise ValueError("No slices provided for segmentation")

    # Sample slices if requested
    if sample_slices and len(slices) > sample_slices:
        indices = np.linspace(0, len(slices) - 1, sample_slices, dtype=int)
        slices = [slices[i] for i in indices]

    return segment_image(slices, model, device, batch_size)