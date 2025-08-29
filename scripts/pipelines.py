from scripts import classifier, segmenter
import torch
from typing import Optional, Dict, List, Union, Any
from torch import nn
import segmentation_models_pytorch as smp
from .config import validate_model_paths

# Define model types
ModelType = Union[nn.Module, smp.Unet]
ModelsDict = Dict[str, Optional[ModelType]]

# Initialize models dictionary with type annotation
models: ModelsDict = {
    "classifier": None,
    "t1": None,
    "t2": None,
    "flair": None,
    "t1ce": None
}

def get_device() -> str:
    """Determine the available device"""
    return "cuda" if torch.cuda.is_available() else "cpu"

def load_models(device: Optional[str] = None) -> None:
    """Load all models with lazy loading for segmentation models"""
    device = device or get_device()
    try:
        # Validate all model paths before loading
        validate_model_paths()
        
        # Load models
        models["classifier"] = classifier.load_model().to(device)
        models["t1"] = segmenter.load_seg_model("t1", device)
    except Exception as e:
        raise RuntimeError(f"Failed to load initial models: {str(e)}")

def load_modality_model(modality: str, device: str = "cpu") -> smp.Unet:
    """
    Lazy loading of segmentation models
    
    Args:
        modality: The modality type (t1, t2, flair, t1ce)
        device: The device to load the model on
    
    Returns:
        Loaded segmentation model
    """
    if models.get(modality.lower()) is None:
        models[modality.lower()] = segmenter.load_seg_model(modality.lower(), device)
    return models[modality.lower()]  # type: ignore

def _2Dpipeline(
    tensors: List[torch.Tensor], 
    analysis_type: str, 
    modality: str = "t1"
) -> Dict[str, Any]:
    """
    2D pipeline with modality support
    
    Args:
        tensors: List of input tensors
        analysis_type: Type of analysis to perform
        modality: MRI modality type
    
    Returns:
        Dictionary containing results
    """
    analysis_type = analysis_type.lower()
    results: Dict[str, Any] = {}
    
    if analysis_type in ["classification", "both"]:
        if models["classifier"] is None:
            raise RuntimeError("Classifier model not loaded")
        results["classification"] = classifier.classify_slices(
            tensors, 
            models["classifier"]
        )
    
    if analysis_type in ["segmentation", "both"]:
        seg_model = load_modality_model(modality)
        results["segmentation"] = segmenter.segment_image(tensors, seg_model)
    
    return results

def _3Dpipeline(
    tensors: List[torch.Tensor], 
    analysis_type: str, 
    modality: str = "t1"
) -> Dict[str, Any]:
    """
    3D pipeline implementation
    
    Args:
        tensors: List of input tensors
        analysis_type: Type of analysis to perform
        modality: MRI modality type
    
    Returns:
        Dictionary containing results
    """
    analysis_type = analysis_type.lower()
    results: Dict[str, Any] = {}
    
    if analysis_type in ["classification", "both"]:
        if models["classifier"] is None:
            raise RuntimeError("Classifier model not loaded")
        label, confidence = classifier.classify_volume(
            tensors, 
            models["classifier"],
            sample_slices=20
        )
        results["classification"] = {
            "label": label,
            "confidence": confidence
        }
    
    if analysis_type in ["segmentation", "both"]:
        seg_model = load_modality_model(modality)
        results["segmentation"] = segmenter.segment_volume(
            tensors,
            seg_model,
            sample_slices=50
        )
    
    return results