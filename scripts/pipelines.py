from scripts import classifier, segmenter, preprocessor as pp
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

def twoDpipeline(
    input_files,
    tensors: List[torch.Tensor], 
    analysis_type: str, 
    modality: str = "t1"
) -> Dict[str, Any]:
    """
    2D pipeline with modality support
    
    Args:
        input_files: Original input files for reprocessing if needed
        tensors: List of input tensors (classification tensors)
        analysis_type: Type of analysis to perform
        modality: MRI modality type
    
    Returns:
        Dictionary containing results
    """
    analysis_type = analysis_type.lower()
    results: Dict[str, Any] = {}
    device = get_device()
    
    if analysis_type in ["classification", "both"]:
        if models["classifier"] is None:
            raise RuntimeError("Classifier model not loaded")
        
        # Use existing classification tensors (3 channels)
        class_tensors = [tensor.to(device) for tensor in tensors]
        
        results["classification"] = classifier.classify_slices(
            class_tensors, 
            models["classifier"],
            device=device
        )
    
    if analysis_type in ["segmentation", "both"]:
        seg_model = load_modality_model(modality, device)
        
        # Get new tensors for segmentation (1 channel)
        seg_tensors = pp.get_tensors_for_task(input_files, "2D", "segmentation")
        if seg_tensors:
            seg_tensors = [tensor.to(device) for tensor in seg_tensors]
            results["segmentation"] = segmenter.segment_image(seg_tensors, seg_model, device)
        else:
            raise RuntimeError("Failed to get segmentation tensors")
    
    return results

def threeDpipeline(
    input_files,
    tensors: List[torch.Tensor], 
    analysis_type: str, 
    modality: str = "t1"
) -> Dict[str, Any]:
    """
    3D pipeline implementation
    
    Args:
        input_files: Original input files for reprocessing if needed
        tensors: List of input tensors (classification tensors)
        analysis_type: Type of analysis to perform
        modality: MRI modality type
    
    Returns:
        Dictionary containing results
    """
    analysis_type = analysis_type.lower()
    results: Dict[str, Any] = {}
    device = get_device()
    
    if analysis_type in ["classification", "both"]:
        if models["classifier"] is None:
            raise RuntimeError("Classifier model not loaded")
        
        # Use existing classification tensors (3 channels)
        class_tensors = [tensor.to(device) for tensor in tensors]
        
        label, confidence = classifier.classify_volume(
            class_tensors, 
            models["classifier"],
            device=device,
            sample_slices=20
        )
        results["classification"] = {
            "label": label,
            "confidence": confidence
        }
    
    if analysis_type in ["segmentation", "both"]:
        seg_model = load_modality_model(modality, device)
        
        # Get new tensors for segmentation (1 channel)
        seg_tensors = pp.get_tensors_for_task(input_files, "3D", "segmentation")
        if seg_tensors:
            seg_tensors = [tensor.to(device) for tensor in seg_tensors]
            results["segmentation"] = segmenter.segment_volume(
                seg_tensors,
                seg_model,
                device=device,
                sample_slices=50
            )
        else:
            raise RuntimeError("Failed to get segmentation tensors")
    
    return results

# Keep backward compatibility with old function names
def _2Dpipeline(*args, **kwargs):
    return twoDpipeline(*args, **kwargs)

def _3Dpipeline(*args, **kwargs):
    return threeDpipeline(*args, **kwargs)