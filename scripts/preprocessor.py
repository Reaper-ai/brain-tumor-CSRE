from io import BytesIO
import nib
import pydicom
from PIL import Image
import torch
from torchvision import transforms
from typing import List, Optional, Union
import numpy as np


class PreprocessingError(Exception):
    """Custom exception for preprocessing errors"""
    pass


# Preprocessing pipeline for classification (3 channels)
preprocess_2d_classification = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.Grayscale(num_output_channels=3),  # Convert to 3 channels
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Preprocessing pipeline for segmentation (1 channel)
preprocess_2d_segmentation = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),  # Keeps 1 channel
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Default to classification for backward compatibility
preprocess_2d = preprocess_2d_classification


def img_to_tensors(uploaded_files: List[BytesIO], for_segmentation: bool = False) -> List[torch.Tensor]:
    """Convert images to tensors"""
    tensors: List[torch.Tensor] = []
    preprocess = preprocess_2d_segmentation if for_segmentation else preprocess_2d_classification
    
    for f in uploaded_files:
        try:
            img = Image.open(f).convert("L")
            tensor = preprocess(img).unsqueeze(0)
            tensors.append(tensor)
        except Exception as e:
            raise PreprocessingError(f"Error processing image file: {str(e)}")
    return tensors


def dicoms_to_tensors(uploaded_files: List[BytesIO], for_segmentation: bool = False) -> List[torch.Tensor]:
    """Convert DICOM files into tensors"""
    tensors: List[torch.Tensor] = []
    preprocess = preprocess_2d_segmentation if for_segmentation else preprocess_2d_classification
    
    for f in uploaded_files:
        try:
            ds = pydicom.dcmread(f)
            img = ds.pixel_array.astype(float)
            img = (img - img.min()) / (img.max() - img.min() + 1e-8) * 255.0
            img = Image.fromarray(img.astype("uint8")).convert("L")
            tensor = preprocess(img).unsqueeze(0)
            tensors.append(tensor)
        except Exception as e:
            raise PreprocessingError(f"Error processing DICOM file: {str(e)}")
    return tensors


def nii_to_tensors(uploaded_file: BytesIO, for_segmentation: bool = False) -> List[torch.Tensor]:
    """Convert NIfTI file into tensors"""
    preprocess = preprocess_2d_segmentation if for_segmentation else preprocess_2d_classification
    
    try:
        file_bytes = uploaded_file.read()
        file_like = BytesIO(file_bytes)

        nii_img = nib.load(file_like)
        img_np = nii_img.get_fdata()

        if len(img_np.shape) < 3:
            raise PreprocessingError("Invalid NIfTI file: expected 3D volume")

        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8) * 255.0
        img_np = np.uint8(img_np)

        slice_tensors: List[torch.Tensor] = []
        for i in range(img_np.shape[2]):
            slice_2d = img_np[:, :, i]
            pil_img = Image.fromarray(slice_2d).convert("L")
            tensor = preprocess(pil_img).unsqueeze(0)
            slice_tensors.append(tensor)

        return slice_tensors
    except Exception as e:
        raise PreprocessingError(f"Error processing NIfTI file: {str(e)}")


def get_tensors_for_task(input_files: Union[BytesIO, List[BytesIO]], mode: str, task: str) -> Optional[List[torch.Tensor]]:
    """Get tensors preprocessed for specific task"""
    for_segmentation = (task.lower() == "segmentation")
    
    try:
        if mode.upper() == "2D":
            if not isinstance(input_files, list):
                input_files = [input_files]

            file_type = deduce_file_type(input_files[0])

            if file_type == 1:  # PNG/JPG
                return img_to_tensors(input_files, for_segmentation)
            elif file_type == 2:  # DICOM
                return dicoms_to_tensors(input_files, for_segmentation)
            else:
                raise PreprocessingError("Unsupported file type for 2D mode")

        elif mode.upper() == "3D":
            if isinstance(input_files, list):
                if len(input_files) > 1:
                    raise PreprocessingError("3D mode expects a single NIfTI file")
                input_files = input_files[0]
            return nii_to_tensors(input_files, for_segmentation)

        else:
            raise PreprocessingError(f"Invalid mode: {mode}")

    except Exception as e:
        raise PreprocessingError(f"Preprocessing failed: {str(e)}")


def deduce_file_type(input: BytesIO) -> Optional[int]:
    if input.name.endswith((".png", ".jpg", ".jpeg")):
        return 1
    elif input.name.endswith(".dcm"):
        return 2
    return None


def pre_processor(input_files: Union[BytesIO, List[BytesIO]], mode: str) -> Optional[List[torch.Tensor]]:
    """Main preprocessing function - defaults to classification for backward compatibility"""
    return get_tensors_for_task(input_files, mode, "classification")