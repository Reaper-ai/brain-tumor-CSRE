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


# Preprocessing pipeline for 2D slices - now converts to 3 channels
preprocess_2d = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.Grayscale(num_output_channels=3),  # Convert to 3 channels
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize all channels
])


# Modify the preprocessing functions to move tensors to GPU
def img_to_tensors(uploaded_files: List[BytesIO]) -> List[torch.Tensor]:
    tensors: List[torch.Tensor] = []
    for f in uploaded_files:
        try:
            img = Image.open(f).convert("L")
            tensor = preprocess_2d(img).unsqueeze(0).cuda()  # Add .cuda() here
            tensors.append(tensor)
        except Exception as e:
            raise PreprocessingError(f"Error processing image file: {str(e)}")
    return tensors


def dicoms_to_tensors(uploaded_files: List[BytesIO]) -> List[torch.Tensor]:
    """Convert DICOM files into tensors"""
    tensors: List[torch.Tensor] = []
    for f in uploaded_files:
        try:
            ds = pydicom.dcmread(f)
            img = ds.pixel_array.astype(float)
            img = (img - img.min()) / (img.max() - img.min() + 1e-8) * 255.0
            img = Image.fromarray(img.astype("uint8")).convert("L")
            tensor = preprocess_2d(img).unsqueeze(0)  # Shape will be [1, 3, 256, 256]
            tensors.append(tensor)
        except Exception as e:
            raise PreprocessingError(f"Error processing DICOM file: {str(e)}")
    return tensors


def nii_to_tensors(uploaded_file: BytesIO) -> List[torch.Tensor]:
    """Convert NIfTI file into tensors"""
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
            tensor = preprocess_2d(pil_img).unsqueeze(0)  # Shape will be [1, 3, 256, 256]
            slice_tensors.append(tensor)

        return slice_tensors
    except Exception as e:
        raise PreprocessingError(f"Error processing NIfTI file: {str(e)}")

def deduce_file_type(input: BytesIO) -> Optional[int]:
    if input.name.endswith((".png", ".jpg", ".jpeg")):
        return 1
    elif input.name.endswith(".dcm"):
        return 2
    return None

def pre_processor(input_files: Union[BytesIO, List[BytesIO]], mode: str) -> Optional[List[torch.Tensor]]:
    """Main preprocessing function"""
    try:
        if mode.upper() == "2D":
            # Ensure input_files is a list
            if not isinstance(input_files, list):
                input_files = [input_files]

            # Get file type from first file
            file_type = deduce_file_type(input_files[0])

            # Process based on file type
            if file_type == 1:  # PNG/JPG
                return img_to_tensors(input_files)
            elif file_type == 2:  # DICOM
                return dicoms_to_tensors(input_files)
            else:
                raise PreprocessingError("Unsupported file type for 2D mode")

        elif mode.upper() == "3D":
            # 3D mode expects a single file
            if isinstance(input_files, list):
                if len(input_files) > 1:
                    raise PreprocessingError("3D mode expects a single NIfTI file")
                input_files = input_files[0]  # Take the first file
            return nii_to_tensors(input_files)

        else:
            raise PreprocessingError(f"Invalid mode: {mode}")

    except Exception as e:
        raise PreprocessingError(f"Preprocessing failed: {str(e)}")