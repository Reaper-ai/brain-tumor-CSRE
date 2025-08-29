from io import BytesIO
import nib
import pydicom
from PIL import Image
import torch
from torchvision import transforms
from typing import List
import numpy as np

# Preprocessing pipeline for 2D slices
preprocess_2d = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

def img_to_tensors(uploaded_files: List[BytesIO]) -> List[torch.Tensor]:
    """
    Convert uploaded PNG/JPG files into a list of preprocessed tensors for 2D models.

    Args:
        uploaded_files (List[BytsIO]): List of PNG files from Streamlit uploader

    Returns:
        List[torch.Tensor]: List of tensors [1,1,H,W], one per uploaded file
    """
    tensors: List[torch.Tensor] = []
    for f in uploaded_files:
        img = Image.open(f).convert("L")
        tensors.append(preprocess_2d(img).unsqueeze(0))
    return tensors

def dicoms_to_tensors(uploaded_files: List[BytesIO]) -> List[torch.Tensor]:
    """
    Convert uploaded single-slice DICOM files into a list of preprocessed tensors for 2D models.

    Args:
        uploaded_files (List[BytsIO]): List of DICOM files from Streamlit uploader

    Returns:
        List[torch.Tensor]: List of tensors [1,1,H,W], one per uploaded file
    """
    tensors: List[torch.Tensor] = []
    for f in uploaded_files:
        ds = pydicom.dcmread(f)
        img = ds.pixel_array.astype(float)
        img = (img - img.min()) / (img.max() - img.min() + 1e-8) * 255.0
        img = Image.fromarray(img.astype("uint8")).convert("L")
        tensors.append(preprocess_2d(img).unsqueeze(0))
    return tensors


# NIfTI → list of 2D tensors (all slices)
def nii_to_tensors(uploaded_file: BytesIO) -> List[torch.Tensor]:
    """
    Convert an NIfTI file into a list of preprocessed 2D tensors.
    Suitable for 2D slice-based models (classifier or segmenter).

    Args:
        uploaded_file (List[BytsIO]): NIfTI file uploaded via Streamlit

    Returns:
        List[torch.Tensor]: List of tensors, each [1, 1, H, W]
    """
    # Read uploaded file into a temporary buffer
    file_bytes = uploaded_file.read()
    file_like = BytesIO(file_bytes)

    nii_img = nib.load(file_like)
    img_np = nii_img.get_fdata()

    # Normalize to [0,255]
    img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8) * 255.0
    img_np = np.uint8(img_np)

    slice_tensors: List[torch.Tensor] = []
    for i in range(img_np.shape[2]):
        slice_2d = img_np[:, :, i]
        pil_img = Image.fromarray(slice_2d).convert("L")
        t = preprocess_2d(pil_img).unsqueeze(0)
        slice_tensors.append(t)

    return slice_tensors


def deduce(input):
    if input.name.endswith((".png", ".jpg", ".jpeg")):
        return 1
    elif input.name.endswith(".dcm"):
        return 2
    return None

# In preprocessor.py
def pre_processor(input, mode : str):  # Fixed spelling
    if mode == "2D":
        file_type = deduce(input)
        if file_type == 1:
            return img_to_tensors([input])
        elif file_type == 2:
            return dicoms_to_tensors([input])
    if mode == "3D":
        return nii_to_tensors(input)
    return None