import numpy as np
import nibabel as nib
import io
import plotly.graph_objects as go
from typing import List
import torch
import PIL
import matplotlib.pyplot as plt
import streamlit as st


def visualize_3d_mask(pred_masks: List[torch.Tensor], class_idx: int = 1):
    """
    Create an interactive 3D volume visualization of a specific class from 2D predicted masks.

    Args:
        pred_masks (List[torch.Tensor]): List of predicted masks [classes, H, W]
        class_idx (int): Index of the class to visualize (default=1, e.g., tumor)
    """
    # Stack slices into a 3D numpy array
    volume = np.stack([m[class_idx].numpy() for m in pred_masks], axis=0)  # [D,H,W]

    # Optional: normalize for better visualization
    volume = (volume - volume.min()) / (volume.max() - volume.min() + 1e-8)

    # Create 3D volume plot
    fig = go.Figure(data=go.Volume(
        x=np.arange(volume.shape[2]),
        y=np.arange(volume.shape[1]),
        z=np.arange(volume.shape[0]),
        value=volume,
        opacity=0.1,          # controls transparency
        surface_count=15,     # number of isosurfaces
        colorscale="Hot"
    ))

    fig.update_layout(scene=dict(
        xaxis_title='Width',
        yaxis_title='Height',
        zaxis_title='Depth'
    ))

    st.plotly_chart(fig, use_container_width=True)


def overlay_2d_mask_multiclass(image: torch.Tensor, mask: torch.Tensor):
    """
    Overlay a 3-class predicted mask on a 2D image and display in Streamlit.

    Args:
        image (torch.Tensor): Original image [1,1,H,W]
        mask (torch.Tensor): Predicted mask [classes, H, W] (classes=3)
        class_names (List[str]): Names of the classes
    """
    class_names = ["Edema", "Enhancing", "Necrosis"]
    img_np = image.squeeze().numpy()

    # Calculate tumor areas for each class
    for i, name in enumerate(class_names):
        class_mask = mask[i].numpy()
        pixels = np.sum(class_mask > 0.5)
        percent = pixels / class_mask.size * 100
        st.write(f"{name} area: {pixels} pixels ({percent:.2f}%)")

    # Overlay with different colors
    colors = ["Reds", "Greens", "Blues"]
    fig, ax = plt.subplots()
    ax.imshow(img_np, cmap='gray')
    for i, cmap in enumerate(colors):
        ax.imshow(mask[i].numpy(), cmap=cmap, alpha=0.3)
    ax.axis('off')
    st.pyplot(fig)



def save_volume_nifti_multiclass(pred_masks: List[torch.Tensor], voxel_spacing=(1,1,1)):
    """
    Convert 3-class predicted masks to NIfTI, display middle slice overlay,
    print tumor volumes per class, and provide download in Streamlit.

    Args:
        pred_masks (List[torch.Tensor]): List of predicted masks [classes, H, W]
        class_names (List[str]): Names of the classes
        voxel_spacing (tuple): voxel spacing in mm
    """

    class_names = ["Edema", "Enhancing", "Necrosis"]
    num_classes = len(class_names)
    # Stack slices into [D,H,W,C] then move classes to last dimension
    volume = np.stack([m.numpy() for m in pred_masks], axis=0)  # [D,C,H,W]
    volume = np.transpose(volume, (0,2,3,1))  # [D,H,W,C]

    # Compute tumor volumes per class
    for i, name in enumerate(class_names):
        mask_class = volume[:,:,:,i]
        voxels = np.sum(mask_class > 0.5)
        volume_mm3 = voxels * np.prod(voxel_spacing)
        st.write(f"{name} volume: {voxels} voxels (~{volume_mm3:.2f} mm³)")

    # Save as NIfTI (C axis can be saved as multiple channels)
    nifti_img = nib.Nifti1Image(volume.astype(np.uint8), affine=np.eye(4))
    buf = io.BytesIO()
    nib.save(nifti_img, buf)
    buf.seek(0)

    st.download_button(
        label="Download multi-class NIfTI",
        data=buf,
        file_name="segmentation_multiclass.nii.gz",
        mime="application/gzip"
    )

    # Preview middle slice overlay
    mid_idx = len(pred_masks) // 2
    fig, ax = plt.subplots()
    ax.imshow(np.sum(volume[mid_idx,:,:,:], axis=-1), cmap="hot", alpha=0.5)
    ax.axis('off')
    st.pyplot(fig)

# In postprocessor.py
def post_processor(tensors: List[torch.Tensor], outputs: dict):
    """Process and display model outputs"""
    if not outputs:
        st.error("No results to display")
        return

    if "classification" in outputs:
        st.subheader("Classification Results")
        if isinstance(outputs["classification"], dict):
            st.write(f"Predicted: {outputs['classification']['label']}")
            st.write(f"Confidence: {outputs['classification']['confidence']:.2f}%")
        else:
            for i, label in enumerate(outputs["classification"]):
                st.write(f"Slice {i+1}: {label}")

    if "segmentation" in outputs:
        st.subheader("Segmentation Results")
        masks = outputs["segmentation"]
        
        if len(masks) > 1:
            selected_slice = st.slider(
                "Select slice", 
                0, 
                len(masks)-1, 
                len(masks)//2
            )
            overlay_2d_mask_multiclass(
                tensors[selected_slice], 
                masks[selected_slice]
            )
            
            if st.button("Show 3D Visualization"):
                visualize_3d_mask(masks)
                
            if st.button("Save as NIfTI"):
                save_volume_nifti_multiclass(masks)
        else:
            overlay_2d_mask_multiclass(tensors[0], masks[0])