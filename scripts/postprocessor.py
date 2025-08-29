import numpy as np
import nibabel as nib
import io
import plotly.graph_objects as go
from typing import List
import torch
import PIL
import matplotlib.pyplot as plt
import streamlit as st


def visualize_3d_mask(pred_masks: List[torch.Tensor], class_idx: int = 0):
    """
    Create an interactive 3D volume visualization of a specific class from 2D predicted masks.

    Args:
        pred_masks (List[torch.Tensor]): List of predicted masks [3, H, W]
        class_idx (int): Index of the class to visualize (0, 1, or 2)
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
        image (torch.Tensor): Original image [1,1,H,W] (1 channel grayscale)
        mask (torch.Tensor): Predicted mask [3, H, W] (3 classes)
    """
    class_names = ["Edema", "Enhancing", "Necrosis"]
    colors = ["red", "green", "blue"]
    
    # Get the grayscale image - squeeze all dimensions to get [H, W]
    img_np = image.squeeze().numpy()

    # Calculate tumor areas for each class
    for i, name in enumerate(class_names):
        class_mask = mask[i].numpy()
        pixels = np.sum(class_mask > 0.5)
        percent = pixels / class_mask.size * 100
        st.write(f"{name} area: {pixels} pixels ({percent:.2f}%)")

    # Create the overlay visualization
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Show the original grayscale image
    ax.imshow(img_np, cmap='gray')
    
    # Overlay each class with different colors
    for i, (name, color) in enumerate(zip(class_names, colors)):
        class_mask = mask[i].numpy()
        # Create a colored overlay where mask > 0.5
        overlay = np.zeros((*class_mask.shape, 4))  # RGBA
        
        # Set color for the overlay
        if color == "red":
            overlay[:, :, 0] = class_mask  # Red channel
        elif color == "green":
            overlay[:, :, 1] = class_mask  # Green channel
        elif color == "blue":
            overlay[:, :, 2] = class_mask  # Blue channel
            
        overlay[:, :, 3] = class_mask * 0.6  # Alpha channel
        
        # Only show where mask is significant
        overlay[class_mask < 0.1] = 0
        
        ax.imshow(overlay)
    
    ax.axis('off')
    ax.set_title("Segmentation Overlay")
    st.pyplot(fig)
    plt.close()


def save_volume_nifti_multiclass(pred_masks: List[torch.Tensor], voxel_spacing=(1,1,1)):
    """
    Convert 3-class predicted masks to NIfTI, display middle slice overlay,
    print tumor volumes per class, and provide download in Streamlit.

    Args:
        pred_masks (List[torch.Tensor]): List of predicted masks [3, H, W]
        voxel_spacing (tuple): voxel spacing in mm
    """
    class_names = ["Edema", "Enhancing", "Necrosis"]
    
    # Stack slices into [D,C,H,W] then move classes to last dimension
    volume = np.stack([m.numpy() for m in pred_masks], axis=0)  # [D,3,H,W]
    volume = np.transpose(volume, (0,2,3,1))  # [D,H,W,3]

    # Compute tumor volumes per class
    for i, name in enumerate(class_names):
        mask_class = volume[:,:,:,i]
        voxels = np.sum(mask_class > 0.5)
        volume_mm3 = voxels * np.prod(voxel_spacing)
        st.write(f"{name} volume: {voxels} voxels (~{volume_mm3:.2f} mm³)")

    # Save as NIfTI
    nifti_img = nib.Nifti1Image(volume.astype(np.float32), affine=np.eye(4))
    buf = io.BytesIO()
    nib.save(nifti_img, buf)
    buf.seek(0)

    st.download_button(
        label="Download multi-class NIfTI",
        data=buf,
        file_name="segmentation_multiclass.nii.gz",
        mime="application/gzip"
    )

    # Preview middle slice overlay - combine all classes
    mid_idx = len(pred_masks) // 2
    fig, ax = plt.subplots()
    combined_mask = np.sum(volume[mid_idx,:,:,:], axis=-1)
    ax.imshow(combined_mask, cmap="hot", alpha=0.8)
    ax.axis('off')
    ax.set_title("Middle Slice Preview")
    st.pyplot(fig)
    plt.close()


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
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Show 3D Visualization"):
                    class_options = ["Edema", "Enhancing", "Necrosis"]
                    selected_class = st.selectbox("Select class:", class_options)
                    class_idx = class_options.index(selected_class)
                    visualize_3d_mask(masks, class_idx)
            
            with col2:
                if st.button("Save as NIfTI"):
                    save_volume_nifti_multiclass(masks)
        else:
            overlay_2d_mask_multiclass(tensors[0], masks[0])