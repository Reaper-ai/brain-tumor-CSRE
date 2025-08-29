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
    Only for 3D volume analysis.

    Args:
        pred_masks (List[torch.Tensor]): List of predicted masks [4, H, W]
        class_idx (int): Index of the class to visualize (0-3)
    """
    if class_idx >= len(pred_masks[0]):
        st.error(f"Invalid class index {class_idx}. Available classes: 0-{len(pred_masks[0])-1}")
        return
        
    # Stack slices into a 3D numpy array
    volume = np.stack([m[class_idx].numpy() for m in pred_masks], axis=0)  # [D,H,W]

    # Apply softmax-like normalization for better visualization
    volume = np.exp(volume) / (1 + np.exp(volume))  # sigmoid
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


def save_volume_nifti_multiclass(pred_masks: List[torch.Tensor], voxel_spacing=(1,1,1)):
    """
    Convert 4-class predicted masks to NIfTI and provide download.
    Only for 3D volume analysis.

    Args:
        pred_masks (List[torch.Tensor]): List of predicted masks [4, H, W]
        voxel_spacing (tuple): voxel spacing in mm
    """
    class_names = ["Background", "Tumor Region 1", "Tumor Region 2", "Tumor Region 3"]
    
    # Convert logits to class predictions
    pred_classes_volume = []
    for mask in pred_masks:
        mask_probs = torch.softmax(mask, dim=0)
        pred_class = torch.argmax(mask_probs, dim=0).numpy()
        pred_classes_volume.append(pred_class)
    
    # Stack into volume [D, H, W]
    volume = np.stack(pred_classes_volume, axis=0)

    # Compute tumor volumes per class (skip background)
    for i in range(1, len(class_names)):
        voxels = np.sum(volume == i)
        volume_mm3 = voxels * np.prod(voxel_spacing)
        st.write(f"{class_names[i]} volume: {voxels} voxels (~{volume_mm3:.2f} mm³)")

    # Save as NIfTI
    nifti_img = nib.Nifti1Image(volume.astype(np.uint8), affine=np.eye(4))
    buf = io.BytesIO()
    nib.save(nifti_img, buf)
    buf.seek(0)

    st.download_button(
        label="Download segmentation NIfTI",
        data=buf,
        file_name="segmentation_classes.nii.gz",
        mime="application/gzip"
    )


def display_2d_segmentation(image: torch.Tensor, mask: torch.Tensor):
    """
    Display 2D segmentation results with original image and overlayed segmentation.
    """
    # Real class names
    class_names = ["Background", "Edema", "Non-Enhancing Tumor", "Enhancing Tumor"]
    class_colors = [
        (0, 0, 0),  # Background
        (1, 0, 0),  # Edema - red
        (0, 1, 0),  # Non-Enhancing Tumor - green
        (0, 0, 1)  # Enhancing Tumor - blue
    ]

    # Convert inputs
    img_np = image.squeeze().numpy()
    mask_probs = torch.softmax(mask, dim=0)
    pred_classes = torch.argmax(mask_probs, dim=0).numpy()

    # Areas
    total_pixels = img_np.size
    for i in range(1, len(class_names)):
        class_pixels = np.sum(pred_classes == i)
        percent = (class_pixels / total_pixels) * 100
        st.write(f"**{class_names[i]}** area: {class_pixels} pixels ({percent:.2f}%)")

    # Build overlay mask
    overlay = np.zeros((*pred_classes.shape, 3), dtype=np.float32)
    for i, color in enumerate(class_colors):
        overlay[pred_classes == i] = color

    # Normalize grayscale & expand to RGB
    img_norm = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)
    img_rgb = np.stack([img_norm] * 3, axis=-1)

    # Blend (increase overlay visibility → 0.5 overlay, 0.5 image)
    blended = (0.5 * img_rgb + 0.5 * overlay).clip(0, 1)

    # Create smaller figure (not full container)
    fig, axes = plt.subplots(1, 2, figsize=(6, 3), dpi=120)  # smaller size

    axes[0].imshow(img_np, cmap="gray")
    axes[0].set_title("Original MRI", fontsize=11)
    axes[0].axis("off")

    axes[1].imshow(blended)
    axes[1].set_title("Segmentation Overlay", fontsize=11)
    axes[1].axis("off")

    # Add legend (skip background)
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, facecolor=color, label=class_names[i])
        for i, color in enumerate(class_colors) if i > 0
    ]
    axes[1].legend(handles=legend_elements, loc="lower right", fontsize=8)

    plt.tight_layout()

    # Control Streamlit rendering size
    st.pyplot(fig, clear_figure=True)
    plt.close(fig)


def post_processor_2d(tensors: List[torch.Tensor], outputs: dict):
    """Process and display 2D model outputs - no 3D features"""
    if not outputs:
        st.error("No results to display")
        return

    if "classification" in outputs:
        st.subheader("Classification Results")
        for i, label in enumerate(outputs["classification"]):
            st.write(f"**Slice {i+1}:** {label}")

    if "segmentation" in outputs:
        st.subheader("Segmentation Results")
        masks = outputs["segmentation"]
        
        if len(masks) > 1:
            selected_slice = st.slider(
                "Select slice for visualization", 
                0, 
                len(masks)-1, 
                len(masks)//2
            )
            
            display_2d_segmentation(
                tensors[selected_slice], 
                masks[selected_slice]
            )
        else:
            display_2d_segmentation(tensors[0], masks[0])


def post_processor_3d(tensors: List[torch.Tensor], outputs: dict):
    """Process and display 3D model outputs - includes 3D features"""
    if not outputs:
        st.error("No results to display")
        return

    if "classification" in outputs:
        st.subheader("Classification Results")
        if isinstance(outputs["classification"], dict):
            st.write(f"**Predicted:** {outputs['classification']['label']}")
            st.write(f"**Confidence:** {outputs['classification']['confidence']:.2f}%")

    if "segmentation" in outputs:
        st.subheader("Segmentation Results")
        masks = outputs["segmentation"]
        
        # Show sample slice
        mid_slice = len(masks) // 2
        selected_slice = st.slider(
            "Select slice for visualization", 
            0, 
            len(masks)-1, 
            mid_slice
        )
        
        display_2d_segmentation(
            tensors[selected_slice], 
            masks[selected_slice]
        )
        
        # 3D specific features
        st.subheader("3D Volume Analysis")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Show 3D Visualization"):
                class_options = ["Background", "Tumor Region 1", "Tumor Region 2", "Tumor Region 3"]
                selected_class = st.selectbox("Select class:", class_options[1:])  # Skip background
                class_idx = class_options.index(selected_class)
                visualize_3d_mask(masks, class_idx)
        
        with col2:
            if st.button("Save as NIfTI"):
                save_volume_nifti_multiclass(masks)


# Main post processor function - delegates to appropriate handler
def post_processor(tensors: List[torch.Tensor], outputs: dict, mode: str = "2D"):
    """
    Main post processor that delegates to appropriate handler based on mode
    
    Args:
        tensors: Display tensors
        outputs: Model outputs
        mode: "2D" or "3D" 
    """
    if mode.upper() == "3D":
        post_processor_3d(tensors, outputs)
    else:
        post_processor_2d(tensors, outputs)