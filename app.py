import streamlit as st
import numpy as np
from PIL import Image

# ------------------- SETUP -------------------
st.markdown(
    """
    <style>
    /* ---- Button styling ---- */
    div[data-testid="stButton"] > button {
        background-color: #008080 !important;   /* teal */
        color: white !important;               /* text color */
        border: none !important;               /* remove default border */
        border-radius: 8px !important;         /* rounded corners */
        padding: 0.6em 1.2em !important;       /* nicer padding */
        font-weight: 600 !important;
        transition: background-color 0.2s ease-in-out;
    }
    
    div[data-testid="stButton"] > button:hover {
        background-color: #00AFAF !important;  /* lighter teal on hover */
        color: white !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)





# ------------------- DUMMY PIPELINES -------------------

def ClassificationPipeline(uploaded_file):
    # Just open the image to confirm it displays
    img = Image.open(uploaded_file)
    # Return dummy classification result
    return {"class": "Glioma", "confidence": 0.87, "preview": img}

def SegmentationPipeline2D(flair_file, t1ce_file):
    # Load the FLAIR image and return a dummy overlay (colored noise)
    flair_img = np.array(Image.open(flair_file).convert("RGB"))
    # Make dummy segmentation mask (random colors)
    random_mask = np.random.randint(0, 255, flair_img.shape, dtype=np.uint8)
    overlay_img = (0.6 * flair_img + 0.4 * random_mask).astype(np.uint8)
    return Image.fromarray(overlay_img)

def SegmentationPipeline3D(flair_bulk, t1ce_bulk):
    # Just return a dummy text output
    return "3D segmentation would be displayed or downloadable here (dummy output)."

# ------------------- FUNCTIONS -------------------

def overlay(input_img, segmask):
    # Dummy overlay: return the same image for now
    return input_img


def display3D(outputs):
    # Dummy 3D display: return a text output
    st.write(outputs)
# ------------------- HEADER -------------------
st.title("Brain Tumor CSRE")
st.subheader("Brain Tumor Classification and Segmentation")
tabs = ["Classification", "2D Segmentation", "3D Segmentation"]
tab1, tab2, tab3 = st.tabs(tabs)

# ------------------- TAB1 -------------------
with tab1:
    st.subheader("General Purpose Classifier")
    uploaded_file = st.file_uploader("Upload images for Classification", type=["png", "jpg", "jpeg"])
    if uploaded_file is not None:
        with st.spinner("Processing..."):
            outputs = ClassificationPipeline(uploaded_file)
            st.success(" Processing complete!")
            # now display the outputs
            col1, col2, col3 = st.columns([1, 2, 1])  # middle column is wider
            with col2:
                st.image(outputs["preview"])
                st.write(f"Predicted: {outputs['class']} conf. {outputs['confidence']}")

# ------------------- TAB2 -------------------
with tab2:

    st.subheader("Single Slice Segmenter")
    flair_slice = st.file_uploader("Upload image of FLAIR modality", type=["png", "jpg", "jpeg"])
    t1ce_slice = st.file_uploader("Upload image of T1 ce/gd modality", type=["png", "jpg", "jpeg"])
    tab2.warning("Missing or wrong modalities will produce uncertain results", icon="⚠️")
    process1 = st.button("Process", key="process2D")
    if process1:
        if not flair_slice or not t1ce_slice:
            st.error("Please upload both FLAIR and T1ce images")
        else:
            outputs = SegmentationPipeline2D(flair_slice, t1ce_slice)

            if len(outputs["images"]) == 2:
                st.image(overlay(outputs["images"]))
                st.write(" ")
            else:
                img1, img2 = overlay(outputs["images"])
                st.image(img1, caption="FLAIR")
                st.image(img2, caption="T1ce")
                st.write(" ")



# ------------------- TAB3 -------------------
with tab3:
    st.subheader("3D Volume Segmenter")
    flair_bulk = st.file_uploader("Upload NifTI file of FLAIR modality", type=["nii", "nii.gz"])
    t1ce_bulk = st.file_uploader("Upload NifTI file of T1 ce/gd modality", type=["nii", "nii.gz"])
    tab3.warning("Missing or wrong modalities will produce uncertain results", icon="⚠️")
    process2 = st.button("Process", key="process3D")
    if process2:
        if not flair_bulk or not t1ce_bulk:
            st.error("Please upload both NIfTI files")
        else:
            outputs = SegmentationPipeline3D(flair_bulk, t1ce_bulk)
            col1, col2 , col3 = st.columns([1, 2, 1])
            with col2:
                display3D(outputs)
                st.write(outputs)

# ------------------- FOOTER -------------------
st.markdown("---")
st.markdown("Made with ❤️")
