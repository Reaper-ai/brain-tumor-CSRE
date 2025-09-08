import streamlit as st
from PIL import Image
from scripts import loader
from scripts.classification import ClassificationPipeline
from scripts.segmentation import SegmentationPipeline2D
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
# model1 -> classification, model2 -> segmentation
device , model1 , model2 = loader.load_models()

# ------------------- DUMMY PIPELINES -------------------


def SegmentationPipeline3D(flair_bulk, t1ce_bulk):
    # Just return a dummy text output
    return "3D segmentation would be displayed or downloadable here (dummy output)."

# ------------------- FUNCTIONS -------------------

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
            outputs = ClassificationPipeline(uploaded_file, model1, device )
            st.success(" Processing complete!")
            # now display the outputs
            col1, col2, col3 = st.columns([1, 2, 1])  # middle column is wider
            with col2:
                img = Image.open(uploaded_file)
                st.image(img)
                st.write(f"Predicted: {outputs['class']} conf. {outputs['confidence']:.3f}")

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
                col1, col2, col3 = st.columns([1,1,1])
                with col1:
                    st.image(outputs["images"][0])
                with col2:
                    st.image(outputs["images"][1])
                    st.write(outputs["uncertainty"])
                with col3:
                    st.image(outputs["images"][2])
            else :
                col1, col2, col3 , col4 = st.columns([1,2,2,1])

                with col2:
                    st.image(outputs["images"][0])
                with col3:
                    st.image(outputs["images"][1])

                st.write(outputs["uncetainity"])







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
