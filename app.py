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

# ------------------- HEADER -------------------
st.title("Brain Tumor CSRE")
st.subheader("Brain Tumor Classification and Segmentation")
tabs = ["Classification", "2D Segmentation"]
tab1, tab2 = st.tabs(tabs)

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
        if not (flair_slice or t1ce_slice):
            st.error("Please upload at least one of FLAIR or T1ce images")
        else:
            outputs = SegmentationPipeline2D(flair_slice, t1ce_slice, model2, device)

            col1, col2, col3 = st.columns([1,1,1])
            with col1:
                st.image(outputs["seg_overlay"], caption="Segmentation overlay")
            with col2:
                st.image(outputs["entropy_overlay"], caption="Entropy map")
            with col3:
                st.image(outputs["confidence_overlay"], caption="Confidence map")


# ------------------- FOOTER -------------------
st.markdown("---")
st.markdown("Made with ❤️")
