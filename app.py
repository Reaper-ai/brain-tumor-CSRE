import streamlit as st
from scripts import preprocessor as pp, pipelines as pl, postprocessor as ptp

# ---------------------------
# App Config
# ---------------------------
st.set_page_config(
    page_title="Brain Tumor MRI Analyzer",
    page_icon="",
    layout="wide"
)

# Load initial models (t1 and classifier)
pl.load_models()

MAX_FILES = 8
st.title("Brain Tumor MRI Analyzer")
st.write("Upload MRI slices and get **classification** + **segmentation** results.")

# ----------------及ひ-----------
# Analysis Type and Modality Selection
# ---------------------------
col1, col2 = st.columns(2)
with col1:
    analysis_type = st.radio(
        "Select Analysis Type:", 
        ["Classification", "Segmentation", "Both"], 
        horizontal=True
    )
with col2:
    modality = st.selectbox(
        "Select MRI Modality:",
        ["T1", "T2", "FLAIR", "T1CE"],
        help="Select the MRI sequence type"
    )

# ---------------------------
# Tabs for workflow
# ---------------------------
tab1, tab2 = st.tabs(["Single Slice Analysis", "3D Volume Analysis"])

# ---------------------------
# SINGLE SLICE TAB
# ---------------------------
with tab1:
    st.header("Single Slice Analysis")
    uploaded_file = st.file_uploader(
        "Upload an MRI slice:\n upload only one type of file", 
        accept_multiple_files=True, 
        type=["png", "jpg", "jpeg", "dcm"], 
        key="single_slice_uploader"
    )

    if uploaded_file:
        if len(uploaded_file) > MAX_FILES:
            st.warning(f"Maximum number of files reached. Only the first {MAX_FILES} will be processed.")
            files_to_process = uploaded_file[:MAX_FILES]
        else:
            files_to_process = uploaded_file

        try:
            with st.spinner("Processing files..."):
                # Start with classification preprocessing (default)
                tensors = pp.pre_processor(files_to_process, "2D")
                if tensors:
                    outputs = pl.twoDpipeline(files_to_process, tensors, analysis_type, modality)
                    # Use display_tensors from pipeline results
                    display_tensors = outputs.get("display_tensors", tensors)
                    ptp.post_processor(display_tensors, outputs)
                else:
                    st.error("Failed to process files")
        except Exception as e:
            st.error(f"Error during processing: {str(e)}")

# ---------------------------
# 3D VOLUME ANALYSIS TAB
# ---------------------------
with tab2:
    st.header("3D Volume Analysis")
    uploaded_files = st.file_uploader(
        "Upload MRI volume", 
        type=["nii", "nii.gz"], 
        key="volume_uploader"
    )

    if uploaded_files:
        with st.spinner("Processing volume..."):
            # Start with classification preprocessing (default)
            tensors = pp.pre_processor(uploaded_files, "3D")
            if tensors:
                outputs = pl.threeDpipeline(uploaded_files, tensors, analysis_type, modality)
                # Use display_tensors from pipeline results
                display_tensors = outputs.get("display_tensors", tensors)
                ptp.post_processor(display_tensors, outputs)