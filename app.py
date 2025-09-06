import streamlit as st
from scripts.classification import ClassificationPipeline
from scripts.segmentation import SegmentationPipeline2D, SegmentationPipeline3D

st.title("Brain Tumor CSRE")
st.subheader("Brain Tumor Classification and Segmentation")
tabs = ["Classification", "2D Segmentation", "3D Segmentation"]
tab1, tab2, tab3 = st.tabs(tabs)

tab1.subheader("General Purpose Classifier")
uploaded_file = tab1.file_uploader("Upload images for Classification", type=["png", "jpg", "jpeg"])
if uploaded_file is not None:
    outputs = ClassificationPipeline(uploaded_file)


tab2.subheader("Single Slice Segmenter")
flair_slice = tab2.file_uploader("Upload image of FLAIR modality", type=["png", "jpg", "jpeg"])
t1ce_slice = tab2.file_uploader("Upload images of T1 ce", type=["png", "jpg", "jpeg"])
process1 = tab2.button("Process", key="process2D")
if process1:
    SegmentationPipeline2D(flair_slice, t1ce_slice)


tab3.subheader("3D Volume Segmenter")
flair_bulk = tab3.file_uploader("Upload image of FLAIR modality", type=["nii", "nii.gz"])
t1ce_bulk = tab3.file_uploader("Upload images of T1 ce", type=["nii", "nii.gz"])
process2 = tab3.button("Process", key="process3D")
if process2:
    SegmentationPipeline3D(flair_bulk, t1ce_bulk)

st.markdown("---")
st.markdown("Made with ❤️ ")

