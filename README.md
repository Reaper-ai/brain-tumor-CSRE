# Brain Tumor MRI Classifier & Multiclass Segmenter

A deep learning project for **brain tumor MRI analysis** that performs:

1. **Classification** – Detects whether a brain tumor is present in an MRI slice.
2. **Multiclass Segmentation** – Identifies and highlights tumor subregions.
3. **Uncertainty & Confidence Estimation** – Provides per-pixel entropy/confidence maps for reliable predictions.

Built with **PyTorch** and deployed with **Streamlit**.

---

## Features

* **Tumor Classification**: CNN-based classifier for MRI slices.
* **Multiclass Tumor Segmentation**: U-Net-style model segmenting multiple tumor classes (e.g., edema, enhancing core, necrosis).
* **Uncertainty & Confidence Maps**: Pixel-level entropy/confidence visualization.
* **Interactive Visualization**: Overlay masks and uncertainty maps on MRI scans.
* **Deployment**: Lightweight Streamlit web app for local demo.

---

## Project Structure

```
├── .streamlit             # Streamlit theme config
├── notebooks/             # Colab/Kaggle notebooks for training
├── scales/                # Images of scales used in output overlays
├── srcipts/               # Core code
│   ├── __init__.py
│   ├── classification.py
│   ├── segmentation.py
│   ├── loader.py
├── weights/               # Trained model weights
├── app.py                 # Streamlit app
├── requirements.txt
├── requirements-dev.txt
├── README.md
└── demo/                  # Example outputs
```

---

## Dataset
* **Classification**: [Brain Tumor MRI Dataset @Kaggle](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)
* **Segmentation**: [BraTS20 Dataset @Kaggle](https://www.kaggle.com/datasets/awsaf49/brats20-dataset-training-validation)
* **Modalities used**: **FLAIR** and **T1ce** only.
---

## Model Architectures

* **Classifier**: Mobile-Net-V2
* **Segmenter**: Efficient-Net-B0 encoder based U-Net.
* **Loss Functions**:

  * Classification → CrossEntropyLoss
  * Segmentation → Dice Loss + CrossEntropy

---

## Installation & Usage

### 1. Clone repo

```bash
git clone https://github.com/yourusername/brain-tumor-mri.git
cd brain-tumor-mri
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run Streamlit app

```bash
streamlit run app.py
```

---

## Results

### Classification

* Accuracy: **99%**
* Precision: **99%**
* F1 score: **99%**

### Multiclass Segmentation

* IOU Score (average): **74.76%**

---
## Sample outputs 

![classification output](demo/classification_result.png)
Output from classifer 

![segementation output both modalities](demo/segementation_results_both_modalities.png)
Output from segmentation mode, both flair and t1ce modalities were passed, the result has less uncertainty and the model is more confident


![segemwntation output single modality](demo/segementation_results_single_modality.png)
Output from segmentation mode, only flair modality was pass this time leading to large uncertainty and less confidence
---
## 📌 Future Work

* Extend to **all four modalities** (FLAIR, T1, T1ce, T2).
* Implement **3D tumor segmentation**.
* Improve runtime efficiency for clinical deployment.

---

## 🙌 Acknowledgements

* BraTS dataset organizers.
* PyTorch.
* Kaggle.
* Streamlit for deployment.

---