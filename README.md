# MRI Brain Tumor Classification 🧠

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![FastAI](https://img.shields.io/badge/FastAI-2.7-red.svg)](https://docs.fast.ai/)
[![Dataset](https://img.shields.io/badge/Dataset-Kaggle-blue.svg)](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset/data)

A deep learning project aimed at accurately classifying brain MRI scans into four different categories using Convolutional Neural Networks (CNNs). This project uses the **FastAI** library and a pretrained **ResNet18** architecture, focusing on resizing, data augmentations, duplicate removal, and optimal learning rate selection to achieve high accuracy in a medical imaging context.

## 🎯 Project Overview
Classifying brain tumors accurately from MRI scans is a critical step in medical diagnosis. This project trains a model to classify scans into one of four classes (3 kinds of tumors or no tumor):
1. **Glioma**
2. **Meningioma**
3. **Pituitary**
4. **No Tumor**

Here's an example for a small batch of scans, as shown in the model notebook:
<img width="716" height="735" alt="image" src="https://github.com/user-attachments/assets/a1e21bf2-74a1-4ae8-b8ec-b98421d497ee" />


**Key Technical Highlights:**
* **Medical Data Integrity (Resizing):** Explored different image resizing techniques. Chose **Zero-Padding** over squishing (which distorts proportions) or random cropping (which can crop out the tumor itself) to preserve the true aspect ratio.

* **Data Cleaning & Leakage Prevention:** Actively identified and removed duplicate scans across the training and testing sets using FastAI's `ImageClassifierCleaner` to ensure the model generalizes properly rather than memorizing leaked data. (Not that many duplicates were removed, but the accuracy was still improved)

* **Learning Rate Optimization:** Utilized FastAI's learning rate finder (`lr_find`) and `fit_one_cycle` methods to optimize model weights, reducing the final error rate to ~3.69%.

## 📊 Dataset
The model is trained on the [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset/data) via Kaggle.
* **Format:** MRI Jpg images.
* **Warning / Note on Data Leakage:** The author of this dataset doesn't explicitly mention if there might be scans from the same patient in both the train and test sets, so patient-level data leakage is possible.

## 🚀 Results & Performance
After experimenting with multiple training strategies, learning rates, and data augmentations (`aug_transforms`), the final fine-tuned model achieved excellent generalization on the validation set.

* **Final Error Rate:** ~3.69% (Accuracy of **~96.3%**)
* **False Negatives:** The model demonstrates a very low False Negative rate for the "No Tumor" class, which is crucial for medical applications. 
* **Challenges:** The model occasionally struggles with differentiating Glioma from Meningioma and other tumors, sometimes showing high confidence in wrong predictions. This might be due to the 2D nature of the dataset versus 3D spatial views (DICOM/NIfTI).



Here's the confusion matrix of the last fine-tuned model:

<img width="467" height="489" alt="image" src="https://github.com/user-attachments/assets/062b615c-36eb-4402-affa-46e2ae76fd36" />


## 📂 Repository Structure
```text
├── mri-tumor-classification - EDA.ipynb   # Exploratory Data Analysis and class balancing
├── preprocessing-mri-dataset.ipynb        # Dataset downloading and image resizing experiments
├── tumor-classification-model.ipynb       # Model training, LR experiments, and evaluation
├── model_funcs.py                         # Helper functions for training and transformation
├── app.ipynb                              # Notebook for application logic and testing
├── app.py                                 # Deployment application script
├── model.pkl                              # Exported trained model weights
├── requirements.txt                       # Project dependencies
└── README.md                              # Project documentation

## How to Run
Make sure you have Python 3.13 installed. I created a requirements.txt file which you can run in your preferred environment to install the exact dependcies I used with the following bash command:
**pip install -r requirements.txt**

Execution

1. Explore the Dataset: Run the mri-tumor-classification - EDA.ipynb notebook to see initial image properties and dataset structure, as well as class balance.

2. Train the model: Run the tumor-classification-model.ipynb to run the whole training pipeline, as well as different experiments.

It is highly recommended to run the tumor-classification-model.ipynb in a GPU-based environment such as Kaggle/Colab.

Note: I ran the model notebook using Colab's GPU T4 accelerator. One may run it on another accelerator but different results are expected.
