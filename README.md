# mri-tumor-classification

## Dataset
- The dataset for this project was taken from Kaggle and it contains train and test sets of MRI brain scans including 3 kinds of tumors: Meningioma, Glioma and Pituitary, and normal ones.
- Link: https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset/data

## Pictures and Graphs
- First training:
<img width="355" height="213" alt="image" src="https://github.com/user-attachments/assets/6ecb6a7e-9ce7-461a-8370-e4cfa9065e5e" />

With seed=42: in the 4th epoch, both the validation loss and error_rate raised a bit, which probably means that the model is overfitted, so I decided to change the number of epochs to 3.


## Notes
- I ran the model notebook using Kaggle's GPU accelerator. One may run it on another accelerator.
