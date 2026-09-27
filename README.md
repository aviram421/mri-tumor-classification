# mri-tumor-classification

## Dataset
- The dataset for this project was taken from Kaggle and it contains train and test sets of MRI brain scans including 3 kinds of tumors: Meningioma, Glioma and Pituitary, and normal ones.
- Link: https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset/data

## Conclusions (for the meanwhile)
- After the first training with 3 epochs (ResNet) I saw that in the 3rd epoch, both the validation loss and error_rate raised a bit, which probably means that the model is overfitted, so I decided to change the number of epochs to 2.
