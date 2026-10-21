from pathlib import Path
from fastcore.all import *
from fastai.vision.all import *
from fastai.vision.widgets import *
import shutil

def train_model(num_epochs: int) -> Learner:
    """
    Fine tunes resnet18 model with dls object and error rate metric.
    
    Arguments:
        num_epochs(int): the desired number of epochs for the fine tuning.

    Returns:
        learn(Learner): A vision learner trained on the MRI scans and ResNet18 model.
    """
    learn = vision_learner(dls, resnet18, metrics=error_rate)
    learn.fine_tune(num_epochs)
    return learn


def predict_tumor_type(img_destination: str) -> None:
    """
    Creates prediction of the image label and prints it as well as
    the confidence of the model it's actually the label of the image.

    Arguments:
        img_destination (str): The image path inside the current working directory.

    Returns:
        None
    """
    prediction, pred_idx, probs = learn.predict(img_destination)
    print(f'Prediction: This is a brain with {prediction}')
    print(f'Probability it\'s a brain with {prediction}: {probs[pred_idx]:.3f}')

    
def update_transformations(data_block: DataBlock, new_item_tfms: List, path, new_batch_tfms=None) -> DataLoaders:
    """
    Creates a new DataBlock with new resizing and/or augmentation item_tfms for the 
    data_block input, shows a batch with 9 images and returns a dataloaders object based on the new datablock.

    Arguments:
        data_block (DataBlock object): An initial DataBlock containing all the variables needed for loading the data into the model.
        new_item_tfms (fast.ai list): New transformation method for the data items.
        new_batch_tfms: New transformation method for the data batches.
        path: The path in which the images exist.
        
    Returns:
        dls (DataLoaders object): A dataloaders object based on the new DataBlock and the folder containing the data.
    """
    set_seed(42, reproducible=True)
    data_block = data_block.new(item_tfms=new_item_tfms, batch_tfms=new_batch_tfms)
    dls = data_block.dataloaders(path)
    dls.show_batch(max_n=9)
    
    return dls
