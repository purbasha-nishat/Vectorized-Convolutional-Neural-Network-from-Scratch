# Assignment 4

This repository contains the final assignment of CSE 472: Machine Learning Sessional offered by the CSE Department of BUET.

The task is to build a vectorized version of a Convolutional Neural Network using only numpy without any deeplearning frameworks. Training and testing of the developed model is done on the [NumtaDB: Bengali Handwritten Digits](https://www.kaggle.com/datasets/BengaliAI/numta) dataset. The `trainig-a`, `training-b` and `training-c` folders are used for training and `training-d` is used for testing. You can find the assignment details [here](Assignment_4_V3.pdf). A detailed report on the various experiments and final results can be found in [1705067_report.pdf](1705067_report.pdf).

## Model Blocks

The repository contains fully vectorized implementations for Convolution, MaxPool, Fully Connected layers. 

<img src="figures\model.JPG"/>

## Datasets and Hyperparameters

<img src="figures\datasets.JPG"/>

## Training

In order to run your model in the numta dataset, please download the dataset from [here](https://www.kaggle.com/datasets/BengaliAI/numta) to the resources directory. To run the training script, first install the packages from requirements.txt and run `python train_1705067.py`

## Testing

To run the testing script, first install the packages from requirements.txt and run `python test_1705067.py`


## Results

Results for learning rate 0.001 and 0.01 respectively:

#### Epochs vs Train loss
<p align="center">
    <img src="0.001_learning_rate\1.png" width="45%" alt="S2">
    <img src="0.01_learning_rate\train_loss.png" width="45%" alt="S1">
</p>

#### Epochs vs Validation loss
<p align="center">
    <img src="0.001_learning_rate\2.png" width="45%" alt="S2">
    <img src="0.01_learning_rate\validation_loss.png" width="45%" alt="S1">
</p>

#### Epochs vs Validation Accuracy
<p align="center">
    <img src="0.001_learning_rate\3.png" width="45%" alt="S2">
    <img src="0.01_learning_rate\validation_acc.png" width="45%" alt="S1">
</p>

#### Epochs vs Validation Macro-F1 Score
<p align="center">
    <img src="0.001_learning_rate\4.png" width="45%" alt="S2">
    <img src="0.01_learning_rate\validation_F1.png" width="45%" alt="S1">
</p>

#### Confusion Matrix for learning Rate 0.01
<p align="center">
    <img src="0.01_learning_rate\conf.png" width="50%" alt="S1">
</p>

## Independent Test Performance

The best model was chosen based on best F1-score. Then the chosen model was used to predict the labels of images from the ‘training-d’ set.
Accuracy: 0.8253575357535754
Macro_F1 Score: 0.8207252304239467

<p align="center">
    <img src="0.01_learning_rate\conf_1.png" width="50%" alt="S1">
</p>