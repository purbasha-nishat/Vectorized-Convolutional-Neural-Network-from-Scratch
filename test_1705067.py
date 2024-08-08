import numpy as np
import pandas as pd
import os
import cv2
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, accuracy_score, f1_score, confusion_matrix
from matplotlib import pyplot as plt
from tqdm import tqdm
import pickle
from train_1705067 import *
import sys
import csv

np.random.seed(200)

# get images
def getImages(path):
    images = []
    filenames = []
    for file in sorted(os.listdir(path)):
        img = cv2.imread(os.path.join(path, file))
        img = cv2.resize(img, (28, 28))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = (255-img.transpose(2, 0, 1))/255
        images.append(img)
        filenames.append(file)

    return images, filenames


# get labels
def getLabels(path,cnt):
    labels = []
    count = cnt
    
    df = pd.read_csv(path)
    # print(len(df))
    if count >= len(df):
        count = len(df)
    labels = df['digit'][:count]
    return np.array(labels)


def load_test_data():
    count = 100000
    images = getImages('data/training-d.csv', count)

    labels = getLabels('data/training-d.csv', count)

    # one hot encode
    labels = np.eye(10)[labels].astype(int)

    print("len: ",len(images))
    print(images[0].shape)
    print(labels.shape)
    
    return images, labels


# def predict(model, X_test, Y_test):
#     x_out = X_test
#     for layer in model:
#         x_out = layer.forward(x_out)

#     val_accuracy = accuracy_score(np.argmax(Y_test, axis=1), np.argmax(x_out, axis=1))
#     val_F1 = f1_score(np.argmax(Y_test, axis=1), np.argmax(x_out, axis=1), average='macro')
#     print(f'val_accuracy: {val_accuracy}')
#     print(f'val_F1: {val_F1}')

#     return x_out

def predict(model, X_test):
    x_out = X_test
    for layer in model:
        x_out = layer.forward(x_out)
    return x_out


if __name__ == "__main__":

    path = sys.argv[1]
    X_test, filename = getImages(path)
    X_test = np.array(X_test)

    # load model
    with open('1705067_model.pickle', 'rb') as f:
        model = pickle.load(f)

    Y_pred = predict(model, X_test)
    Y_pred = np.array(Y_pred)

    # print("----------CSV----------")
    # Create a new CSV file
    with open("1705067_prediction.csv", "w", newline="") as f:
        writer = csv.writer(f)

        # Write the header row
        writer.writerow(["FileName", "Digit"])

        # Write data in a loop
        for i in range(Y_pred.shape[0]):
            row = [filename[i], np.argmax(Y_pred[i])]
            writer.writerow(row)

