
# All import
import numpy as np
import pandas as pd
import os
import cv2
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, accuracy_score, f1_score, confusion_matrix
from matplotlib import pyplot as plt
from tqdm import tqdm
import pickle
import itertools


np.random.seed(200)

# Base Model

class baseModel:
    def forward(self, x):
        pass

    def backward(self, output, learning_rate):
        pass


############# Convolution Layer

class ConvolutionLayer(baseModel):
    def __init__(self, output_channel, kernel_size, stride, padding):
        self.output_channel = output_channel
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.weights = None
        self.bias = None
        self.x = None
        self.window_arr = None

    def forward(self, x):
        self.x = x # batch_size, channel, height, width
        batch_size, channel, height, width = x.shape
        output_height = (height - self.kernel_size + 2 * self.padding) // self.stride + 1
        output_width = (width - self.kernel_size + 2 * self.padding) // self.stride + 1
        output = np.zeros((batch_size, self.output_channel,  output_height, output_width))
        if self.weights is None:
            # init weight with Xavier method
            self.weights = np.random.randn(self.output_channel, channel, self.kernel_size, self.kernel_size) * np.sqrt(2 / (channel * self.kernel_size * self.kernel_size))
            self.bias = np.zeros(self.output_channel)
        
        # pad the input
        x_padded = np.pad(x, ((0,0), (0,0), (self.padding, self.padding), (self.padding, self.padding)), 'constant')

        window_arr = np.lib.stride_tricks.as_strided(x_padded, 
            shape=(batch_size, channel, output_height, output_width, self.kernel_size, self.kernel_size), 
            strides=(x_padded.strides[0], x_padded.strides[1], x_padded.strides[2]*self.stride, x_padded.strides[3]*self.stride, x_padded.strides[2], x_padded.strides[3]))

        self.window_arr = window_arr
        output = np.einsum('bihwkl,oikl->bohw', window_arr, self.weights) + self.bias[None, :, None, None]
        return output


    def backward(self, output, learning_rate):
        
        x = self.x
        window_arr = self.window_arr
        dilate = self.stride - 1
        if self.padding == 0:
            padding = self.kernel_size - 1
        else:
            padding = self.padding
        working_input = output

        # pad the input if needed
        if padding != 0:
            working_input = np.pad(working_input, ((0,0), (0,0), (padding, padding), (padding, padding)), 'constant')

        # dilate the input if needed
        if dilate != 0:
            working_input = np.insert(working_input, range(1, output.shape[2]), 0, axis=2)
            working_input = np.insert(working_input, range(1, output.shape[3]), 0, axis=3)

        in_batch, in_channel, out_height, out_width = x.shape
        out_batch, out_channel, _, _ = output.shape
        stride = 1

        window_arr_out = np.lib.stride_tricks.as_strided(working_input,
            shape=(out_batch, out_channel, out_height, out_width, self.kernel_size, self.kernel_size),
            strides=(working_input.strides[0], working_input.strides[1], stride*working_input.strides[2], stride*working_input.strides[3], working_input.strides[2], working_input.strides[3]))

        rotated_kernel = np.rot90(self.weights, 2, axes=(2, 3))

        dx = np.einsum('bohwkl,oikl->bihw', window_arr_out, rotated_kernel)
        db = np.sum(output, axis=(0, 2, 3))
        dw = np.einsum('bihwkl,bohw->oikl', window_arr, output)
        
        self.weights -= learning_rate * dw
        self.bias -= learning_rate * db

        return dx   
        


############# ReLU Layer

class ReLULayer(baseModel):
    def __init__(self):
        self.x = None

    def forward(self, x):
        self.x = x
        return np.maximum(0, x)

    def backward(self, output, learning_rate):
        return output * (self.x > 0)


############ Maxpool Layer

class MaxPoolingLayer(baseModel):
    def __init__(self, pool_size, stride):
        self.pool_size = pool_size
        self.stride = stride
        self.x = None

    def forward(self, x):
        self.x = x
        batch_size, channel, height, width = x.shape
        output_height = (height - self.pool_size) // self.stride + 1
        output_width = (width - self.pool_size) // self.stride + 1
        output = np.zeros((batch_size, channel, output_height, output_width))
        
        # maxpooling without loop
        window_arr = np.lib.stride_tricks.as_strided(x, 
            shape=(batch_size, channel, output_height, output_width, self.pool_size, self.pool_size), 
            strides=(x.strides[0], x.strides[1], x.strides[2] * self.stride, x.strides[3] * self.stride, x.strides[2], x.strides[3]))
        output = np.max(window_arr, axis=(4, 5))
        return output

    def backward(self, output, learning_rate):
        x = self.x
        batch_size, channel, height, width = x.shape
        out_batch, out_channel, out_height, out_width = output.shape

        dx = np.zeros(shape=x.shape)

        for i in range(batch_size):
            for j in range(channel):
                for k in range(out_height):
                    for l in range(out_width):
                        # index i,j with maximum vakue
                        i_t, j_t = np.where(np.max(x[i, j, k * self.stride : k * self.stride + self.pool_size, l * self.stride : l * self.stride + self.pool_size]) == x[i, j, k * self.stride : k * self.stride + self.pool_size, l * self.stride : l * self.stride + self.pool_size])
                        i_t, j_t = i_t[0], j_t[0]
                        # print(i_t, j_t)
                        # i,j index gets value from output 
                        dx[i, j, k * self.stride : k * self.stride + self.pool_size, l * self.stride : l * self.stride + self.pool_size][i_t, j_t] = output[i, j, k, l]
        return dx
        

########### Flattening Layer

class FlatteningLayer(baseModel):
    def __init__(self):
        self.x = None

    def forward(self, x):
        self.x = x
        batch_size, channel, height, width = x.shape
        output = x.reshape((batch_size, channel * height * width))
        return output

    def backward(self, output, learning_rate):
        dx = output.reshape(self.x.shape)
        return dx


######### Fullyconnected Layer

class FullyConnectedLayer(baseModel):
    def __init__(self, output_channel):
        self.output_channel = output_channel
        self.weights = None
        self.bias = None
        self.x = None
        

    def forward(self, x):
        self.x = x

        # print(f'fully connected layer input: {self.x}' )

        if self.weights is None:
            self.weights = np.random.randn(self.x.shape[1], self.output_channel) * np.sqrt(2 / self.x.shape[1])
            self.bias = np.zeros(self.output_channel)
            
        output = np.dot(self.x, self.weights) + self.bias

        # print(f'fully connected layer output: {self.output}' )
        return output

    def backward(self, output, learning_rate):

        dw = np.dot(self.x.T, output)
        db = np.sum(output, axis=0)
        dx = np.dot(output, self.weights.T)

        self.weights -= learning_rate * dw 
        self.bias -= learning_rate * db
        
        return dx

# Softmax Layer

class SoftmaxLayer(baseModel):
    def __init__(self):
        self.x = None

    def forward(self, x):
        self.x = x
        
        # normalize input
        x -= np.max(x, axis=1, keepdims=True)
        x_exp = np.exp(x)
        sum_exp = np.sum(x_exp, axis=1, keepdims=True)
        sum_exp[ sum_exp == 0] = 1
        output = x_exp / sum_exp

        # print(f'softmax layer output: {self.output}' )
        return output

    def backward(self, output, learning_rate):
        return output


################## LOAD DATA #####################

# get images
def getImages(path,cnt):
    images = []
    count = 0

    path_split = path.split('.')
    new_path = path_split[0]
    df = pd.read_csv(path)
    files = df['filename']
    
    for file in files:
        if count == cnt:
            break
        img = cv2.imread(os.path.join(new_path, file))
        img = cv2.resize(img, (28, 28))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = (255-img.transpose(2, 0, 1))/255
        images.append(img)
        count += 1

    return images

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


def load_data():
    count = 10000
    images = getImages('data/training-a.csv', count)
    images += getImages('data/training-b.csv', count)
    images += getImages('data/training-c.csv', count)

    labels = getLabels('data/training-a.csv', count)
    labels = np.concatenate((labels, getLabels('data/training-b.csv', count)))
    labels = np.concatenate((labels, getLabels('data/training-c.csv', count)))

    # one hot encode
    labels = np.eye(10)[labels].astype(int)

    # print(len(images))
    # print(images[0].shape)
    # print(labels.shape)
    # # view an image
    # plt.imshow(images[1].transpose(1, 2, 0))
    # print(labels[1])

    return images, labels


########### Train ############

# train
def train(model, X_train, X_test, Y_train, Y_test, learning_rate, epochs):
    batch_size = 32

    # init metrics array
    train_loss_arr = []
    train_accuracy_arr = []
    val_loss_arr = []
    val_accuracy_arr = []
    val_macroF1_arr = []

    for epoch in range(epochs):
        print(f'epoch: {epoch+1}/{epochs}')
        num_batches = X_train.shape[0] // batch_size
        loss = 0
        accuracy = 0
        for i in tqdm(range(num_batches)):
            # forward
            x_batch = X_train[i*batch_size: (i+1)*batch_size]
            y_output = Y_train[i*batch_size: (i+1)*batch_size]

            x_output = x_batch
            for layer in model:
                x_output = layer.forward(x_output)
            
            #loss
            loss += log_loss(y_output, x_output)
            accuracy += accuracy_score(np.argmax(y_output, axis=1), np.argmax(x_output, axis=1))

            dL = np.copy(x_output)
            dL -= y_output
            dL /= batch_size
            # backward
            for layer in reversed(model):
                dL = layer.backward(dL, learning_rate)
        
        #train loss and accuracy
        train_loss = loss/num_batches
        train_accuracy = accuracy/num_batches

        train_loss_arr.append(train_loss)
        train_accuracy_arr.append(train_accuracy)

        # test
        val_loss = 0
        
        x_out = X_test
        for layer in model:
            x_out = layer.forward(x_out)

        
        val_loss = log_loss(Y_test, x_out)
        val_accuracy = accuracy_score(np.argmax(Y_test, axis=1), np.argmax(x_out, axis=1))
        val_F1 = f1_score(np.argmax(Y_test, axis=1), np.argmax(x_out, axis=1), average='macro')

        val_loss_arr.append(val_loss)
        val_accuracy_arr.append(val_accuracy)
        val_macroF1_arr.append(val_F1)

        print(f'loss: {train_loss}, val_loss: {val_loss}')
        print(f'accuracy: {train_accuracy}, val_accuracy: {val_accuracy}')
        print(f'val_F1: {val_F1}')

    return train_loss_arr, val_loss_arr, val_accuracy_arr, val_macroF1_arr


################### MODEL ################

def createModel():
    model = []
    
    model.append(ConvolutionLayer(6, 5, 1, 0))
    model.append(ReLULayer())
    model.append(MaxPoolingLayer(2, 2))
    model.append(ConvolutionLayer(16, 5, 1, 0))
    model.append(ReLULayer())
    model.append(MaxPoolingLayer(2, 2))
    model.append(FlatteningLayer())
    model.append(FullyConnectedLayer(120))
    model.append(ReLULayer())
    model.append(FullyConnectedLayer(84))
    model.append(ReLULayer())
    model.append(FullyConnectedLayer(10))
    model.append(SoftmaxLayer())
    # print('model created: ', model)
    return model



######################################### TRAIN DATA #####################################

# initialize model
model = createModel()

# train model

images, labels = load_data()

X_train, X_test, Y_train, Y_test = train_test_split(images, labels, test_size=0.2, random_state=42)
X_train = np.array(X_train)
X_test = np.array(X_test)
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)



epochs = 80
learning_rate = 0.001
train_loss_arr, val_loss_arr, val_accuracy_arr, val_macroF1_arr = train(model, X_train, X_test, Y_train, Y_test, learning_rate, epochs)



################ GRAPH PLOT ###############

plt.plot(train_loss_arr, label='train loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()


plt.plot(val_loss_arr, label='validation loss ')
plt.xlabel('Epoch')
plt.ylabel('loss')
plt.legend()
plt.show()


plt.plot(val_accuracy_arr, label='validation accuracy')
plt.xlabel('Epoch')
plt.ylabel('accuracy')
plt.legend()
plt.show()

plt.plot(val_macroF1_arr, label='validation macro F1 score')
plt.xlabel('Epoch')
plt.ylabel('Macro F1')
plt.legend()
plt.show()

################# SAVE MODEL ################


with open('1705067_model.pkl', 'wb') as f:
    pickle.dump(model, f)