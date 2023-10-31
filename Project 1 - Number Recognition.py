#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 10:37:22 2023

@author: aniltanaktan
"""
import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib.cm as cm 

# Import data from local folder. x = data, y = labels

# Paths in local directory
train_x_p = "./data/train-images.idx3-ubyte"
train_y_p = "./data/train-labels.idx1-ubyte"

test_x_p = "./data/t10k-images.idx3-ubyte"
test_y_p = "./data/t10k-labels.idx1-ubyte"

train_x = open(train_x_p, "rb")
train_y = open(train_y_p, "rb")
test_x = open(test_x_p, "rb")
test_y = open(test_y_p, "rb")


# Pixels start from 16 and labels start from 8 according to http://yann.lecun.com/exdb/mnist/
train_x.read(16)
train_y.read(8)
test_x.read(16)
test_y.read(8)

# Amount of datapoints
train_n = 60000
test_n = 10000

train = []
train_l = []
for i in range(train_n):
    label = [ord(train_y.read(1))]
    train_l.append(label)
    train_temp =[]
    for pixel in range(784):
        # append each pixel's Greyscale value to the related list and also divide them by 255 for faster training
        train_temp.append(ord(train_x.read(1))/255.) 
    train.append(train_temp)

train_x.close()
train_y.close()

# Now we will do the same for the test data
test = []
test_l = []
for i in range(test_n):
    label = [ord(test_y.read(1))]
    test_l.append(label)
    test_temp =[]
    for pixel in range(784):
        test_temp.append(ord(test_x.read(1))/255.) 
    test.append(test_temp)

test_x.close()
test_y.close()



train= np.array(train).T
train_l= np.array(train_l).T

test= np.array(test).T
test_l= np.array(test_l).T

"""
m, n = train.shape
print(m,n)

m, n = train_l.shape
print(m,n)

m, n = test.shape
print(m,n)

m, n = test_l.shape
print(m,n)
"""

"""
# Visualizing the training data and see how frequent each digit occured
index = 1049
plt.title((train_l[index]))
plt.imshow(train[index].reshape(28,28), cmap=cm.binary)

print("Train Data Frequencies:")
y_value=np.zeros((1,10))
for i in range (10):
    print(i,"occurances: ",np.count_nonzero(train_l==i))
    y_value[0,i-1]= np.count_nonzero(train_l==i)

"""


# Now we have 2 arrays, train is 60000x784 matrix, contains each data, 
#                   and train_l is 60000x1 matrix, contains each label
# and we have 2 arrays, test is 10000x784 matrix, contains each data, 
#                   and test_l is 10000x1 matrix, contains each label

# Neural network will contain 2 layers, one with 784 features (pixels) and one with 10 features (classes)
# and a hidden layers with N = 300, 500, 1000
# learning coefficient = 0.01, 0.05, 0.09
# activation function as tanh and RELU

# Initializing weights, b0 (bias) and needed activation functions and their derivatives for backpropagation

def initialize(N): # Weights have -0.01 : 0.01 uniform dist.
    W1 = np.random.uniform(-0.01, 0.01, (N, 784))
    b1 = np.random.uniform(-0.01, 0.01, (N, 1))
    W2 = np.random.uniform(-0.01, 0.01, (10, N))
    b2 = np.random.uniform(-0.01, 0.01, (10, 1))
    return W1, b1, W2, b2

def RELU(v):
    return np.maximum(0, v)

def d_RELU(v):
    return v>0

def tanh(v):
    return (np.exp(v)-np.exp(-v))/(np.exp(v)+np.exp(-v))

def d_tanh(v):
    a = (np.exp(v)-np.exp(-v))/(np.exp(v)+np.exp(-v))
    return 1-a**2

def sigmoid(v):
    return 1/(1+np.exp(-v))

def d_sigmoid(v):
    return sigmoid(v) * (1 - sigmoid(v))
    

# Functions for each case 3x3x2 = 18

# For Case 1, y will have 1 for true class, and -1 for others
def one_min_one(y):
    one_min_one_y = np.zeros((y.size, y.max() + 1)) - 1
    one_min_one_y[np.arange(y.size), y] = 1 # each row of the minus one vector which corresponds to classified number will become 1
    # for example: if classification = 5, 5th elementh of zeroes vector will become 1, others will stay as -1
    one_min_one_y = one_min_one_y.T
    return one_min_one_y
    

# For Case 2, y will have 1 for true class, and 0 for others
def one_zero(y):
    one_zero_y = np.zeros((y.size, y.max() + 1))
    one_zero_y[np.arange(y.size), y] = 1 # each row of the zeroes vector which corresponds to classified number will become 1
    # for example: if classification = 5, 5th elementh of zeroes vector will become 1
    one_zero_y = one_zero_y.T
    return one_zero_y


# A function to simply pass the inputs X through initialized weights and biases to get v1, v2, o1, o2
def forward_propagation(W1, b1, W2, b2, X, activation):
    
    v1 = W1.dot(X)+b1
    if activation == "relu":
        
        o1 = RELU(v1)
    
        v2 = W2.dot(o1)+b2

        o2 = sigmoid(v2)
    
    elif activation == "tanh":
        
        o1 = tanh(v1)
    
        v2 = W2.dot(o1)+b2

        o2 = sigmoid(v2)
    
    else:
        raise ValueError("Use activation = 'relu' to use relu for hidden layer and sigmoid for output layer. \n Use activation = 'tanh' to use tanh for all layers.")
    
    return v1, o1, v2, o2

# Main function to find the 
def back_propagation(v1, o1, v2, o2, W2, x, y, activation):
    m = y.size
    
    if activation == "relu":
        
        dv2 = o2 - one_zero(y)
        # Gradient for Output Layer
        dW2 = (1/m) * dv2.dot(o1.T)
        db2 = (1/m) * np.sum(dv2, axis=1).reshape(-1,1)
    
        dv1 = W2.T.dot(dv2) * d_RELU(v1)
        # Gradient for Hidden Layer
        dW1 = (1/m) * dv1.dot(x.T)
        db1 = (1/m) * np.sum(dv1, axis=1).reshape(-1,1)
        
        E = np.sum((o2 - one_zero(y)) ** 2)
        Eave = np.sum(E) / m
        
    elif activation == "tanh":
        
        dv2 = o2 - one_min_one(y)
        # Gradient for Output Layer
        dW2 = (1/m) * dv2.dot(o1.T)
        db2 = (1/m) * np.sum(dv2, axis=1).reshape(-1,1)
    
        dv1 = W2.T.dot(dv2) * d_tanh(v1)
        # Gradient for Hidden Layer
        dW1 = (1/m) * dv1.dot(x.T)
        db1 = (1/m) * np.sum(dv1, axis=1).reshape(-1,1)
        
        E = np.sum((o2 - one_min_one(y)) ** 2)
        Eave = np.sum(E) / m
        
    else:
        raise ValueError("Use activation = 'relu' to use relu for hidden layer and sigmoid for output layer. \n Use activation = 'tanh' to use tanh for all layers.")
    
    
    return dW1, db1, dW2, db2, E, Eave

# Update weights with the gradients of hidden and output layer
def update_weights(W1, b1, W2, b2, dW1, db1, dW2, db2, l_rate):
    W1 = W1 - l_rate*dW1
    b1 = b1 - l_rate*db1
    W2 = W2 - l_rate*dW2
    b2 = b2 - l_rate*db2
    
    return W1, b1, W2, b2

# Functions for getting results
# Find the average succes rate
def accuracy(predictions, y):
    return np.sum(predictions == y) / y.size

# For passing test data with found W1, W2, b1 and b2
def pass_data(X, W1, b1, W2, b2):
    v1, o1, v2, o2 = forward_propagation(W1, b1, W2, b2, X, "relu")
    predictions = np.argmax(o2, 0)
    return predictions

# Main function, combine all algorithms
def neural_network(x, y, epochs, l_rate, N, activation):
    W1, b1, W2, b2 = initialize(N) # Initialize weights and biases
    start_time = time.time() # Calculate time for all epochs to end
    
    for i in range(epochs): #in each epoch
        v1, o1, v2, o2 = forward_propagation(W1, b1, W2, b2, x, activation) # Pass the data through weights to find v's and o's
        dW1, db1, dW2, db2, E, Eave = back_propagation(v1, o1, v2, o2, W2, x, y, activation) # find gradients of weights for 2 layers
        W1, b1, W2, b2 = update_weights(W1, b1, W2, b2, dW1, db1, dW2, db2, l_rate) # update weights
        
        if i % 10 == 0: # Show accuarcy for per 10 epochs
            print("---------")
            print("Epoch: ", i)
            print("Training Set Success Rate:", accuracy(np.argmax(o2, 0), y))
            print("Mean Square Error:", Eave)
        
    end_time = time.time()
    
    
    # End results for the neural network model
    print("---------")
    print("Number of Epochs:", epochs)
    print("Time:", end_time - start_time, "seconds\n")
    print("        Mean Square Error:", Eave)
    print("Training Set Success Rate:", accuracy(np.argmax(o2, 0), y))
    print("    Test Set Success Rate:", accuracy(pass_data(test, W1, b1, W2, b2), test_l))
    print("           Test Set Error:", 1- accuracy(pass_data(test, W1, b1, W2, b2), test_l))
    
    return W1, b1, W2, b2

# Function: Neural Network
# Inputs: (x, y, number of epochs, learning rate, hidden layer neuron number N, activation function ("relu" or "tanh"))
# (CASE 1): Use activation = 'tanh' to use tanh for all layers. 
# (CASE 2): Use activation = 'relu' to use relu for hidden layer and sigmoid for output layer. 
W1, b1, W2, b2 = neural_network(train, train_l, 100, 0.09, 300, "relu")






def test_prediction(index, W1, b1, W2, b2):
    current_image = train[:, index, None]
    prediction = pass_data(train[:, index, None], W1, b1, W2, b2)
    label = train_l[:, index, None]
    print("Prediction: ", prediction)
    print("Label: ", label)
    
    current_image = current_image.reshape((28, 28)) * 255
    plt.gray()
    plt.imshow(current_image, interpolation='nearest')
    plt.show()

test_prediction(0, W1, b1, W2, b2)
test_prediction(1, W1, b1, W2, b2)
test_prediction(2, W1, b1, W2, b2)
test_prediction(3, W1, b1, W2, b2)





