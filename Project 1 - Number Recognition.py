#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 10:37:22 2023

@author: aniltanaktan
"""

import matplotlib.pyplot as plt
import matplotlib.cm as cm 
import numpy as np



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
    test.append(train_temp)

test_x.close()
test_y.close()



train= np.array(train)
train_l= np.array(train_l)

"""
m, n = train.shape
print(m,n)

m, n = train_l.shape
print(m,n)
"""

test= np.array(test)
test_l= np.array(test_l)

"""
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
# activation function as tanh and RELU
# learning coefficient = 0.01, 0.05, 0.09

# Initializing weights, b0 (bias) and needed activation functions

def initialize(N): # Weights have -0.01 : 0.01 uniform dist.
    W1 = np.random.uniform(-0.01, 0.01, (N, 784))
    b1 = np.random.uniform(-0.01, 0.01, (N, 1))
    W2 = np.random.uniform(-0.01, 0.01, (N, 784))
    b2 = np.random.uniform(-0.01, 0.01, (N, 1))
    return W1, b1, W2, b2

def RELU(v):
    return np.maximum(0, v)

def d_RELU(v):
    if v > 0:
        return 1
    
    else:
        return 0

def tanh(v):
    return (np.exp(v)-np.exp(-v))/(np.exp(v)+np.exp(-v))

def d_tanh(v):
    a = (np.exp(v)-np.exp(-v))/(np.exp(v)+np.exp(-v))
    return 1-a**2

# Functions for each case 3x3x2 = 18

# For Case 1, y will have 1 for true class, and -1 for others

def one_min_one(y):
    one_min_one_y = np.zeros((y.size, y.max() + 1)) - 1
    one_min_one_y[np.arrange(y.size), y] = 1 # each row of the minus one vector which corresponds to classified number will become 1
    # for example: if classification = 5, 5th elementh of zeroes vector will become 1, others will stay as -1
    one_min_one_y = one_min_one_y.T
    return one_min_one_y
    

# For Case 2, y will have 1 for true class, and 0 for others

def one_zero(y):
    one_zero_y = np.zeros((y.size, y.max() + 1))
    one_zero_y[np.arrange(y.size), y] = 1 # each row of the zeroes vector which corresponds to classified number will become 1
    # for example: if classification = 5, 5th elementh of zeroes vector will become 1
    one_zero_y = one_zero_y.T
    return one_zero_y
    
def forward_propagation(v1, o1, v2, o2, W2, y, activation):
    
    return

def back_propagation(v1, o1, v2, o2, W2, y, activation):
    
    return






