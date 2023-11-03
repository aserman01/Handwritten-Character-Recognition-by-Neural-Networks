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

        o2 = tanh(v2)
    
    else:
        raise ValueError("Use activation = 'relu' to use relu for hidden layer and sigmoid for output layer. \n Use activation = 'tanh' to use tanh for all layers.")
    
    return v1, o1, v2, o2

# Main function to find the gradients using backpropagation
def back_propagation(v1, o1, v2, o2, W1, W2, x, y, lambda_v, activation):
    m = y.size
     
    if activation == "relu":
        E = np.sum((o2 - one_zero(y)) ** 2) + (lambda_v / 2) * (np.sum(W1 ** 2) + np.sum(W2 ** 2))
        Eave = np.sum(E) / m 
        
        # Regularized term for the gradients
        reg_term_W2 = lambda_v * W2
        reg_term_W1 = lambda_v * W1
        
        # error for output layer
        e2 = o2 - one_zero(y)
        # Gradient for Output Layer with L2 regularization term
        dW2 = (1/m) * (e2.dot(o1.T)) + reg_term_W2
        db2 = (1/m) * np.sum(e2, axis=1).reshape(-1,1)
        
        # error for hidden layer
        e1 = (W2.T.dot(e2)) * d_RELU(v1)
        # Gradient for Hidden Layer with L2 regularization term
        dW1 = (1/m) * (e1.dot(x.T)) + reg_term_W1
        db1 = (1/m) * np.sum(e1, axis=1).reshape(-1,1)
    
    elif activation == "tanh":
        E = np.sum((o2 - one_min_one(y)) ** 2) + (lambda_v / 2) * (np.sum(W1 ** 2) + np.sum(W2 ** 2))
        Eave = np.sum(E) / m 
        
        # Regularized term for the gradients
        reg_term_W2 = lambda_v * W2
        reg_term_W1 = lambda_v * W1
        
        # error for output layer
        e2 = o2 - one_min_one(y)
        # Gradient for Output Layer
        dW2 = (1/m) * e2.dot(o1.T) + reg_term_W2
        db2 = (1/m) * np.sum(e2, axis=1).reshape(-1,1)
        
        # error for hidden layer
        e1 = W2.T.dot(e2) * d_tanh(v1)
        # Gradient for Hidden Layer
        dW1 = (1/m) * e1.dot(x.T) + reg_term_W1
        db1 = (1/m) * np.sum(e1, axis=1).reshape(-1,1)
        
        
        
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

# Functions for getting results after training is complete
# Find the average succes rate
def accuracy(predictions, y):
    return np.sum(predictions == y) / y.size

# For passing test data with found W1, W2, b1 and b2
def pass_data(X, W1, b1, W2, b2, activation):
    v1, o1, v2, o2 = forward_propagation(W1, b1, W2, b2, X, activation)
    predictions = np.argmax(o2, 0)
    return predictions

# Main function, combine all algorithms
def neural_network(x, y, epochs, l_rate, N, lambda_v, activation):
    W1, b1, W2, b2 = initialize(N) # Initialize weights and biases
    start_time = time.time() # Calculate time for all epochs to end
    
    for i in range(epochs): #in each epoch
        v1, o1, v2, o2 = forward_propagation(W1, b1, W2, b2, x, activation) # Pass the data through weights to find v's and o's
        dW1, db1, dW2, db2, E, Eave = back_propagation(v1, o1, v2, o2, W1, W2, x, y, lambda_v, activation) # find gradients of weights for 2 layers
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
    print("Mean Square Error:", Eave)
    print("Training Set Success Rate:", accuracy(np.argmax(o2, 0), y))
    print("Test Set Success Rate:", accuracy(pass_data(test, W1, b1, W2, b2, activation), test_l))
    print("Test Set Error:", 1- accuracy(pass_data(test, W1, b1, W2, b2, activation), test_l))
    
    return W1, b1, W2, b2

# Function: Neural Network
# Inputs: (x, y, number of epochs, learning rate, hidden layer neuron number N, lambda_v, activation function ("relu" or "tanh"))
# (CASE 1): Use activation = 'tanh' to use tanh for all layers. 
# (CASE 2): Use activation = 'relu' to use relu for hidden layer and sigmoid for output layer. 
W1, b1, W2, b2 = neural_network(train, train_l, 400, 0.09, 100, 0, "tanh")








def test_prediction(index, W1, b1, W2, b2, activation):
    current_image = train[:, index, None]
    prediction = pass_data(train[:, index, None], W1, b1, W2, b2, activation)
    label = train_l[:, index, None]
    print("Prediction: ", prediction)
    print("Label: ", label)
    
    current_image = current_image.reshape((28, 28)) * 255
    plt.gray()
    plt.imshow(current_image, interpolation='nearest')
    plt.title('Prediction: {}, Label: {}'.format(prediction, label[0]))
    plt.show()

test_prediction(0, W1, b1, W2, b2, "tanh")
test_prediction(1, W1, b1, W2, b2, "tanh")
test_prediction(2, W1, b1, W2, b2, "tanh")
test_prediction(3, W1, b1, W2, b2, "tanh")
test_prediction(1042, W1, b1, W2, b2, "tanh")


"""
# Neural network will contain 2 layers, one with 784 features (pixels) as the input and one with 10 features (classes) as the output
# and a hidden layers with N = 300, 500, 1000 in between.
# learning coefficient = 0.01, 0.05, 0.09
# activation function as tanh and RELU

Notes:  -At beginning epochs, as weights are random, the success rate is around 0.1 which is random guessing.
        -As the learning rate is so low, our weights didn't converge to true weights in time.
        -RELU function is around x3 faster however tanh activation function results in a higher success rate.
        -Using a larger hidden layer slows down training time but grants a better success rate in the end.
        Using ReLu with a high learning rate succeeded in the best balance between training time and success rate.
        -Because of the early stopping of the training (at 100 epochs), it seems like our model is not overfitted. 
        That is why introducing regularization has no big effect.

# 1 - N = 300:
    1.1. l_rate = 0.01:
        1.1.1. tanh:
            
            Number of Epochs: 100
            Time: 190.25479006767273 seconds

            Mean Square Error: 3.590489151416886
            Training Set Success Rate: 0.38461666666666666
            Test Set Success Rate: 0.4352
            Test Set Error: 0.5648
            
        1.1.2. relu:
        
            Number of Epochs: 100
            Time: 67.26897048950195 seconds

            Mean Square Error: 0.9277162446213076
            Training Set Success Rate: 0.12015
            Test Set Success Rate: 0.1226
            Test Set Error: 0.8774
            
    
    1.2. l_rate = 0.03:
        1.2.1. tanh:
            Number of Epochs: 100
            Time: 175.89676141738892 seconds

            Mean Square Error: 2.5207181522633246
            Training Set Success Rate: 0.71725
            Test Set Success Rate: 0.6606
            Test Set Error: 0.33940000000000003
            
        1.2.2. relu:
            Number of Epochs: 100
            Time: 63.6363525390625 seconds

            Mean Square Error: 0.9036041746809448
            Training Set Success Rate: 0.25711666666666666
            Test Set Success Rate: 0.2655
            Test Set Error: 0.7344999999999999
            
           
    1.3. l_rate = 0.09:
        1.3.1. tanh:
            
            Number of Epochs: 100
            Time: 189.34866499900818 seconds

            Mean Square Error: 1.193835694288484
            Training Set Success Rate: 0.86065
            Test Set Success Rate: 0.8582
            Test Set Error: 0.14180000000000004
            
        1.3.2. relu:
            Number of Epochs: 100
            Time: 64.31954503059387 seconds

            Mean Square Error: 0.5432723696851491
            Training Set Success Rate: 0.7431
            Test Set Success Rate: 0.7519
            Test Set Error: 0.2481
                        

  2 - N = 500:
    2.1. l_rate = 0.01:
        2.1.1. tanh:
            
            Number of Epochs: 100
            Time: 316.01254177093506 seconds

            Mean Square Error: 3.5382426501077355
            Training Set Success Rate: 0.6108666666666667
            Test Set Success Rate: 0.6175
            Test Set Error: 0.38249999999999995
            
        2.1.2. relu:
            Number of Epochs: 100
            Time: 103.2268660068512 seconds

            Mean Square Error: 0.9256072513599892
            Training Set Success Rate: 0.18455
            Test Set Success Rate: 0.184
            Test Set Error: 0.8160000000000001
    
    2.2. l_rate = 0.03:
        2.2.1. tanh:
            Number of Epochs: 100
            Time: 303.54504585266113 seconds

            Mean Square Error: 2.305068160955473
            Training Set Success Rate: 0.7529333333333333
            Test Set Success Rate: 0.7629
            Test Set Error: 0.23709999999999998
            
        2.2.2. relu:
            Number of Epochs: 100
            Time: 104.78797602653503 seconds

            Mean Square Error: 0.8930698598559726
            Training Set Success Rate: 0.4789833333333333
            Test Set Success Rate: 0.4951
            Test Set Error: 0.5049
            
           
    2.3. l_rate = 0.09:
        2.3.1. tanh:
            Number of Epochs: 100
            Time: 294.28605222702026 seconds

            Mean Square Error: 1.138310291933822
            Training Set Success Rate: 0.86865
            Test Set Success Rate: 0.8774
            Test Set Error: 0.12260000000000004
            
        2.3.2. relu:
            Number of Epochs: 100
            Time: 99.51801633834839 seconds

            Mean Square Error: 0.4975329447238578
            Training Set Success Rate: 0.7751833333333333
            Test Set Success Rate: 0.7859
            Test Set Error: 0.21409999999999996
            
  3 - N = 1000:
    3.1. l_rate = 0.01:
        3.1.1. tanh:
            Number of Epochs: 100
            Time: 553.5273218154907 seconds

            Mean Square Error: 3.3780574602148916
            Training Set Success Rate: 0.71315
            Test Set Success Rate: 0.7165
            Test Set Error: 0.2835
            
        3.1.2. relu:
            Number of Epochs: 100
            Time: 171.98229837417603 seconds

            Mean Square Error: 0.9222179338942486
            Training Set Success Rate: 0.19208333333333333
            Test Set Success Rate: 0.1924
            Test Set Error: 0.8076
    
    3.2. l_rate = 0.03:
        3.2.1. tanh:
            Number of Epochs: 100
            Time: 543.1981992721558 seconds

            Mean Square Error: 2.018986430644982
            Training Set Success Rate: 0.7882
            Test Set Success Rate: 0.796
            Test Set Error: 0.20399999999999996
            
        3.2.2. relu:
            Number of Epochs: 100
            Time: 171.20726227760315 seconds

            Mean Square Error: 0.8647632131167379
            Training Set Success Rate: 0.67555
            Test Set Success Rate: 0.6829
            Test Set Error: 0.31710000000000005
            
           
    3.3. l_rate = 0.09:
        3.3.1. tanh:
            Number of Epochs: 100
            Time: 548.7319149971008 seconds
            
            Mean Square Error: 1.0829777295931151
            Training Set Success Rate: 0.87435
            Test Set Success Rate: 0.8821
            Test Set Error: 0.1179
            
        3.3.2. relu:
            Number of Epochs: 100
            Time: 170.9559588432312 seconds

            Mean Square Error: 0.45442036881538966
            Training Set Success Rate: 0.79805
            Test Set Success Rate: 0.8071
            Test Set Error: 0.19289999999999996

"""
# BEST ONE: tanh with N = 1000, l_rate = 0.09 with 0.8821 test success


"""
Now N= 10, 50, 100

N=10:
    
    Number of Epochs: 100
    Time: 12.301090240478516 seconds

    Mean Square Error: 2.684031240669328
    Training Set Success Rate: 0.5734333333333334
    Test Set Success Rate: 0.577
    Test Set Error: 0.42300000000000004

N=50:
    
    Number of Epochs: 100
    Time: 36.696799755096436 seconds

    Mean Square Error: 1.5410302849045827
    Training Set Success Rate: 0.8162
    Test Set Success Rate: 0.8263
    Test Set Error: 0.17369999999999997

N=100:
    
    Number of Epochs: 100
    Time: 62.340402603149414 seconds

    Mean Square Error: 1.3415265775541685
    Training Set Success Rate: 0.8403166666666667
    Test Set Success Rate: 0.8503
    Test Set Error: 0.14970000000000006


lambda=0.01 and lambda=0.001

lambda=0.01:
    Number of Epochs: 100
    Time: 61.98282790184021 seconds

    Mean Square Error: 1.395543985838822
    Training Set Success Rate: 0.8339166666666666
    Test Set Success Rate: 0.8427
    Test Set Error: 0.1573

lambda=0.001:
    Number of Epochs: 100
    Time: 60.69019412994385 seconds

    Mean Square Error: 1.3617456877778122
    Training Set Success Rate: 0.8364666666666667
    Test Set Success Rate: 0.8462
    Test Set Error: 0.15380000000000005
    
lambda=0:
    Number of Epochs: 100
    Time: 62.340402603149414 seconds

    Mean Square Error: 1.3415265775541685
    Training Set Success Rate: 0.8403166666666667
    Test Set Success Rate: 0.8503
    Test Set Error: 0.14970000000000006

"""

# BEST, 400 epochs

"""
Number of Epochs: 400
Time: 255.46789479255676 seconds

Mean Square Error: 0.6437990591401012
Training Set Success Rate: 0.9109833333333334
Test Set Success Rate: 0.9166
Test Set Error: 0.08340000000000003
"""
