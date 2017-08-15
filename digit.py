'''
August 2017
@author: Niv Vosco
'''

import math
import numpy as np
import pandas as pd

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def randInitializeWeights(in_layer_size, out_layer_size):
    return (np.random.rand(out_layer_size, in_layer_size + 1) * 2 * epsilon - epsilon)

def getParametersOfLayer(theta, layer, X):
    m,n = X.shape
    ptr_first = 0
    ptr_last = (hidden_layer_size * (input_layer_size + 1))
    a = np.hstack((np.ones((X.shape[0], 1)), X))
    theta1 = (theta[ptr_first:ptr_last]).reshape(hidden_layer_size, input_layer_size + 1)
    z = np.matmul(a ,theta1.T)
    for i in range(0,layer):
        ptr_first = ptr_last
        if (i < num_hidden_layers):
            ptr_last += (layer_size[i+2] * (layer_size[i+1] + 1))
        thetal = (theta[ptr_first:ptr_last]).reshape(layer_size[i+2], layer_size[i+1] + 1)
        a = np.hstack((np.ones((z.shape[0], 1)), sigmoid(z)))
        z = np.matmul(a ,thetal.T)
    return [a,z]

def forwardPropogation(theta, X):
    m,n = X.shape
    ptr_first = 0
    ptr_last = (hidden_layer_size * (input_layer_size + 1))
    z = X
    a = np.hstack((np.ones((z.shape[0], 1)), z))
    for i in range(0,num_hidden_layers+1):
        thetal = (theta[ptr_first:ptr_last]).reshape(layer_size[i+1], layer_size[i] + 1)
        ptr_first = ptr_last
        if (i < num_hidden_layers):
            ptr_last += (layer_size[i+2] * (layer_size[i+1] + 1))
        z = np.matmul(a ,thetal.T)
        a = np.hstack((np.ones((z.shape[0], 1)), sigmoid(z)))
    return a[:,1:]

def gradientDescent(theta, X, y, debug):
    m,n = X.shape
    a1 = np.hstack((np.ones((X.shape[0], 1)), X))
    delta4 = np.zeros((m, num_labels))
    if debug:
        print("Start: gradient descent")
    for k in range(0, iter_grad):
        ptr = 0
        theta1 = (theta[ptr:(ptr+(hidden_layer_size * (input_layer_size + 1)))]).reshape(hidden_layer_size, input_layer_size + 1)
        ptr += (hidden_layer_size * (input_layer_size + 1))
        theta2 = (theta[ptr:(ptr+(hidden_layer_size * (hidden_layer_size + 1)))]).reshape(hidden_layer_size, hidden_layer_size + 1)
        ptr += (hidden_layer_size * (hidden_layer_size + 1))
        theta3 = (theta[ptr:(ptr+(hidden_layer_size * (input_layer_size + 1)))]).reshape(num_labels, hidden_layer_size + 1)
        z2 = np.matmul(a1 ,theta1.T)
        a2 = np.hstack((np.ones((z2.shape[0], 1)), sigmoid(z2)))
        z3 = np.matmul(a2 ,theta2.T)
        a3 = np.hstack((np.ones((z3.shape[0], 1)), sigmoid(z3)))
        z4 = np.matmul(a3 ,theta3.T)
        a4 = sigmoid(z4)
        cost = 0
        for i in range(0,m):
            for j in range(0, num_labels):
                if (y.item(i,0) == j):
                    if debug:
                        cost = cost - math.log(a4[i,j])
                    delta4[i,j] = a4[i,j] - 1
                else:
                    if debug:
                        cost = cost - math.log(1 - a4[i,j])
                    delta4[i,j] = a4[i,j]

        if (debug and k % (iter_grad / 10) == 0):
            cost = cost / m
            T = np.sum(np.square(np.delete(theta1, -1, axis=1)))
            T = T + np.sum(np.square(np.delete(theta2, -1, axis=1)))
            cost = cost + ((T * lambda_reg) / (2 * m))
            print("Iteration %d, cost function value: %f" % (k, cost))

        theta3_grad = (np.matmul(a3.T, delta4) / m).T
        theta3_t = (np.delete(theta3.T, 0, 0)).T
        theta3_t = np.hstack((np.zeros((theta3_t.shape[0], 1)), theta3_t))
        theta3_grad = theta3_grad + (lambda_reg / m) * theta3_t

        z3 = np.hstack((np.ones((z3.shape[0], 1)), z3))
        delta3 = np.multiply(np.matmul(delta4, theta3), (np.multiply(sigmoid(z3), (1 - sigmoid(z3)))))
        delta3 = (np.delete(delta3.T, 0, 0)).T
        theta2_grad = (np.matmul(a2.T, delta3) / m).T
        theta2_t = (np.delete(theta2.T, 0, 0)).T
        theta2_t = np.hstack((np.zeros((theta2_t.shape[0], 1)), theta2_t))
        theta2_grad = theta2_grad + (lambda_reg / m) * theta2_t

        z2 = np.hstack((np.ones((z2.shape[0], 1)), z2))
        delta2 = np.multiply(np.matmul(delta3, theta2), (np.multiply(sigmoid(z2), (1 - sigmoid(z2)))))
        delta2 = (np.delete(delta2.T, 0, 0)).T
        theta1_grad = (np.matmul(a1.T, delta2) / m).T
        theta1_t = (np.delete(theta1.T, 0, 0)).T
        theta1_t = np.hstack((np.zeros((theta1_t.shape[0], 1)), theta1_t))
        theta1_grad = theta1_grad + (lambda_reg / m) * theta1_t
        theta_grad = (np.append(theta1_grad.reshape(-1,1), theta2_grad.reshape(-1,1))).reshape(-1,1)
        theta_grad = (np.append(theta_grad, theta3_grad.reshape(-1,1))).reshape(-1,1)
        theta = theta - (alpha * theta_grad)
    if debug:
        print("Done: gradient descent")
    return theta

def predict(theta, x):
    m,n = x.shape
    pred = np.zeros((m, 1))
    h = forwardPropogation(theta, x)
    for i in range(0,m):
        h_i = ((h[i,:]).T).reshape(-1,1)
        pred[i] = ((h_i.argmax(axis=0))[0])
    return pred

def calculateAccuracy(predictions, X, y):
    m,n = X.shape
    temp = predictions - y
    return (100 * ((np.count_nonzero(temp==0)) / m))


## Setup the parameters
image_height = 28
image_width = 28
input_layer_size  = image_height * image_width
hidden_layer_size = image_height * 2
num_hidden_layers = 2
num_labels = 10
epsilon = 0.12
lambda_reg = 0.5
iter_grad = 10000
alpha = 0.5
precentage_for_cv = 20
layer_size = [0] * (num_hidden_layers + 2)
layer_size[0] = input_layer_size
layer_size[num_hidden_layers + 1] = num_labels
for i in range(1,num_hidden_layers+1):
    layer_size[i] = hidden_layer_size

## Read the trainig set
train = pd.read_csv('train_digit.csv')
y = train[["label"]]
y = np.asarray(y.astype(np.float32).values)
X = np.zeros((train.shape[0],input_layer_size))
for i in range(0,input_layer_size):
    pixel = "pixel%d" % (i)
    temp = train[[pixel]]
    temp = np.asarray(temp.astype(np.float32).values)
    X[:,i:i+1] = temp

## Read the test set
test = pd.read_csv('test_digit.csv')
X_test = np.zeros((test.shape[0],input_layer_size))
for i in range(0,input_layer_size):
    pixel = "pixel%d" % (i)
    temp = test[[pixel]]
    temp = np.asarray(temp.astype(np.float32).values)
    X_test[:,i:i+1] = temp

## Inialize the parameters
initial_nn_params = np.zeros((0,0))
for i in range(0,num_hidden_layers + 1):
    initial_theta = randInitializeWeights(layer_size[i], layer_size[i+1])
    initial_nn_params = (np.append(initial_nn_params, initial_theta.reshape(-1,1))).reshape(-1,1)

#X_cv = X[:int(train.shape[0]/precentage_for_cv), :]
#y_cv = y[:int(train.shape[0]/precentage_for_cv)]
#X = X[int(train.shape[0]/precentage_for_cv):, :]
#y = y[int(train.shape[0]/precentage_for_cv):]

## Train the network
#nn_params = gradientDescent(initial_nn_params, X, y)
nn_params = gradientDescent(initial_nn_params, X, y, False)

## Check accuaracy
# pred = predict(nn_params, X)
# print("Training Set Accuracy: ", calculateAccuracy(pred, X, y))
pred = predict(nn_params, X)
print("Training Set Accuracy: ", calculateAccuracy(pred, X, y))
#pred = predict(nn_params, X_cv)
#print("Cross validation Accuracy: ", calculateAccuracy(pred, X_cv, y_cv))

## Predict
pred = predict(nn_params, X_test)

## Write result to csv file
pred = np.hstack((1+(np.asarray(list(range(pred.shape[0]))).reshape(X_test.shape[0],1)).astype(np.int32), pred.astype(np.int32)))
res = pd.DataFrame(pred, columns=['ImageId','Label'])
res.to_csv('res_digit.csv', index=False)
