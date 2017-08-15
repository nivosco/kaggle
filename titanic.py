'''
August 2017
@author: Niv Vosco
'''

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

def gradientDescent3(theta, X, y):
    m,n = X.shape
    a1 = np.hstack((np.ones((X.shape[0], 1)), X))
    delta4 = np.zeros((m, num_labels))
    print("Start: gradient descent")
    for k in range(0, iter_grad):
        ptr = 0
        if (k % (iter_grad / 10) == 0):
            print("Iteration %d" % k)
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
        for i in range(0,m):
            for j in range(0, num_labels):
                if (y.item(i,0) == j):
                    delta4[i,j] = a4[i,j] - 1
                else:
                    delta4[i,j] = a4[i,j]

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
    print("Iteration %d" % iter_grad)
    print("Done: gradient descent")
    return theta

def gradientDescent(theta, X, y):
    m,n = X.shape
    print("Start: gradient descent")
    for k in range(0, iter_grad):
        if (k % (iter_grad / 10) == 0):
            print("Iteration %d" % k)
        deltal = np.zeros((m, num_labels))
        h = forwardPropogation(theta, X)
        for i in range(0,m):
            for j in range(0, num_labels):
                if (y.item(i,0) == j):
                    deltal[i,j] = h[i,j] - 1
                else:
                    deltal[i,j] = h[i,j]
        [a,z] = getParametersOfLayer(theta, num_hidden_layers, X)
        theta_grad = (np.matmul(a.T, deltal) / m).T
        theta_grad_t = (np.delete(theta_grad.T, 0, 0)).T
        theta_grad_t = np.hstack((np.zeros((theta_grad_t.shape[0], 1)), theta_grad_t))
        theta_grad = theta_grad + (lambda_reg / m) * theta_grad_t
        ptr_last = theta.shape[0]
        ptr_first = 0
        for i in range(0,num_hidden_layers):
            ptr_first = ptr_first + (layer_size[i+1] * (layer_size[i] + 1))
        for i in range(1,num_hidden_layers+1):
            thetal = (theta[ptr_first:ptr_last]).reshape(layer_size[num_hidden_layers + 2 - i], layer_size[num_hidden_layers + 1 - i] + 1)
            ptr_last = ptr_first
            ptr_first = 0
            for j in range(0,num_hidden_layers-i):
                ptr_first = ptr_first + (layer_size[j+1] * (layer_size[j] + 1))
            [a,z] = getParametersOfLayer(theta, num_hidden_layers - i, X)
            z = np.hstack((np.ones((z.shape[0], 1)), z))
            delta = np.multiply(np.matmul(deltal, thetal), (np.multiply(sigmoid(z), (1 - sigmoid(z)))))
            delta = (np.delete(delta.T, 0, 0)).T
            theta_grad_t = (np.matmul(a.T, delta) / m).T
            theta_grad_tt = (np.delete(theta_grad_t.T, 0, 0)).T
            theta_grad_tt = np.hstack((np.zeros((theta_grad_tt.shape[0], 1)), theta_grad_tt))
            theta_grad_t = theta_grad_t + (lambda_reg / m) * theta_grad_tt
            theta_grad = (np.append(theta_grad_t.reshape(-1,1), theta_grad.reshape(-1,1))).reshape(-1,1)
            deltal = delta
        theta = theta - (alpha * theta_grad)
    print("Iteration %d" % iter_grad)
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
base_features = 7
extra_features = 2
input_layer_size  = base_features + extra_features
hidden_layer_size = 20
num_hidden_layers = 2
num_labels = 2
epsilon = 0.12
lambda_reg = 2
iter_grad = 100000
alpha = 1
layer_size = [0] * (num_hidden_layers + 2)
layer_size[0] = input_layer_size
layer_size[num_hidden_layers + 1] = num_labels
for i in range(1,num_hidden_layers+1):
    layer_size[i] = hidden_layer_size

## Read the trainig set
train = pd.read_csv('train_titanic.csv')
train = train.replace(["male","female"],[0,1])
train = train.replace(['S','C','Q'],[0,1,2])
train = train.fillna(0)
X_t = train[["Pclass", "Sex","Age","SibSp","Parch","Fare","Embarked"]]
y = train[["Survived"]]
X = np.zeros((train.shape[0],input_layer_size))
X[:,:base_features] = np.asarray(X_t.astype(np.float32).values)
y = np.asarray(y.astype(np.float32).values)


## Read the test set
test = pd.read_csv('test_titanic.csv')
test = test.replace(["male","female"],[0,1])
test = test.replace(['S','C','Q'],[0,1,2])
test = test.fillna(0)
X_test_t = test[["Pclass", "Sex","Age","SibSp","Parch","Fare","Embarked"]]
X_test = np.zeros((test.shape[0],input_layer_size))
X_test[:,:base_features] = np.asarray(X_test_t.astype(np.float32).values)


## Add ticket number
# ticket_train = np.zeros((train.shape[0]))
# ticket_test = np.zeros((test.shape[0]))
# ticket_str_train = train["Ticket"].apply(str)
# ticket_str_test = test["Ticket"].apply(str)
# for i in range(0,train.shape[0]):
    # try:
        # ticket_train[i] = [int(s) for s in ticket_str_train[i].split() if s.isdigit()][0]
    # except:
        # ticket_train[i] = 0

# for i in range(0,test.shape[0]):
    # try:
        # ticket_test[i] = [int(s) for s in ticket_str_test[i].split() if s.isdigit()][0]
    # except:
        # ticket_test[i] = 0

# X[:,input_layer_size-extra_features] = ticket_train
# X_test[:,input_layer_size-extra_features] = ticket_test


## Add cabin variable
# cabin_train = np.zeros((train.shape[0]))
# cabin_test = np.zeros((test.shape[0]))
# cabin_str_train = train["Cabin"].apply(str)
# cabin_str_test = test["Cabin"].apply(str)
# for i in range(0,train.shape[0]):
    # try:
        # if cabin_str_train[i][0] == 'A':
            # cabin_train[i] = 1000 + int(cabin_str_train[i][1:])
        # elif cabin_str_train[i][0] == 'B':
            # cabin_train[i] = 2000 + int(cabin_str_train[i][1:])
        # elif cabin_str_train[i][0] == 'C':
            # cabin_train[i] = 3000 + int(cabin_str_train[i][1:])
        # elif cabin_str_train[i][0] == 'D':
            # cabin_train[i] = 4000 + int(cabin_str_train[i][1:])
        # elif cabin_str_train[i][0] == 'E':
            # cabin_train[i] = 5000 + int(cabin_str_train[i][1:])
        # elif cabin_str_train[i][0] == 'F':
            # cabin_train[i] = 6000 + int(cabin_str_train[i][1:])
        # else:
            # cabin_train[i] = 0
    # except:
        # cabin_train[i] = 0

# for i in range(0,test.shape[0]):
    # try:
        # if cabin_str_test[i][0] == 'A':
            # cabin_test[i] = 1000 + int(cabin_str_test[i][1:])
        # elif cabin_str_test[i][0] == 'B':
            # cabin_test[i] = 2000 + int(cabin_str_test[i][1:])
        # elif cabin_str_test[i][0] == 'C':
            # cabin_test[i] = 3000 + int(cabin_str_test[i][1:])
        # elif cabin_str_test[i][0] == 'D':
            # cabin_test[i] = 4000 + int(cabin_str_test[i][1:])
        # elif cabin_str_test[i][0] == 'E':
            # cabin_test[i] = 5000 + int(cabin_str_test[i][1:])
        # elif cabin_str_test[i][0] == 'F':
            # cabin_test[i] = 6000 + int(cabin_str_test[i][1:])
        # else:
            # cabin_test[i] = 0
    # except:
        # cabin_test[i] = 0

# X[:,input_layer_size-extra_features] = cabin_train
# X_test[:,input_layer_size-extra_features] = cabin_test


## Add Age variable
age = 0
count = 0
for i in range(0,test.shape[0]):
    if (X_test[i][2]>0):
        age += X_test[i][2]
        count += 1
for i in range(0,train.shape[0]):
    if (X[i][2]>0):
        age += X[i][2]
        count += 1
age = age / count
for i in range(0,test.shape[0]):
    if (X_test[i][2]>0):
        X_test[i][2] = age
for i in range(0,train.shape[0]):
    if (X[i][2]>0):
        X[i][2] = age


## Add last name unique variable
name_train = train["Name"].apply(str)
initial_train = np.zeros((name_train.size))
name_test = test["Name"].apply(str)
initial_test = np.zeros((name_test.size))
last_names = ["" for i in range(0,name_train.size+name_test.size)]
for i in range(0,name_train.size):
    name_i = name_train[i]
    try:
        last_name_i = name_i[2+name_i.index('.'):]
        last_names[i] = last_name_i
        if (name_i[name_i.index('.')-4:name_i.index('.')]=='Miss' or name_i[name_i.index('.')-4:name_i.index('.')]=='Mlle' or name_i[name_i.index('.')-2:name_i.index('.')]=='Ms'):
            initial_train[i] = 1
        elif (name_i[name_i.index('.')-3:name_i.index('.')]=='Mme' or name_i[name_i.index('.')-3:name_i.index('.')]=='Mrs'):
            initial_train[i] = 2
        elif (name_i[name_i.index('.')-2:name_i.index('.')]=='Mr'):
            initial_train[i] = 3
        elif (name_i[name_i.index('.')-6:name_i.index('.')]=='Master'):
            initial_train[i] = 4
    except:
        last_names[i] = name_train[i]
        initial_train[i] = 0

for i in range(0,name_test.size):
    name_i = name_test[i]
    try:
        last_name_i = name_i[2+name_i.index('.'):]
        last_names[name_test.size+i] = last_name_i
        if (name_i[name_i.index('.')-4:name_i.index('.')]=='Miss' or name_i[name_i.index('.')-4:name_i.index('.')]=='Mlle' or name_i[name_i.index('.')-2:name_i.index('.')]=='Ms'):
            initial_test[i] = 1
        elif (name_i[name_i.index('.')-3:name_i.index('.')]=='Mme' or name_i[name_i.index('.')-3:name_i.index('.')]=='Mrs'):
            initial_test[i] = 2
        elif (name_i[name_i.index('.')-2:name_i.index('.')]=='Mr'):
            initial_test[i] = 3
        elif (name_i[name_i.index('.')-6:name_i.index('.')]=='Master'):
            initial_test[i] = 4
    except:
        last_names[name_train.size+i] = name_test[i]
        initial_test[i] = 0

last_name_acc = np.zeros((name_train.size+name_test.size))
for i in range(0,name_train.size+name_test.size):
    last_name_acc[i] = last_names.count(last_names[i])

last_name_acc = np.asarray(last_name_acc.astype(np.float32))
last_name_train = last_name_acc[:name_train.size]
last_name_test = last_name_acc[name_train.size:]
X[:,input_layer_size-extra_features] = last_name_train
X[:,input_layer_size-extra_features+1] = initial_train
X_test[:,input_layer_size-extra_features] = last_name_test
X_test[:,input_layer_size-extra_features+1] = initial_test


## Feature scaling
X_vec = np.append(X.reshape(-1,1),X_test.reshape(-1,1)).reshape(X.shape[0] + X_test.shape[0],input_layer_size)
X_vec = (X_vec - np.mean(X_vec, axis=0)) / (np.amax(X_vec,axis=0) - np.amin(X_vec,axis=0))
X = X_vec[:X.shape[0],:]
X_test = X_vec[X.shape[0]:,:]


## Inialize the parameters
initial_nn_params = np.zeros((0,0))
for i in range(0,num_hidden_layers + 1):
    initial_theta = randInitializeWeights(layer_size[i], layer_size[i+1])
    initial_nn_params = (np.append(initial_nn_params, initial_theta.reshape(-1,1))).reshape(-1,1)


# X_cv = X[train.shape[0]-200:, :]
# y_cv = y[train.shape[0]-200:]
# X = X[:train.shape[0]-200, :]
# y = y[:train.shape[0]-200]

## Train the network
#nn_params = gradientDescent(initial_nn_params, X, y)
nn_params2 = gradientDescent3(initial_nn_params, X, y)


## Check accuaracy
# pred = predict(nn_params, X)
# print("\nTraining Set Accuracy: ", calculateAccuracy(pred, X, y))
pred = predict(nn_params2, X)
print("\nTraining Set Accuracy: ", calculateAccuracy(pred, X, y))
# pred = predict(nn_params2, X_cv)
# print("\nTraining Set Accuracy: ", calculateAccuracy(pred, X_cv, y_cv))


## Predict
pred = predict(nn_params2, X_test)


## Write result to csv file
pred = np.hstack(((1+y.shape[0]+np.asarray(list(range(pred.shape[0]))).reshape(X_test.shape[0],1)).astype(np.int32), pred.astype(np.int32)))
res = pd.DataFrame(pred, columns=['PassengerId','Survived'])
res.to_csv('res_titanic.csv', index=False)
