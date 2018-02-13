"""
Simple Neural Network
Apply logistic sigmoid only to hidden layer, and apply only softmax to output layer.
"""

import numpy as np
from math import exp

inputList = []
one_hot_output = []

f = open("Assignment1Data.txt",'r')
t_line = f.readline()

while t_line:

    t_line = t_line.strip("\n") #remove \n if it exists
    line_as_arr = t_line.split(" ") #return array of 3 strings, split on space

    print(line_as_arr)
    inputList.append([float(line_as_arr[0]),float(line_as_arr[1])])
    if(line_as_arr[2]=='0'):
        one_hot_output.append([1,0])
    elif(line_as_arr[2]=='1'):
        one_hot_output.append([0,1])
    t_line = f.readline()

f.close()

# W = np.random.rand(5,2)
W = np.array([[ 0.34468784, -0.35377386],
 [ 0.15598414,  0.37063321],
 [-0.06614087, -0.0313791 ],
 [ 0.35304239, -0.3518681 ],
 [-0.39688351,  0.23887037]])

# Wp = np.random.rand(2,5)
W2 = np.array([[-0.33896502,  0.23030052,  0.09597328, -0.09398411, -0.0798749 ],
 [ 0.18766971, -0.32885567, -0.48833292,  0.23299937,  0.1530219]])

#parameters for stoping criteria
lrate = 0.1 #leaerning rate
maxIteration = 1000
errorThreshold = 0.2 #theta
actualErrorMag = 10 #value of err magnitue used for stopping

def sigActivation(xi):
    return (2/(1+exp(-xi**2)))-1

def derivative(xi):
    return 4*xi*exp(-xi**2)/(1+exp(-xi**2))**2

h = np.vectorize(sigActivation)
hprime =  np.vectorize(derivative)

print("W",W)
print("W2", W2)
print("\ninputArr\n",inputList)
print("\none_hot_output",one_hot_output)
print("\n\n")

inputArr = np.array(inputList)
one_hot_output_arr = np.array(one_hot_output)

def softMax(xi):
    den = exp(xi[0]) + exp(xi[1])
    return [exp(xi[0]) / den, exp(xi[1]) / den]

errorOverTime = [] #stores the error to be graphed later

"""
Performs forward-Backward pass for one sample point (x1,x2) and y
Updates the weight matrix W and W2 (global variables) in the end
"""
def applyModel(Xin,Yj):

    global W
    global W2
    global errorOverTime

    X = Xin.T
    prod1 = np.matmul(W,X)
    Z1 = h(prod1)
    L1hprime = hprime(prod1)

    prod2 = np.matmul(W2, Z1.T)

    Z2 = prod2 #no outputlayer activation o() in this case

    normalizedOutput = softMax(Z2)

    errorGrad =[]

    if(Yj[0]==1): #[1,0] : 0
        errorGrad = [normalizedOutput[0]-1,normalizedOutput[1]]
    else:
        errorGrad = [normalizedOutput[0],normalizedOutput[1]-1]

    errGrad2 = np.matmul(W2.T, errorGrad)
    deltaW2 = np.outer(errorGrad, Z1)

    errGrad1 = np.multiply(errGrad2, L1hprime) #element-wise multiply
    eerrGrad3 = np.matmul(W.T, errGrad1.T)
    deltaW1 = np.outer(errGrad1, X)

    W2 = np.add(W2, -lrate * deltaW2)
    W = np.add(W, -lrate * deltaW1)

    global actualErrorMag
    actualErrorMag= (errorGrad[0] ** 2 + errorGrad[1] ** 2) ** 0.5

    errorOverTime.append(abs(errorGrad[0]))

    print(np.array([Xin[0],Xin[1],normalizedOutput[0],normalizedOutput[1],errorGrad[0],errorGrad[1]]))

print("\nApply the model until stopping criteria is reached\n")

"""
This loop constitues the training. It trains until maxIteration is
reached or when the stoping criteria actualErrorMag<errorThresdhold is true
"""
for j in range(maxIteration):
    applyModel(inputArr[j],one_hot_output_arr[j])
    if(actualErrorMag<errorThreshold):
        break

#print the error over time to see if it converges to 0
import matplotlib.pyplot as plt
xval = range(1,100*100)
toPrint = zip(xval,errorOverTime)
r_set = np.array(toPrint)
plt.scatter(r_set[:,0], r_set[:,1],c='r')
plt.show()