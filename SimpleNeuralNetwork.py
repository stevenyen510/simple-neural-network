"""
SIMPLE NEURAL NETWORK
Steven Yen, 2018

Construct NN with 2-node input layer, 5-node hidden layer, and 2-node output layer
Train the NN to classify data points greater/less than a sine function.
The training data sine.txt consists of 2000 data points lebeled either '0' or '1'
"""

import numpy as np
import math

#import data from text file

inputList = []
one_hot_output = []

f = open("training_data.txt",'r')
t_line = f.readline()

while t_line:

    t_line = t_line.strip("\n") #remove \n if it exists
    line_as_arr = t_line.split(" ") #return array of 3 strings, split on space

    #print(line_as_arr)
    inputList.append([float(line_as_arr[0]),float(line_as_arr[1])])
    if(line_as_arr[2]=='0'):
        one_hot_output.append([1,0])
    elif(line_as_arr[2]=='1'):
        one_hot_output.append([0,1])
    t_line = f.readline()

f.close()

#logistic sigmoid activation function h()
def sigActivation(xi):
    return (2/(1+math.exp(-xi**2)))-1
    #return np.tanh(xi)

#derivative of logistic sigmoid activation hprime()
def derivative(xi):
    return 4*xi*math.exp(-xi**2)/(1+math.exp(-xi**2))**2
    #return 1 - np.tanh(xi) ** 2

#vectorize activation function so it can be applied to vectors
h = np.vectorize(sigActivation)
hprime =  np.vectorize(derivative)

def softMax(xi):
    den = math.exp(xi[0]) + math.exp(xi[1])
    return [math.exp(xi[0]) / den, math.exp(xi[1]) / den]

#learning parameters
lrate = 0.001 #leaerning rate
maxIteration = 2000
errorThreshold = 0.2 #theta

#Generate random starting weight matrix W (aka W1)
#W = np.random.uniform(low=-0.5, high=0.5, size=(5,2))
#hard-coded generated matrix so output is reproducible for comparison
W = np.array([[ 0.34468784, -0.35377386],
 [ 0.15598414,  0.37063321],
 [-0.06614087, -0.0313791 ],
 [ 0.35304239, -0.3518681 ],
 [-0.39688351,  0.23887037]])

#Generate random starting weight matrix W2
#W2 = np.random.uniform(low=-0.5, high=0.5, size=(2,5))
#hard-coded generated matrix so output is reproducible for comparison
W2 = np.array([[-0.33896502,  0.23030052,  0.09597328, -0.09398411, -0.0798749 ],
 [ 0.18766971, -0.32885567, -0.48833292,  0.23299937,  0.1530219]])

#Additional global variables used
actualErrorMag = 10 #error magnitue used for stopping (compared against threshold)
errorOverTime = [] #stores the error to be graphed later
inputArr = np.array(inputList) #convert list to np.array
one_hot_output_arr = np.array(one_hot_output) #convert list to np.array

"""
Performs forward-Backward pass for one sample point (x1,x2) and y
Performs the forward pass to calcualte error gradient
Then performs back-propagation with the error gradient
Finally, update the weight matrix W and W2 (global variables)
"""
def applyModel(Xin,Yj):

    global W #global so it updates the one copy of weight matrix
    global W2 #global so it updates the one copy of weight matrix
    global errorOverTime #used to track error so it can be plot at the end

    X = Xin.T #transpose to
    prod1 = np.matmul(W,X) #apply weight matrix to input: WX
    Z1 = h(prod1) #apply activation to the linear combination h(WX)
    L1hprime = hprime(prod1) #calculate and store derivative h'(WX)

    prod2 = np.matmul(W2, Z1.T) #apply weight matrix 2: W2(h'(WX))
    Z2 = prod2 #Z2 is just prod2 because we don't apply logistic sigmoid activation here

    normalizedOutput = softMax(Z2) #apply the soft max to the result of output layer

    #calculate the error gradient, which is simply the difference between
    #the softmax and the one hot vector (this is derivative of cross entropy loss)

    errorGrad =[]
    if(Yj[0]==1): #label '0', one_hot [1,0]
        errorGrad = [normalizedOutput[0]-1,normalizedOutput[1]]
    else: #label '1', one_hot [0,1]
        errorGrad = [normalizedOutput[0],normalizedOutput[1]-1]

    #multiply transpose of W2 with error gradient to get errGrad2
    #errGrad2 is the top error gradient of Weight layer 2
    errGrad2 = np.matmul(W2.T, errorGrad)
    deltaW2 = np.outer(errorGrad, Z1) #weight updates for W2

    #perform element-wise multiplication
    #errGrad1 is the top error gradient of weight layer 1
    errGrad1 = np.multiply(errGrad2, L1hprime) #element-wise multiply
    errGrad3 = np.matmul(W.T, errGrad1.T)
    deltaW1 = np.outer(errGrad1, X)

    #update the weight matrices by adding delta*learning rate
    W2 = np.add(W2, -lrate * deltaW2)
    W = np.add(W, -lrate * deltaW1)

    #Calculae the L2 norm of the error gradient to check against stoping criteria
    global actualErrorMag #made this error so it can be tacked across method calls.
    actualErrorMag= (errorGrad[0] ** 2 + errorGrad[1] ** 2) ** 0.5 #L2 norm

    #Append the error to an array so it can be plotted at the end to check
    errorOverTime.append(abs(errorGrad[0]))

    #calculate loss function (cross-entropy)
    Lce = -(math.log(normalizedOutput[0])*Yj[0]+math.log(normalizedOutput[1])*Yj[1])

    #print x1, x2, o1(x), o2(x), abs(errorGradient[0]), Lce
    #note errorGrad[0] and errorGrad[1] are equal, just have opposite signs.
    print(np.array([Xin[0],Xin[1],normalizedOutput[0],normalizedOutput[1],
                    errorGrad[0],errorGrad[1],Lce]))

print("{input}  {softmax}  {error gradient}   {loss function}")
print("[x1 x2  o1(x),o2(x) errGrad[0],errGrad[1]   Lce      ]")

"""
This loop constitues the training. It trains until the stopping criteria is
reached either when (1) maxIteration is reached or
(2) when the stoping criteria actualErrorMag<errorThresdhold is true
"""
for j in range(maxIteration):
    applyModel(inputArr[j],one_hot_output_arr[j])
    if(actualErrorMag<errorThreshold): #stopping criteria
        break

print("Final Trained NN Weights:")
print("W",W)
print("W2",W2)
