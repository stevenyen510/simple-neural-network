import numpy as np
from math import exp
import matplotlib.pyplot as plt

inputList = []
one_hot_output = []

zero_set = []
one_set = []

f = open("training_data.txt",'r')
t_line = f.readline()
while t_line:

    t_line = t_line.strip("\n") #remove \n if it exists
    line_as_arr = t_line.split(" ") #return array of 3 strings, split on space

    print(line_as_arr)
    inputList.append([float(line_as_arr[0]),float(line_as_arr[1])])
    if(line_as_arr[2]=='0'):
        one_hot_output.append([1,0])
        zero_set.append([float(line_as_arr[0]), float(line_as_arr[1])])
    elif(line_as_arr[2]=='1'):
        one_hot_output.append([0,1])
        one_set.append([float(line_as_arr[0]), float(line_as_arr[1])])
    t_line = f.readline()

f.close()

zero_set = np.array(zero_set)
one_set = np.array(one_set)
plt.scatter(zero_set[:, 0], zero_set[:, 1], c='r', marker='o')
plt.scatter(one_set[:, 0], one_set[:, 1], c='b', marker='x')
plt.xlabel('x1')
plt.ylabel('x2')
plt.show()
