import numpy as np
from math import exp
import matplotlib.pyplot as plt

inputList = []
one_hot_output = []

r_set = []
b_set = []

f = open("Assignment1Data.txt",'r')
t_line = f.readline()
while t_line:

    t_line = t_line.strip("\n") #remove \n if it exists
    line_as_arr = t_line.split(" ") #return array of 3 strings, split on space

    print(line_as_arr)
    inputList.append([float(line_as_arr[0]),float(line_as_arr[1])])
    if(line_as_arr[2]=='0'):
        one_hot_output.append([1,0])
        r_set.append([float(line_as_arr[0]),float(line_as_arr[1])])
    elif(line_as_arr[2]=='1'):
        one_hot_output.append([0,1])
        b_set.append([float(line_as_arr[0]),float(line_as_arr[1])])
    t_line = f.readline()

f.close()

r_set = np.array(r_set)
b_set = np.array(b_set)
plt.scatter(r_set[:,0], r_set[:,1],c='r')
plt.scatter(b_set[:,0], b_set[:,1],c='b')
plt.show()
