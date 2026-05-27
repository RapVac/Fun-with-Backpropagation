'''
A small-scale proof of concept.
The goal is to identify when there are 3 objects in a row on a 3x3 grid.
'''

import numpy as np
from math import e
import random

number_of_layers=3

rng = np.random.default_rng()

## Input
test_input=np.array([[1],
                    [0],
                    [0],
                    [0],
                    [1],
                    [0],
                    [0],
                    [0],
                    [1]])

L1_weights=rng.random((6, 9))

L2_activations=np.matmul(L1_weights, test_input)
L2_weights=rng.random((4, 6))

L3_activations=np.matmul(L2_weights, L2_activations)
L3_weights=rng.random((2, 4))

output=np.array((2, 1))

Layers=[[],
        [test_input, L1_weights],
        [L2_activations, L2_weights],
        [L3_activations, L3_weights],
        [output]]


def sigmoid(x):
    return 1/(1+e**(-x))

def dsigmoid(x):
    return e**(-x)/(1+e**(-x))**2

## input_vector: (1, 9) -> (1, 2)
def process(input_vector):
    Layers[1][0] = input_vector
    
    L1_result=np.matmul(L1_weights, input_vector)
    L1_result=sigmoid(L1_result)
    Layers[2][0] = L1_result
        
    L2_result=np.matmul(L2_weights, L1_result)
    L2_result=sigmoid(L2_result)
    Layers[3][0] = L2_result
    
    L3_result=np.matmul(L3_weights, L2_result)
    L3_result=sigmoid(L3_result)
    Layers[4][0]=L3_result
    
    return L3_result

def cost(input_vector, expected):
    cost=0
    for o, e in zip(input_vector, expected):
        cost+=(o-e)**2
    return cost

def delta(i, expected_vector):
    if i==number_of_layers:
        return 2*( np.multiply(sigmoid( np.matmul(Layers[i][1], Layers[i][0]) ) - expected_vector,
                               dsigmoid( np.matmul(Layers[i][1], Layers[i][0]))) )

    return np.multiply(np.matmul(np.transpose(Layers[i+1][1]),
                                 delta(i+1, expected_vector)),
                       dsigmoid(np.matmul(Layers[i][1],
                                          Layers[i][0])))


data=open("3InARow_training_data", "r").read().strip().split("\n")
data=[d.split(";") for d in data]
random.shuffle(data)

def string_to_vector(s):
    l=list(s)
    l=[int(l) for l in l]
    return np.transpose([np.array(l)])

def test(d=None):
    if not d:
        d=random.choice(data)
    print(d[0][0:3].replace("1", "X").replace("0", "_"))
    print(d[0][3:6].replace("1", "X").replace("0", "_"))
    print(d[0][6:9].replace("1", "X").replace("0", "_"))
    
    print(f"Correct answer is: {d[1]}")
    res=process(string_to_vector(d[0]))
    res=np.transpose(res)[0]
    if res[0] > res[1]:
        r="Y"
    else:
        r="N"
    print(f"Model says: {r} ({res})")
    return r==d[1]

net_d3=net_d2=net_d1=0
for h in range(0, 12000):
    random.shuffle(data)
    L3_weights+=net_d3/512
    L2_weights+=net_d2/512
    L1_weights+=net_d1/512
    net_d3=net_d2=net_d1=0
    
    net_d3=net_d2=net_d1=0
    
    for i in range(0, len(data)):
        result=data[i][1]
        input_vector=string_to_vector(data[i][0])
        if result=="Y":
            expected_result=np.array([[1],
                                      [0]])
        else:
            expected_result=np.array([[0],
                                      [1]])

        process(input_vector)

        d3=delta(3, expected_result)
        d2=delta(2, expected_result)
        d1=delta(1, expected_result)

        d3=np.matmul(d3, np.transpose(Layers[3][0]))
        d2=np.matmul(d2, np.transpose(Layers[2][0]))
        d1=np.matmul(d1, np.transpose(Layers[1][0]))

##        d3*=-1
##        d2*=-1
##        d1*=-1
        net_d1-=d1
        net_d2-=d2
        net_d3-=d3

correct=0
total=len(data)
for x in data:
    if test(x):
        correct+=1
print(correct/total)
