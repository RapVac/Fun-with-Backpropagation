import numpy as np
import random
from math import e
from PIL import Image
from time import time
import os

rng = np.random.default_rng()

test_input=np.ones((4225, 1))

Layers=[[],
        [test_input, rng.random((80, 4225))],
        [np.array((80, 1)), rng.random((80, 80))],
        [np.array((80, 1)), rng.random((6, 80))],
        [np.array((6, 1))]]

def s(x):
    y=np.abs(x).max()
    if y!=0:
        x=x/y
    #return 1/(1+e**(-x))
    return np.where(x > 0, x, x * 0.01)
    
def ds(x):
    #return e**(-x)/(1+e**(-x))**2
    return np.where(x > 0, 1, 0.01)

def process(input_vector, i=1):
    if i==1:
        Layers[1][0] = input_vector

    if i==len(Layers)-1:
        return 0

    Layers[i+1][0] = s(np.matmul(Layers[i][1],
                                 Layers[i][0]))

    return process(input_vector, i+1)

def image_to_input_vector(file_name):
    f=Image.open(file_name, mode="r")
    f=f.convert("L")
    f=np.array(f)
    f=f.reshape((4225, 1))
    f=f/255
    return f

def delta(i, expected_vector):
    if i==len(Layers)-2:
        return 2*( np.multiply( s(np.matmul(Layers[i][1],
                                            Layers[i][0])) - expected_vector,
                                ds(np.matmul(Layers[i][1],
                                             Layers[i][0]))))

    return np.multiply( np.matmul(np.transpose(Layers[i+1][1]),
                                  delta(i+1, expected_vector)),
                        ds(np.matmul(Layers[i][1], Layers[i][0])))

shapes=["Cube", "Pyramid", "Tetrahedron", "Sphere", "Cylinder", "Cone"]

outputs={"Cube":np.transpose(np.array([[1, 0, 0, 0, 0, 0]])),
      "Pyramid":np.transpose(np.array([[0, 1, 0, 0, 0, 0]])),
      "Tetrahedron":np.transpose(np.array([[0, 0, 1, 0, 0, 0]])),
      "Sphere":np.transpose(np.array([[0, 0, 0, 1, 0, 0]])),
      "Cylinder":np.transpose(np.array([[0, 0, 0, 0, 1, 0]])),
      "Cone":np.transpose(np.array([[0, 0, 0, 0, 0, 1]]))}

files=[]

for item in os.listdir("./training_images"):
    files.append(("./training_images/"+item, item.split("_")[0]))

net_d1=net_d2=net_d3=0
c=0
T=time()
random.shuffle(files)
print(T)
for x in range(0, 8000):
    for f in files:
        if c==250:
            Layers[1][1]+=net_d1/250
            Layers[2][1]+=net_d2/250
            Layers[3][1]+=net_d3/250
            
            net_d1=net_d2=net_d3=0
            c=0

        vector=image_to_input_vector(f[0])
        expected=outputs[f[1]]

        process(vector)

        d1=delta(1, expected)
        d2=delta(2, expected)
        d3=delta(3, expected)

        d1=np.matmul(d1, np.transpose(Layers[1][0]))
        d2=np.matmul(d2, np.transpose(Layers[2][0]))
        d3=np.matmul(d3, np.transpose(Layers[3][0]))
        
        net_d1-=d1
        net_d2-=d2
        net_d3-=d3
        c+=1
        
    random.shuffle(files)
    print(x, time()-T)
    T=time()

print(time())
def test(file_name, r=True):
    if r:
        name="./testing_images/"+file_name+"_"+str(random.randint(0, 14))+".png"
    else:
        name=file_name
    t=image_to_input_vector(name)
    print(name)
    process(t)
    print(Layers[-1][0])
