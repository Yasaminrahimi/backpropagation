import numpy as np
import matplotlib.pyplot as plt



weight1 =np.random.uniform(low=-0.5, high=0.5, size=(64,16))
weight2 =np.random.uniform(low=-0.5, high=0.5, size=(17,63))


X = []

A1 = [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1]

B1 = [1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0]

C1 = [0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0]

D1 = [1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0]

E1 = [1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]

F1 = [1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

G1 = [0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]

H1 = [1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]

I1 = [0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]

J1 = [0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0]
X.append(A1)
X.append(B1)
X.append(C1)
X.append(D1)
X.append(E1)
X.append(F1)
X.append(G1)
X.append(H1)
X.append(I1)
X.append(J1)
plot = np.zeros((30,2), dtype=float)
learn_rate = 0.2


def algorithm (x, weight1, weight2, learn_rate):
    value = []
    layer1 = np.c_[np.ones(1), [x]]
    layer2 = np.c_[np.ones(1), vecmoid(layer1.dot(weight1))]
    layer3 = sigmoid(layer2.dot(weight2))
    delta3 = x - layer3
    delta2 = np.multiply(delta3.dot(weight2.T), np.multiply(layer2, (1-layer2)))
    delta2 = delta2[:,1:]
    weight2 += learn_rate * (np.dot(layer2.T, delta3)) 
    weight1 += learn_rate * (np.dot(layer1.T, delta2))
    

def train(X):
    k=0
    graph = np.zeros((100,2), dtype=float)
    while True :
        for i in range(10):
            x = X[i]
            algorithm (x, weight1, weight2, learn_rate)
        if test(X) == 630:
            break
        graph[k,0] = k
        graph[k,1] = test(X)
        k+=1
    plot [i,0] = k
    plot [i,1] = i
    print (k)
    plt.plot(graph[:,0:], graph[:,1:], 'ro' )
    plt.axis([0, 100, 0, 10])
    plt.show()
    

def sigmoid(y):
    y_out = 1.0 / (1.0 + np.exp(-y))
    return y_out

vecmoid = np.vectorize(sigmoid)

def test(X):
    tol = 0
    tolerance = []
    value = []
    for i in range(10):
        layer1 = np.c_[np.ones(1), [X[i]]]
        layer2 = np.c_[np.ones(1), vecmoid(layer1.dot(weight1))]
        layer3 = sigmoid(layer2.dot(weight2))
        value.append(layer3)
        tolerance = np.subtract (value[i],X[i])
        for i in range (63):
            if  -0.2 <= tolerance[0][i] <= 0.2:
                tol += 1
    if tol == 630 :
        return tol
        
    dis = np.subtract(value[0], X[0])
    error = np.inner(dis , dis)
    return error
train(X)
"""plt.plot(plot[:,0:], plot[:,1:], 'ro' )
plt.axis([0, 30, 0, 800])
plt.show()"""



