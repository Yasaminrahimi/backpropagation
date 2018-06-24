import numpy as np
import matplotlib.pyplot as plt






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
    
    graph = np.zeros((1000,2), dtype=float)
    plot = np.zeros((31,2), dtype=float)
    for i in range (8,31):
        e = 0
        weight1 =np.random.uniform(low=-0.5, high=0.5, size=(64,i))
        weight2 =np.random.uniform(low=-0.5, high=0.5, size=(i+1,63))
        while True :
            for l in range(10):
                x = X[l]
                algorithm (x, weight1, weight2, learn_rate)
            tol = 0
            tolerance = []
            value = []
            for j in range(10):
                layer1 = np.c_[np.ones(1), [X[j]]]
                layer2 = np.c_[np.ones(1), vecmoid(layer1.dot(weight1))]
                layer3 = sigmoid(layer2.dot(weight2))
                value.append(layer3)
                tolerance = np.subtract (value[j],X[j])
                for k in range (63):
                    if  -0.1 <= tolerance[0][k] <= 0.1:
                        tol += 1
            if tol == 630:
                break
            dis = np.subtract(value[0], X[0])
            error = np.inner(dis , dis)
            e += 1
            graph[e,0] = e
            graph[e,1] = error
            k+=1
            plot [i,1] = e
        plot [i,0] = i
        print (e)
        print (i)
        print (plot)
        plt.plot(graph[:,0:], graph[:,1:], 'ro' )
        plt.axis([0, 100, 0, 20])
        plt.show()
    plt.plot(plot[:,0:], plot[:,1:], 'ro' )
    plt.axis([0, 31, 0, 200])
    plt.show()
    print (plot)
    

def sigmoid(y):
    y_out = 1.0 / (1.0 + np.exp(-y))
    return y_out

vecmoid = np.vectorize(sigmoid)


train(X)




