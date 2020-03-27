import numpy as np

# lets plot a sigmoid function

input = np.linspace(-10, 10, 100)

def sigmoid(x):
    return 1/(1+np.exp(-x))

from matplotlib import pyplot as plt
plt.plot(input, sigmoid(input), c="r")

# start the model.here we have an input layer and a output layer.input layer have 3 features and 5 training examples

x = np.array([[0,1,0],[0,0,1],[1,0,0],[1,1,0],[1,1,1]])
y=np.array([[1,0,0,1,1]])
y=y.reshape(1,5)
x =np.transpose(x)
w = np.random.randn(3,1)   #initialise weights
print(w)
b = np.random.rand(1)  # initialize bias term
lr=0.05   #learning rate

#defining the sigmoid function

def sigmoid(z):
  return 1/(1+np.exp(z))
def der_sigmoid(z):
  return sigmoid(z)(1-sigmoid(z)
  
W=np.transpose(w)
print(W.shape)    #print at every step for matrix dimensions and better visualization
print(x.shape)
print(b.shape)

#trainig process

m=5 #trainig examples
for i in range(100):
  z = np.dot(W,x) + b
  a = sigmoid(z)
  dz = a-y
  print(dz)
  dw = np.sum(np.dot(x,np.transpose(dz)))
  dw = dw/m
  db = np.sum(dz)
  db = db/m

  W = W-lr*dw  #modify weights and bias term
  b = b-lr*db

#testing

single_point = np.array([1,0,0])

result = sigmoid(np.dot(W,single_point) + b)
print(result)


