import numpy as np
#Layer Abstraction section
class Layer:
    def forward(self, x):
        pass

    def backward(self, grad):
        pass

    def update(self, lr):
        pass
#dense layer has inputs from neurons and outputs to other neurons, only layer with parameters (w)
class Dense(Layer):
  #in_features is the size of a single sample
  #out_features is the no. of neurons
    def __init__(self, in_features, out_features):
      #random values for weights, randn gives normal distribution (mean 0, std 1)
      #the weights are reversed so that we don't need to perform transpose on the weights matrix
        self.W = np.random.randn(in_features, out_features) * 1.0
        #biases are initialized to zero
        self.b = np.zeros((1, out_features))
        self.x = None
        self.dW = None
        self.db = None

    def forward(self, x):
      #Save this input x inside the layer object so backward() can use it later
        self.x = x
        #it multiples x by weights then adds bias, y=xw+b
        return np.dot(x, self.W) + self.b


    def backward(self, grad_out):
        # Gradients
        #dL/dW = x T * dL/dy
        self.dW = self.x.T @ grad_out
        #sum of output gradients
        self.db = np.sum(grad_out, axis=0, keepdims=True)
        #dL/dX = dL/dy * W T
        grad_input = grad_out @ self.W.T
        return grad_input

    def update(self, lr):
      #lr is the learning rate
        self.W -= lr * self.dW
        self.b -= lr * self.db
