import numpy as np
#section for activation fns


class Sigmoid:
    def forward(self, x):
        self.out = 1 / (1 + np.exp(-x))
        return self.out
#i need to save sigmoid value to use it while calculating its gradient
#dL/dx = dL/dy * (gradient of sigmoid)
    def backward(self, grad):
        return grad * (self.out * (1 - self.out))
#tanh(x)=(e^x - e^-x)/(e^x + e^-x)
class Tanh:
    def forward(self, x):
        self.out = np.tanh(x)
        return self.out
#derivative of tanh(x)= 1- (tanh(x)^2)
    def backward(self, grad):
        return grad * (1 - self.out**2)


 #we won't use this activation function in solving xor as it only outputs positive value, we need non linear activation functions in solving xor like tanh and sigmoid
class ReLU: #rectified linear unit
  def forward(self,x):
    self.output = np.maximum(0,x)