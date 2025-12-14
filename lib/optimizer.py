import numpy as np
class SGD:
  #stochastic gradient descent
  #sets learning rate for gradient descent
    def __init__(self, lr=0.5):
        self.lr = lr

#only updates parameters in dense layers , checks if layer has attribute of update
    def step(self, layers):
        for layer in layers:
            if hasattr(layer, "update"):
                layer.update(self.lr)
class Adam:
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = {}
        self.v = {}
        self.t = 0

    def step(self, layers):
        self.t += 1
        for i, layer in enumerate(layers):
            if hasattr(layer, 'W'):
                if i not in self.m:
                    self.m[i] = np.zeros_like(layer.W)
                    self.v[i] = np.zeros_like(layer.W)

                self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * layer.dW
                self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (layer.dW ** 2)

                m_hat = self.m[i] / (1 - self.beta1 ** self.t)
                v_hat = self.v[i] / (1 - self.beta2 ** self.t)

                layer.W -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
                layer.b -= self.lr * layer.db