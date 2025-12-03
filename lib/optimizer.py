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