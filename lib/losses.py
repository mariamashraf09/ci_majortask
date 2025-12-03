import numpy as np
#mean squared error to caclulate loss
class MSE:
  #Loss= (1/(2*n)) sum(y - y^)^2)
    def forward(self, y_true, y_pred):
        return np.mean((y_true - y_pred)**2)

    def backward(self, y_true, y_pred):
        return 2 * (y_pred - y_true) / y_true.size
