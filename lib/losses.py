import numpy as np
#mean squared error to caclulate loss
class MSE:
    def forward(self, y_pred, y_true):
        self.y_pred = y_pred
        self.y_true = y_true
        return np.mean((y_pred - y_true) ** 2)

    def backward(self, y_pred, y_true):
        batch_size = y_pred.shape[0]
        return (2 / batch_size) * (y_pred - y_true)
class BCE:
    def forward(self, y_pred, y_true):
        eps = 1e-8
        y_pred = np.clip(y_pred, eps, 1 - eps)
        self.y_pred = y_pred
        self.y_true = y_true
        return -np.mean(
            y_true * np.log(y_pred) +
            (1 - y_true) * np.log(1 - y_pred)
        )

    def backward(self, y_pred, y_true):
        eps = 1e-8
        y_pred = np.clip(y_pred, eps, 1 - eps)
        return (y_pred - y_true) / (y_pred * (1 - y_pred) * y_true.shape[0])