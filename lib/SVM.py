import numpy as np
class LinearSVM_SGD:
    def __init__(self, lr=0.001, lambda_reg=0.01, epochs=50):
        self.lr = lr
        self.lambda_reg = lambda_reg
        self.epochs = epochs

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0.0

        for epoch in range(self.epochs):
            for i in range(n_samples):
                margin = y[i] * (np.dot(X[i], self.w) + self.b)

                if margin < 1:
                    self.w -= self.lr * (self.lambda_reg * self.w - y[i] * X[i])
                    self.b += self.lr * y[i]
                else:
                    self.w -= self.lr * self.lambda_reg * self.w

    def decision_function(self, X):
        return X @ self.w + self.b

    def predict(self, X):
        return np.sign(self.decision_function(X))