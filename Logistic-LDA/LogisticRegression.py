import numpy as np

class LogisticRegression:
    def __init__(self, learning_rate=0.01, num_iterations=100):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.bias = None

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, x, y):
        num_samp, num_fts = x.shape

        # Initialize weights and bias
        self.weights = np.zeros(num_fts)
        self.bias = 0

        # Gradient descent
        for _ in range(self.num_iterations):
            linear_mod = np.dot(x, self.weights) + self.bias
            y_pred = self._sigmoid(linear_mod)

            # Compute gradients
            da = (1 / num_samp) * np.dot(x.T, (y_pred - y))
            db = (1 / num_samp) * np.sum(y_pred - y)

            # Update weights and bias
            self.weights -= self.learning_rate * da
            self.bias -= self.learning_rate * db

    def predict(self, x):
        linear_mod = np.dot(x, self.weights) + self.bias
        y_pred = self._sigmoid(linear_mod)
        y_pred_cls = np.round(y_pred).astype(int)
        return y_pred_cls
