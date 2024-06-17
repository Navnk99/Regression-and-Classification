import numpy as np
import pickle


class LinearRegression:
    def __init__(self):
        self.weights = None
        self.bias = None
        self.loss_history = []
        self.weight = []

    def fit(self, x, y, batch_size=32, regularization=0, max_epochs=100, patience=3):
        # Split data into training and validation sets
        Val_size = int(0.1 * x.shape[0])
        x_val, y_val = x[:Val_size], y[:Val_size]
        x_train, y_train = x[Val_size:], y[Val_size:]

        # Initialize weights and bias
        self.weights = np.random.randn(x_train.shape[1])
        self.bias = np.random.randn()

        best_loss = float('inf')
        no_advancement = 0

        for epoch in range(max_epochs):
            # Shuffle training data
            self.loss_history.append(self.score(x_val, y_val))
            indices = np.arange(x_train.shape[0])
            np.random.shuffle(indices)
            x_train = x_train[indices]
            y_train = y_train[indices]

            # Perform batch gradient descent
            for i in range(0, x_train.shape[0], batch_size):
                x_group = x_train[i:i + batch_size]
                y_group = y_train[i:i + batch_size]

                # Compute gradients
                y_pred = self.predict(x_group)
                error = y_pred - y_group
                gradient = np.dot(x_group.T, error) / batch_size

                # Update weights and bias
                self.weights -= gradient + regularization * self.weights
                self.bias -= np.mean(error)

            self.weight.append(self.weights)
            # Calculate validation loss
            Val_loss = self.score(x_val, y_val)

            # Check for early stopping
            if Val_loss < best_loss:
                best_loss = Val_loss
                no_advancement = 0
            else:
                no_advancement += 1
                if no_advancement == patience:
                    break

    def predict(self, x):
        return np.dot(x, self.weights) + self.bias

    def score(self, x, y):
        y_pred = self.predict(x)
        return np.mean((y_pred - y) ** 2)

    def save(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump((self.weights, self.bias), f)

    def load(self, filepath):
        with open(filepath, 'rb') as f:
            self.weights, self.bias = pickle.load(f)
