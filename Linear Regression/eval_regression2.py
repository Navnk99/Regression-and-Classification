import numpy as np
from sklearn.datasets import load_iris
from LinearRegression import LinearRegression

# Load the Iris dataset
iris = load_iris()
x_test = iris.data[:, 2:]  # Use petal length and petal width as input features
y_test = iris.data[:, 0]   # Ground truth sepal length

# Load the model parameters
mod = LinearRegression()
mod.load('model2_params.pkl')

# Evaluate the model on the test set
MSE = mod.score(x_test, y_test)
print(f"Mean Squared Error: {MSE}")
