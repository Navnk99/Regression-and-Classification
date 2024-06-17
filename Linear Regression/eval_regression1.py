import numpy as np
from sklearn.datasets import load_iris
from LinearRegression import LinearRegression

# Load the Iris dataset
iris = load_iris()
x_test = iris.data[:, :2]  # Use sepal length and sepal width as input features
y_test = iris.data[:, 3]   # Ground truth petal width

# Load the model parameters
mod = LinearRegression()
mod.load('model1_params.pkl')

# Evaluate the model on the test set
MSE = mod.score(x_test, y_test)
print(f"Mean Squared Error: {MSE}")
