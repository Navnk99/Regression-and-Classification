import numpy as np
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from LinearRegression import LinearRegression

# Load the Iris dataset
iris = load_iris()
x = iris.data[:, [0, -1]]  # Use sepal length and petal width as input features
y = iris.data[:, 1]   # Predict sepal width


# Train the regression model
mod = LinearRegression()
mod.fit(x, y, batch_size=32, regularization=0.1, max_epochs=100, patience=3)

# Save the model parameters
mod.save('model4_params.pkl')

# Plot the loss
plt.plot(mod.loss_history)
plt.xlabel('Step')
plt.ylabel('Loss')
plt.title('Model 4 Training Loss')
plt.savefig('model4_loss.png')
plt.show()
