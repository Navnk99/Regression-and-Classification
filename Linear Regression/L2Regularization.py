import numpy as np
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from LinearRegression import LinearRegression

# Load the Iris dataset
iris = load_iris()
x = iris.data[:, :2]  # Use sepal length and sepal width as input features
y = iris.data[:, 3]   # Predict petal width
features = iris.feature_names

# Train the regression model
mod = LinearRegression()
mod.fit(x, y, batch_size=32, regularization=0.1, max_epochs=100, patience=3)

# Save the model parameters
mod.save('model1l2_params.pkl')

def error_func(pred, result):
    return np.square(np.subtract(result, pred)).mean()

random_sets = [(0, 1), (0, 2), (0, 3), (1, 0), (1, 2), (1, 3), (2, 0), (2, 1), (2, 3), (3, 0), (3, 1), (3, 2)]
reg = [None] * len(random_sets)
reg = mod.weight
#Picking one of the combinations above and implementing l2 regularization on it to inspects the weights. 

w = np.random.uniform(-1, 1, size=(2,))
plot5 = plt.figure(figsize=[8,5])

arr = [1,0]
#gradient descent
def gradient_des_l2 (w,result, data, reg_l2):
    reg_l2_w = (np.linalg.inv(reg_l2 * np.eye(data.shape[1]) + data.T @ data) @ (data.T @ result))
    return reg_l2_w

data = np.stack((np.ones(len(x)), x[:, arr[0]]), axis=1)
result = x[:, arr[1]]
y_hat = data @ w
list_reg_l2 = []

plot5.suptitle('Visualization of linear regression on l2 regularization', fontsize=16)
print()
for a in range(0, np.shape(result)[0], 32):
    list_reg_l2.append((result[a: a + 32],
                           data[a: a + 32]))
#plotting the points and comparing both cases
for b in range(100):
     for e in range(len(list_reg_l2)):
            val_bt = list_reg_l2[e][1]
            res_bt = list_reg_l2[e][0]
            w = gradient_des_l2(w, res_bt, val_bt,3)
            y_hat = data @ w
            
for e in range(2):
    ax = plot5.add_subplot(1,2,e+1)
    ax.set_xlabel(features[arr[0]])
    ax.set_ylabel(features[arr[1]])
    
    ax.scatter(x[:, arr[0]], x[:, arr[1]],color='blue')
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()

    if e==0:
        ax.axline([0, w[0]], slope=w[1], c='b')

    else:
        ax.axline([0, reg[1][0]], slope=reg[1][1], c='b')

        
    ax.set_xlim([xmin, xmax])
    ax.set_ylim([ymin, ymax])

    
#error and weights with l2 regulaization and without regulaization    
error = error_func(y_hat, result)
plt.show()
print(" Normal Error: ",error_func(data@reg[1],result),"\n Normal weights", reg[1])
print()
print("L2 Regulaization error: ",error,"\n L2 regression weights", w)