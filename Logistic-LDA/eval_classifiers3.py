from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from LogisticRegression import LogisticRegression
from LinearDiscriminantAnalysis import LinearDiscriminantAnalysis
from plot_decision_regions import plot_decision_boundary

# Load the Iris dataset
iris = load_iris()
x = iris.data              # Use all features as input
y = iris.target            # Target variable

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, stratify=y, random_state=42)

# Train the logistic regression model
log_reg = LogisticRegression()
log_reg.fit(x_train, y_train)

# Train the linear discriminant analysis model
lda = LinearDiscriminantAnalysis()
lda.fit(x_train, y_train)

# Predict on the test set
y_pred_lr = log_reg.predict(x_test)
y_pred_lda = lda.predict(x_test)

# Calculate the accuracy
acc_lr = accuracy_score(y_test, y_pred_lr)
acc_lda = accuracy_score(y_test, y_pred_lda)

print(f"Logistic Regression Accuracy: {acc_lr}")
print(f"Linear Discriminant Analysis Accuracy: {acc_lda}")
