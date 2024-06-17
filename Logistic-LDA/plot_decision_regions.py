import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions

def plot_decision_boundary(model, X, y, feature_names):
    # Plot decision boundary
    plot_decision_regions(X, y, clf=model, feature_index=[0, 1], filler_feature_values={2: 0, 3: 0},
                          filler_feature_ranges={2: 1, 3: 1})

    # Add labels and title
    plt.xlabel(feature_names[0])
    plt.ylabel(feature_names[1])
    plt.title('Decision Boundary')

    # Show the plot
    plt.show()
