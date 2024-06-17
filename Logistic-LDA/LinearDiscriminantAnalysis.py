import numpy as np

class LinearDiscriminantAnalysis:
    def __init__(self):
        self.weights = None
        self.bias = None

    def fit(self, x, y):
        num_samp, num_fts = x.shape

        # Compute class priors
        cls_pr = np.zeros(np.max(y) + 1)
        for i in range(len(cls_pr)):
            cls_pr[i] = np.sum(y == i) / num_samp

        # Compute class means
        cls_mn = np.zeros((np.max(y) + 1, num_fts))
        for i in range(len(cls_mn)):
            cls_mn[i] = np.mean(x[y == i], axis=0)

        # Compute covariance matrix
        cov_mtx = np.cov(x.T)

        # Compute weights and bias
        inv_cov_mtx = np.linalg.inv(cov_mtx)
        self.weights = np.dot(inv_cov_mtx, cls_mn.T).T
        self.bias = -0.5 * np.sum(np.dot(cls_mn, inv_cov_mtx) * cls_mn, axis=1) + np.log(cls_pr)

    def predict(self, x):
        linear_mod = np.dot(x, self.weights.T) + self.bias
        y_pred_cls = np.argmax(linear_mod, axis=1)
        return y_pred_cls
