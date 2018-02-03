from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import matplotlib.pyplot as plt


class GaussianFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, N, width_factor=2.0):
        self.N=N
        self.stddev_factor=width_factor

    def gaussian_basis(self, x, y, stddev, axis= None):
        arg = (x - y) / stddev
        return np.exp(-0.5 * np.sum(arg ** 2, axis))

    def fit(self, X, y): # make it private
        self.mu=np.linspace(X.min(), X.max(), self.N)
        self.stddev=self.stddev_factor*(self.mu[1]-self.mu[0])
        return self

    def transform(self, X):
        return self.gaussian_basis(X[:,:, np.newaxis], self.mu, self.stddev, axis=1)

x=10*np.random.RandomState(2018).rand(50)
y=np.sin(x)+0.1*np.random.RandomState(2018).randn(50) # add white noise

x_test = np.linspace(0, 10, 1000)

gauss_model = make_pipeline(GaussianFeatures(10, 1.0),
                            LinearRegression())

gauss_model.fit(x[:, np.newaxis], y)


y_estimated=gauss_model.predict(x_test[:, np.newaxis])

gf=gauss_model.named_steps['gaussianfeatures']
lr=gauss_model.named_steps['linearregression']

fig, ax=plt.subplots(figsize=(16,8))

for i in range(10):
    encoder = np.zeros(10)
    encoder[i] = 1
    X_test = gf.transform(x_test[:, None]) * encoder
    Y_estimated=lr.predict(X_test)
    ax.fill_between(x_test, y_estimated.min(), y_estimated, color='gray', alpha=0.2)

ax.scatter(x,y)
ax.plot(x_test,y_estimated)
ax.set_xlim(0,10)
ax.set_ylim(y_estimated.min(), 1.5)

plt.show()