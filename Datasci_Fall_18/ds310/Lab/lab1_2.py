import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import mean_squared_error, roc_curve, auc



SEED = 1234
SPLIT_RATIO = 0.2

X, y = load_breast_cancer(return_X_y=True)

class Classify(object):
    def __init__(self):
        self.seed = SEED
        self.split_ratio = SPLIT_RATIO
        self.classifier = None

    def simple_split(self):
        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            test_size=self.split_ratio,
                                                            random_state=self.seed)
        return X_train, X_test, y_train, y_test

    def fit(self, X_train, y_train):
        self.classifier.fit(X_train, y_train)


