import matplotlib
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split

x_data = np.linspace(0, 10, 10000)
y_data = x_data * 5 + np.random.randn(len(x_data))
feat_cols = [tf.feature_column.numeric_column("x", shape=[1])]
estimator = tf.estimator.LinearRegressor(feature_columns=feat_cols)

x_train, x_eval, y_train, y_eval = train_test_split(x_data, y_data, test_size=0.3, random_state=2018)

print(x_train.shape)

input_func = tf.estimator.inputs.numpy_input_fn({'x': x_train},
                                                y=y_train,
                                                batch_size=8,
                                                num_epochs=None,
                                                shuffle=True)
train_func = tf.estimator.inputs.numpy_input_fn({'x': x_train},
                                                y_train,
                                                batch_size=8,
                                                num_epochs=1000,
                                                shuffle=False)
eval_func = tf.estimator.inputs.numpy_input_fn({'x': x_eval},
                                               y_eval,
                                               batch_size=8,
                                               num_epochs=1,
                                               shuffle=False)
estimator.train(input_fn=input_func, steps=1000)

train_metric = estimator.evaluate(input_fn=train_func, steps=1000)
eval_metric = estimator.evaluate(input_fn=eval_func, steps=1000)
print(train_metric)

new_data = np.linspace(0, 10, 4)
predict_func = tf.estimator.inputs.numpy_input_fn({'x': new_data},
                                                  shuffle=False)

estimated = estimator.predict(input_fn=predict_func)
y_hat = []
for i in estimated:
    y_hat.append(i["predictions"])  # mark: i["predictions"]

plt.plot(x_data, y_data, 'b')
plt.plot(new_data, y_hat, "r*")
plt.show()


