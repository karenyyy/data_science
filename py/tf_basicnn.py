import matplotlib
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# n_features = 10
# n_dense_neurons = 3
#
# x = tf.placeholder(tf.float32, (None, n_features))
#
# W = tf.Variable(tf.random_normal([n_features, n_dense_neurons]))
# b = tf.Variable(tf.ones([n_dense_neurons]))
#
# xW = tf.matmul(x, W)
# z = tf.add(xW, b)
#
# activation = tf.sigmoid(z)
#
# init = tf.global_variables_initializer()
#
# with tf.Session() as sess:
#     sess.run(init)
#     layout = sess.run(activation, feed_dict={x: np.random.random([1, n_features])})
#
# print layout

x_data = np.linspace(0.0, 10.0, 1000)
noise = np.random.randn(len(x_data))


y = 0.5 * x_data + noise

x_df = pd.DataFrame(data=x_data, columns=["x data"])
y_df = pd.DataFrame(data=y, columns=["y data"])

my_data = pd.concat([x_df, y_df], axis=1)
print(my_data.head())

# my_data.plot(kind='scatter', x="x data", y="y data")
# plt.show()

batch_size = 8

m = tf.Variable(0.81)
b = tf.Variable(0.17)

x_ph = tf.placeholder(tf.float32, [batch_size])  # feed in 8 each time
y_ph = tf.placeholder(tf.float32, [batch_size])

y_model = tf.multiply(m, x_ph) + b

loss = tf.reduce_mean(tf.square(y_ph - y_model))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.05)
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    batches = 100
    for i in range(batches):
        rand_index = np.random.randint(len(x_data), size=batch_size)  # 100 data per batch
        feed = {x_ph: x_data[rand_index], y_ph: y[rand_index]}
        sess.run(train, feed_dict=feed)  # [train function, feed_dict]

    estimated_m, estimated_b = sess.run([m,b])
print(estimated_m, estimated_b)  # estimated m and b

y_hat=x_data*estimated_m+estimated_b

my_data.plot(kind='scatter', x="x data", y="y data")
plt.plot(x_data, y_hat)
#plt.plot(x_data, y_hat, 'r')
plt.show()