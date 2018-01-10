import matplotlib
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


# print(mnist.train.num_examples, mnist.test.num_examples)
# print(mnist.train.images.shape)
# print(mnist.train.images[1])
#
# plt.imshow(mnist.train.images[1].reshape(28,28), cmap="gist_gray")
# plt.show()



# X = tf.placeholder(tf.float32, shape=[None, 784])
# W = tf.Variable(tf.zeros([784, 10]))
# b = tf.Variable(tf.zeros([10]))
#
# y_estimated = tf.add(tf.matmul(X, W), b)
#
# y_true = tf.placeholder(tf.float32, [None, 10])  # cal the prob for each digit given a handwritten digit pic
#
# cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_estimated, labels=y_true))
#
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.05)
# train = optimizer.minimize(cross_entropy)
#
# init = tf.global_variables_initializer()
#
# batch_size = 100
#
# with tf.Session() as sess:
#     sess.run(init)
#     for step in range(1000):
#         batch_x, batch_y = mnist.train.next_batch(batch_size)
#         sess.run(train, feed_dict={X: batch_x, y_true: batch_y})
#     correct_estimation = tf.equal(tf.argmax(y_estimated, axis=1), tf.argmax(y_true, axis=1))
#     # [True, False, True, False .....]
#
#     accuracy = tf.reduce_mean(tf.cast(correct_estimation, tf.float32))
#
#     result = sess.run(accuracy, feed_dict={X: mnist.test.images, y_true: mnist.test.labels})
#
#     print(result)


def init_weights(shape):
    init_weights = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(init_weights)


def init_bias(shape):
    init_bias = tf.constant(0.1, shape)
    return tf.Variable(init_bias)


def conv2d(x, W):
    # x --> [batch, H, W, Channels]
    # W --> [filter H, filter W, Channels input, Channels OUT]
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding="SAME")


def max_pool_2by2(x):
    # x --> [batch, h, w, c]
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")


def convolutional_layer(input_x, shape):
    W = init_weights(shape)
    b = init_bias([shape[3]])
    return tf.nn.relu(conv2d(input_x, W) + b)


def normal_full_layer(input_layer, size):
    input_size = int(input_layer.get_shape()[1])
    W = init_weights([input_size, size])
    b = init_bias([size])
    return tf.matmul(input_layer, W) + b


x = tf.placeholder(tf.float32, shape=[None, 784])
y_true = tf.placeholder(tf.float32, shape=[None, 10])

x_image = tf.reshape(x, [-1, 28, 28, 1])

convo_1 = convolutional_layer(x_image, shape=[5, 5, 1, 32])

convo_1_pooling = max_pool_2by2(convo_1)

convo_2 = convolutional_layer(convo_1_pooling, shape=[5, 5, 32, 64])

convo_2_pooling = max_pool_2by2(convo_2)

convo_2_flat = tf.reshape(convo_2_pooling, [-1, 7 * 7 * 64])
full_layer_one = tf.nn.relu(normal_full_layer(convo_2_flat, 1024))

# dropout
hold_prob = tf.placeholder(tf.float32)
full_one_dropout = tf.nn.dropout(full_layer_one, keep_prob=hold_prob)

y_pred = normal_full_layer(full_one_dropout, 10)

# loss function
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_pred,
                                                                       labels=y_true))

# optimizer
optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
train = optimizer.minimize(cross_entropy)

# initialize
init = tf.global_variables_initializer()

steps = 5000

with tf.Session() as sess:
    sess.run(init)

    for i in range(steps):
        batch_x, batch_y = mnist.train.next_batch(50)
        sess.run(train, feed_dict={x: batch_x,
                                   y_true: batch_y,
                                   hold_prob: 0.5})

        if i % 100 == 0:
            matches = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_true, 1))
            accuracy = tf.reduce_mean(tf.cast(matches, tf.float32))
            print("accuracy:{}".format(accuracy))

            print(sess.run(accuracy, feed_dict={x: mnist.test.images,
                                                y_true: mnist.test.labels,
                                                hold_prob: 1.0}))

