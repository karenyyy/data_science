CIFAR_DIR = 'cifar-10-batches-py/'

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


dirs = ['batches.meta', 'data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5', 'test_batch']

all_data = [0, 1, 2, 3, 4, 5, 6]


def unpickle(file):
    import pickle
    with open(file, 'rb') as f:
        cifar_dict = pickle.load(f, encoding="bytes")
    return cifar_dict


for i, direc in zip(all_data, dirs):
    all_data[i] = unpickle(CIFAR_DIR + direc)

batch_meta = all_data[0]
data_batch_1 = all_data[1]
data_batch_2 = all_data[2]
data_batch_3 = all_data[3]
data_batch_4 = all_data[4]
data_batch_5 = all_data[5]
test_batch = all_data[6]


def one_hot_encode(vec, class_count):
    out = np.zeros((len(vec), class_count))
    out[range(len(vec)), vec] = 1
    return out

print(batch_meta)
#plt.imshow(test_batch[b"data"])
#plt.show()

#print(test_batch[b'labels'])


#
# print(one_hot_encode([1,2,3,4,5,6,7,8,9,0],10))


class CifarHelper():
    def __init__(self):
        self.start=0

        self.all_training_batches = [data_batch_1, data_batch_2, data_batch_3, data_batch_4, data_batch_5]

        self.test_batch = [test_batch]

        self.training_images = None
        self.training_labels = None

        self.testing_images = None
        self.testing_labels = None

    def image_preprocessing(self):
        ## training
        self.training_images = np.vstack([i[b'data'] for i in self.all_training_batches])
        training_size = len(self.training_images)

        self.training_images = self.training_images.reshape(training_size, 3, 32, 32).transpose(0, 2, 3, 1) / 255
        # (sample_size, 32, 32, 3)
        self.training_labels = one_hot_encode(np.hstack([i[b'labels'] for i in self.all_training_batches]), class_count=10)

        ## testing
        self.testing_images = np.vstack([d[b"data"] for d in self.test_batch])
        testing_size = len(self.testing_images)

        self.testing_images = self.testing_images.reshape(testing_size, 3, 32, 32).transpose(0, 2, 3, 1) / 255
        # (sample_size, 32, 32, 3)
        self.testing_labels = one_hot_encode(np.hstack([i[b'labels'] for i in self.test_batch]), class_count=10)

    def next_batch(self, batch_size):
        x=self.training_images[self.start:self.start+batch_size].reshape(batch_size, 32, 32, 3)
        y=self.training_labels[self.start:self.start+batch_size]

        training_size=len(self.training_images)
        self.start=(self.start+batch_size)%training_size
        # return (batch_training_images, batch_training_labels)
        return x, y



cifar=CifarHelper()
cifar.image_preprocessing()


###########################################
## create model
###########################################

x=tf.placeholder(tf.float32, shape=[None, 32, 32, 3])
y_true=tf.placeholder(tf.float32,shape=[None, 10])

hold_prob=tf.placeholder(tf.float32)


def init_weights(shape):
    init_weights = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(init_weights)


def init_bias(shape):
    init_bias = tf.constant(0.1, shape=shape)
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



convo_1 = convolutional_layer(x, shape=[4, 4, 3, 32])

convo_1_pooling = max_pool_2by2(convo_1)

convo_2 = convolutional_layer(convo_1_pooling, shape=[4, 4, 32, 64])

convo_2_pooling = max_pool_2by2(convo_2)

convo_2_flat = tf.reshape(convo_2_pooling, [-1, 8 * 8 * 64])
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
        batch_x, batch_y = cifar.next_batch(50)
        sess.run(train, feed_dict={x: batch_x,
                                   y_true: batch_y,
                                   hold_prob: 0.5})

        if i % 100 == 0:
            matches = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_true, 1))
            accuracy = tf.reduce_mean(tf.cast(matches, tf.float32))

            print(sess.run(accuracy, feed_dict={x: cifar.testing_images,
                                                y_true: cifar.testing_labels,
                                                hold_prob: 1.0}))
