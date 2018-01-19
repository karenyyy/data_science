import tensorflow as tf
import os
from urllib.request import urlretrieve
import gzip
import numpy as np

SOURCE_URL = 'http://yann.lecun.com/exdb/mnist/'
WORK_DIRECTORY = 'MNIST_data'
IMAGE_SIZE = 28
NUM_CHANNELS = 1
PIXEL_DEPTH = 255
NUM_LABELS = 10

VALIDATION_SIZE = 5000  # Size of the validation set.
SEED = 2018  # Set to None for random seed.
BATCH_SIZE = 64
NUM_EPOCHS = 10
EVAL_BATCH_SIZE = 64
EVAL_FREQUENCY = 100  # Number of steps between evaluations.


def check_if_downloaded(filename):
    if not tf.gfile.Exists(WORK_DIRECTORY):
        tf.gfile.MakeDirs(WORK_DIRECTORY)
    filepath = os.path.join(WORK_DIRECTORY, filename)

    if not tf.gfile.Exists(filepath):
        filepath, _ = urlretrieve(SOURCE_URL + filename, filepath)
    return filepath


def extract_data(filename, num_images):
    with gzip.open(filename) as bytestream:
        bytestream.read(16)
        buffer = bytestream.read(IMAGE_SIZE * IMAGE_SIZE * num_images * NUM_CHANNELS)
        images = np.frombuffer(buffer, dtype=np.uint8).astype(np.float32)
        images = (images - (PIXEL_DEPTH / 2.0)) / PIXEL_DEPTH
        images = images.reshape(num_images, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS)
        return images


def extract_labels(filename, num_images):
    with gzip.open(filename) as bytestream:
        bytestream.read(8)
        buf = bytestream.read(1 * num_images)
        labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
    return labels


def accuracy(predictions, labels):
    return 100.0 * np.sum(np.argmax(predictions, 1) == labels) / predictions.shape[0]


if __name__ == '__main__':
    train_data_filename = check_if_downloaded('train-images-idx3-ubyte.gz')
    train_labels_filename = check_if_downloaded('train-labels-idx1-ubyte.gz')
    test_data_filename = check_if_downloaded('t10k-images-idx3-ubyte.gz')
    test_labels_filename = check_if_downloaded('t10k-labels-idx1-ubyte.gz')

    train_data = extract_data(train_data_filename, 60000)
    train_labels = extract_labels(train_labels_filename, 60000)
    test_data = extract_data(test_data_filename, 10000)
    test_labels = extract_labels(test_labels_filename, 10000)

    # generate a validation set
    validation_data = train_data[:VALIDATION_SIZE, ...]
    validation_labels = train_labels[:VALIDATION_SIZE]
    train_data = train_data[VALIDATION_SIZE:, ...]
    train_labels = train_labels[VALIDATION_SIZE:]
    num_epochs = NUM_EPOCHS
    train_size = train_labels.shape[0]

    train_data_node = tf.placeholder(
        tf.float32,
        shape=(BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))
    train_labels_node = tf.placeholder(tf.int64, shape=(BATCH_SIZE,))
    eval_data = tf.placeholder(
        tf.float32,
        shape=(EVAL_BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))


    conv1_weights=tf.Variable(tf.truncated_normal([5,5,NUM_CHANNELS, 32],
                                                  stddev=0.1,
                                                  seed=SEED,
                                                  dtype=tf.float32))

    conv1_biases=tf.Variable(tf.zeros([32], dtype=tf.float32))

    conv2_weights=tf.Variable(tf.truncated_normal([5,5,32,64],
                                                  stddev=0.1,
                                                  seed=SEED,
                                                  dtype=tf.float32))

    conv2_biases = tf.Variable(tf.constant(0.1, shape=[64], dtype=tf.float32))

    fc1_weights=tf.Variable(tf.truncated_normal([IMAGE_SIZE//4*IMAGE_SIZE//4*64, 512],
                                                stddev=0.1,
                                                seed=SEED,
                                                dtype=tf.float32))
    fc1_biases=tf.Variable(tf.constant(0.1,
                                       shape=[512],
                                       dtype=tf.float32))

    fc2_weights = tf.Variable(tf.truncated_normal([512, NUM_LABELS],
                                                  stddev=0.1,
                                                  seed=SEED,
                                                  dtype=tf.float32))
    fc2_biases = tf.Variable(tf.constant(
        0.1, shape=[NUM_LABELS], dtype=tf.float32))


    def model(data, train=False):
        conv = tf.nn.conv2d(data,
                            conv1_weights,
                            strides=[1, 1, 1, 1],
                            padding='SAME')

        relu=tf.nn.relu(tf.nn.bias_add(conv, conv1_biases))

        pool = tf.nn.max_pool(relu,
                              ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1],
                              padding='SAME')

        conv = tf.nn.conv2d(pool,
                            conv2_weights,
                            strides=[1, 1, 1, 1],
                            padding='SAME')
        relu = tf.nn.relu(tf.nn.bias_add(conv, conv2_biases))
        pool = tf.nn.max_pool(relu,
                              ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1],
                              padding='SAME')

        pool_shape = pool.get_shape().as_list()
        reshape = tf.reshape(
            pool,
            [pool_shape[0], pool_shape[1] * pool_shape[2] * pool_shape[3]])


        normal_fc=tf.nn.relu(tf.matmul(reshape, fc1_weights) + fc1_biases)
        if train:
            normal_fc = tf.nn.dropout(normal_fc, 0.5, seed=SEED)
        return tf.matmul(normal_fc, fc2_weights) + fc2_biases


    logits = model(train_data_node, True)
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=train_labels_node, logits=logits))

    regularizers = (tf.nn.l2_loss(fc1_weights) + tf.nn.l2_loss(fc1_biases) +
                    tf.nn.l2_loss(fc2_weights) + tf.nn.l2_loss(fc2_biases))
    # Add the regularization term to the loss to prevent overfit
    loss += 5e-4 * regularizers

    batch=tf.Variable(0, tf.float32)

    # make learning rate decay per batch
    learning_rate=tf.train.exponential_decay(
        0.01,    # base learning rate
        batch*BATCH_SIZE,
        train_size,
        0.95,   # decay step
        staircase=True
    )

    # use simple momentum for optimization
    optimizer=tf.train.MomentumOptimizer(
        learning_rate=learning_rate,
        momentum=0.9).minimize(loss=loss, global_step=batch)

    train_prediction=tf.nn.softmax(logits)

    eval_prediction = tf.nn.softmax(model(eval_data))


    def eval_in_batches(data, sess):
        """Get all predictions for a dataset by running it in small batches."""
        size = data.shape[0]
        predictions = np.ndarray(shape=(size, NUM_LABELS), dtype=np.float32)
        for begin in range(0, size, EVAL_BATCH_SIZE):
            end = begin + EVAL_BATCH_SIZE
            if end <= size:
                predictions[begin:end, :] = sess.run(
                    eval_prediction,
                    feed_dict={eval_data: data[begin:end, ...]})
            else:
                batch_predictions = sess.run(
                    eval_prediction,
                    feed_dict={eval_data: data[-EVAL_BATCH_SIZE:, ...]})
                predictions[begin:, :] = batch_predictions[begin - size:, :]
        return predictions


    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        print('Initialized!')
        # Loop through training steps.
        for step in range(int(num_epochs * train_size) // BATCH_SIZE):
            offset = (step * BATCH_SIZE) % (train_size - BATCH_SIZE)
            batch_data = train_data[offset:(offset + BATCH_SIZE), ...]
            batch_labels = train_labels[offset:(offset + BATCH_SIZE)]

            feed_dict = {train_data_node: batch_data,
                         train_labels_node: batch_labels}

            sess.run(optimizer, feed_dict=feed_dict)

            if step % EVAL_FREQUENCY == 0:
                l, lr, predictions = sess.run([loss, learning_rate, train_prediction],
                                              feed_dict=feed_dict)
                print('Step %d (epoch %.2f)' %
                      (step, float(step) * BATCH_SIZE / train_size))
                print('Minibatch loss: %.3f, learning rate: %.6f' % (l, lr))

                print('Minibatch accuracy: %.1f%%' % accuracy(predictions, batch_labels))
                print('Validation accuracy: %.1f%%' % accuracy(
                    eval_in_batches(validation_data, sess), validation_labels))

                test_accuracy = accuracy(eval_in_batches(test_data, sess), test_labels)
                print('Test error: %.1f%%' % test_accuracy)
