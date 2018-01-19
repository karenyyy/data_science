import os
import re
import sys
import tarfile

import tensorflow as tf

IMAGE_SIZE = 24

# Global constants describing the CIFAR-10 data set.
NUM_CLASSES = 10
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 10000

# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 350.0
LEARNING_RATE_DECAY_FACTOR = 0.1
INITIAL_LEARNING_RATE = 0.1



DATA_URL = 'https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz'

NAME="tf_cifar10_testing"

def _activation_summary(x):
    tensor_name = re.sub('%s_[0-9]*/' % NAME, '', x.op.name)
    tf.summary.histogram(tensor_name + '/activations', x)
    tf.summary.scalar(tensor_name + '/sparsity',
                                       tf.nn.zero_fraction(x))


def _variable_with_weight_decay(name, shape, stddev, wd):
  var = tf.get_variable(
      name,
      shape,
      tf.truncated_normal_initializer(stddev=stddev, dtype=tf.float32))
  if wd is not None:
    weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)
  return var



def inputs(eval_data, data_dir, batch_size):
  if not eval_data:
    filenames = [os.path.join(data_dir, 'data_batch_%d.bin' % i)
                 for i in range(1, 6)]
    num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
  else:
    filenames = [os.path.join(data_dir, 'test_batch.bin')]
    num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_EVAL

    # Create a queue that produces the filenames to read.
    filename_queue = tf.train.string_input_producer(filenames)

    def read_cifar10(filename_queue):
        label_bytes = 1  # 2 for CIFAR-100
        height = 32
        width = 32
        channel = 3
        image_bytes = height * width * channel
        record_bytes = label_bytes + image_bytes
        # Every record consists of a label followed by the image, with a
        # fixed number of bytes for each.

        key, value = tf.FixedLengthRecordReader(record_bytes=record_bytes).read(filename_queue)

        record_bytes = tf.decode_raw(value, tf.uint8)

        label = tf.cast(
            tf.strided_slice(record_bytes, [0], [label_bytes]), tf.int32)

        channel_major = tf.reshape(
            tf.strided_slice(record_bytes, [label_bytes],
                             [label_bytes + image_bytes]),
            [channel, height, width])
        # Convert from [channel, height, width] to [height, width, channel].
        uint8image = tf.transpose(channel_major, [1, 2, 0])

        return height, width, channel, key, label, uint8image

    
