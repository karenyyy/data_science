
# MNIST with CNN (Not TF-Keras)


```python
import tensorflow as tf
```


```python
from tensorflow.examples.tutorials.mnist import input_data
```


```python
mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)
```

    Extracting MNIST_data/train-images-idx3-ubyte.gz
    Extracting MNIST_data/train-labels-idx1-ubyte.gz
    Extracting MNIST_data/t10k-images-idx3-ubyte.gz
    Extracting MNIST_data/t10k-labels-idx1-ubyte.gz



```python
def init_weights(shape):
    init_w = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(init_w)
```


```python
def init_bias(shape):
    init_b = tf.constant(0.1, shape=shape)
    return tf.Variable(init_b)
```

Create a 2D convolution using builtin conv2d from TF. From those docs:

__Computes a 2-D convolution given 4-D `input` and `filter` tensors__.

-  __input tensor__ of shape `[batch_size, in_height, in_width, in_channels]`


-  __filter / kernel tensor__ of shape
`[filter_height, filter_width, in_channels, out_channels]`, this op
performs the following:

    - 1.Flattens the filter to a 2-D matrix with shape
   `[filter_height * filter_width * in_channels, output_channels]`.
    - 2.Extracts image patches from the input tensor to form a *virtual*
   tensor of shape `[batch_size, out_height, out_width,
   filter_height * filter_width * in_channels]`.
    - 3.For each patch, right-multiplies the filter matrix and the image patch
   vector.



```python
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
```

Create a max pooling layer, again using built in TF functions:

Performs the max pooling on the input.

    Args:
      value: A 4-D `Tensor` with shape `[batch_size, height, width, channels]` and
        type `tf.float32`.
      ksize: A list of ints that has length >= 4.  The size of the window for
        each dimension of the input tensor.
      strides: A list of ints that has length >= 4.  The stride of the sliding
        window for each dimension of the input tensor.
      padding: A string, either `'VALID'` or `'SAME'`. 


```python
def max_pool_2by2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')
```


```python
def convolutional_layer(input_x, shape):
    W = init_weights(shape)
    b = init_bias([shape[3]])
    return tf.nn.relu(conv2d(input_x, W) + b)
```


```python
def full_connected_layer(input_layer, size):
    input_size = int(input_layer.get_shape()[1])
    W = init_weights([input_size, size])
    b = init_bias([size])
    return tf.matmul(input_layer, W) + b
```

### Placeholders


```python
x = tf.placeholder(tf.float32,shape=[None,784])
```


```python
y_true = tf.placeholder(tf.float32,shape=[None,10])
```

### Layers


```python
x_image = tf.reshape(x,[-1,28,28,1])
```


```python
convo_1 = convolutional_layer(x_image,shape=[6,6,1,32])
convo_1_pooling = max_pool_2by2(convo_1)
```


```python
convo_2 = convolutional_layer(convo_1_pooling,shape=[6,6,32,64])
convo_2_pooling = max_pool_2by2(convo_2)
```


```python
convo_2_flat = tf.reshape(convo_2_pooling,[-1,7*7*64])
full_layer_one = tf.nn.relu(full_connected_layer(convo_2_flat,1024))
```


```python
# NOTE THE PLACEHOLDER HERE!
hold_prob = tf.placeholder(tf.float32)
full_one_dropout = tf.nn.dropout(full_layer_one,keep_prob=hold_prob)
```


```python
y_pred = full_connected_layer(full_one_dropout,10)
```

### Loss Function


```python
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true,logits=y_pred))
```

### Optimizer


```python
optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)
train = optimizer.minimize(cross_entropy)
```

### Intialize Variables


```python
init = tf.global_variables_initializer()
```

### Session


```python
steps = 5000

with tf.Session() as sess:
    
    sess.run(init)
    
    for i in range(steps):
        
        batch_x , batch_y = mnist.train.next_batch(100)
        
        sess.run(train,feed_dict={x:batch_x,y_true:batch_y,hold_prob:0.5})
        
        # PRINT OUT A MESSAGE EVERY 100 STEPS
        if i%100 == 0:
            
            print('Currently on step {}'.format(i))
            print('Accuracy is:')
            # Test the Train Model
            matches = tf.equal(tf.argmax(y_pred,1),tf.argmax(y_true,1))

            acc = tf.reduce_mean(tf.cast(matches,tf.float32))

            print(sess.run(acc,feed_dict={x:mnist.test.images,y_true:mnist.test.labels,hold_prob:1.0}))
            print('\n')
```

    Currently on step 0
    Accuracy is:
    0.0898
    
    
    Currently on step 100
    Accuracy is:
    0.875
    
    
    Currently on step 200
    Accuracy is:
    0.9207
    
    
    Currently on step 300
    Accuracy is:
    0.9376
    
    
    Currently on step 400
    Accuracy is:
    0.9457
    
    
    Currently on step 500
    Accuracy is:
    0.9547
    
    
    Currently on step 600
    Accuracy is:
    0.9586
    
    
    Currently on step 700
    Accuracy is:
    0.9613
    
    
    Currently on step 800
    Accuracy is:
    0.9659
    
    
    Currently on step 900
    Accuracy is:
    0.9674
    
    
    Currently on step 1000
    Accuracy is:
    0.9708
    
    
    Currently on step 1100
    Accuracy is:
    0.9723
    
    
    Currently on step 1200
    Accuracy is:
    0.972
    
    
    Currently on step 1300
    Accuracy is:
    0.9749
    
    
    Currently on step 1400
    Accuracy is:
    0.976
    
    
    Currently on step 1500
    Accuracy is:
    0.9776
    
    
    Currently on step 1600
    Accuracy is:
    0.9783
    
    
    Currently on step 1700
    Accuracy is:
    0.9787
    
    
    Currently on step 1800
    Accuracy is:
    0.9794
    
    
    Currently on step 1900
    Accuracy is:
    0.9787
    
    
    Currently on step 2000
    Accuracy is:
    0.9811
    
    
    Currently on step 2100
    Accuracy is:
    0.9812
    
    
    Currently on step 2200
    Accuracy is:
    0.982
    
    
    Currently on step 2300
    Accuracy is:
    0.9819
    
    
    Currently on step 2400
    Accuracy is:
    0.9822
    
    
    Currently on step 2500
    Accuracy is:
    0.9822
    
    
    Currently on step 2600
    Accuracy is:
    0.9838
    
    
    Currently on step 2700
    Accuracy is:
    0.9839
    
    
    Currently on step 2800
    Accuracy is:
    0.9857
    
    
    Currently on step 2900
    Accuracy is:
    0.9829
    
    
    Currently on step 3000
    Accuracy is:
    0.9844
    
    
    Currently on step 3100
    Accuracy is:
    0.9859
    
    
    Currently on step 3200
    Accuracy is:
    0.9851
    
    
    Currently on step 3300
    Accuracy is:
    0.9866
    
    

