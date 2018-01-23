
** Download the data for CIFAR from here: https://www.cs.toronto.edu/~kriz/cifar.html **

**Specifically the CIFAR-10 python version link: https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz **



```python
# Put file path as a string here
CIFAR_DIR = '../../cifar-10-batches-py/'
```


```python
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        cifar_dict = pickle.load(fo, encoding='bytes')
    return cifar_dict
```


```python
dirs = ['batches.meta','data_batch_1','data_batch_2','data_batch_3','data_batch_4','data_batch_5','test_batch']
```


```python
all_data = [0,1,2,3,4,5,6]
```


```python
for i,direc in zip(all_data,dirs):
    all_data[i] = unpickle(CIFAR_DIR+direc)
```


```python
batch_meta = all_data[0]
data_batch1 = all_data[1]
data_batch2 = all_data[2]
data_batch3 = all_data[3]
data_batch4 = all_data[4]
data_batch5 = all_data[5]
test_batch = all_data[6]
```


```python
batch_meta
```




    {b'label_names': [b'airplane',
      b'automobile',
      b'bird',
      b'cat',
      b'deer',
      b'dog',
      b'frog',
      b'horse',
      b'ship',
      b'truck'],
     b'num_cases_per_batch': 10000,
     b'num_vis': 3072}




```python
data_batch1.keys()
```




    dict_keys([b'data', b'filenames', b'labels', b'batch_label'])



Loaded in this way, each of the batch files contains a dictionary with the following elements:
- data
    - a 10000x3072 (10000 x (32x32x3)) numpy array of uint8s.
    - each row of the array stores a 32x32 colour image. 
    - the first 1024 entries contain the red channel values, the next 1024 the green, and the final 1024 the blue.
    - the image is stored in row-major order, so that the first 32 entries of the array are the red channel values of the first row of the image.
- labels
    - a list of 10000 numbers in the range 0-9. 
    - the number at index i indicates the label of the ith image in the array data.

The dataset contains another file, called __batches.meta__, containing: 

- label_names
    - a 10-element list which gives meaningful names to the numeric labels in the labels array described above. 
        - For example
            - label_names[0] == "airplane"
            - label_names[1] == "automobile", etc.


```python
import matplotlib.pyplot as plt
%matplotlib inline

import numpy as np
```


```python
X = data_batch1[b"data"] 
```


```python
X.shape
```




    (10000, 3072)




```python
X = X.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("uint8")
```


```python
X[0].max()
```




    255




```python
# normalize
(X[0]/255).max()
```




    1.0




```python
plt.imshow(X[0])
```




    <matplotlib.image.AxesImage at 0x7f100de49b70>




![png](output_16_1.png)



```python
plt.imshow(X[1])
```




    <matplotlib.image.AxesImage at 0x7f100de3c128>




![png](output_17_1.png)



```python
plt.imshow(X[4])
```




    <matplotlib.image.AxesImage at 0x7f100ddfbb00>




![png](output_18_1.png)



```python
def one_hot_encode(vec, vals=10):
    '''
    For use to one-hot encode the 10- possible labels
    in TF-Keras:
        tf_keras.utils.to_categorical(dataset,num_classes)
    '''
    n = len(vec)
    out = np.zeros((n, vals))
    out[range(n), vec] = 1
    return out
```


```python
class CifarHelper():
    
    def __init__(self):
        self.i = 0
        
        self.all_train_batches = [data_batch1,data_batch2,data_batch3,data_batch4,data_batch5]
        self.test_batch = [test_batch]
        
        self.training_images = None
        self.training_labels = None
        
        self.test_images = None
        self.test_labels = None
    
    def set_up_images(self):
        
        self.training_images = np.vstack([d[b"data"] for d in self.all_train_batches])
        train_len = len(self.training_images)
        
        print(self.all_train_batches[0][b'data'].shape)
        print(self.training_images.shape)
        
        self.training_images = self.training_images.reshape(train_len,3,32,32).transpose(0,2,3,1)/255
        self.training_labels = one_hot_encode(np.hstack([d[b"labels"] for d in self.all_train_batches]), 10)
        
        print(self.training_images.shape)
        
        print(len(self.all_train_batches[0][b'labels']))
        print(self.training_labels.shape)
        
        self.test_images = np.vstack([d[b"data"] for d in self.test_batch])
        test_len = len(self.test_images)
        
        self.test_images = self.test_images.reshape(test_len,3,32,32).transpose(0,2,3,1)/255
        self.test_labels = one_hot_encode(np.hstack([d[b"labels"] for d in self.test_batch]), 10)

    def next_batch(self, batch_size):
        x = self.training_images[self.i:self.i+batch_size].reshape(100,32,32,3)
        y = self.training_labels[self.i:self.i+batch_size]
        self.i = (self.i + batch_size) % len(self.training_images)
        return x, y
```


```python
ch = CifarHelper()
ch.set_up_images()
```

    (10000, 3072)
    (50000, 3072)
    (50000, 32, 32, 3)
    10000
    (50000, 10)


## Creating the Model



```python
import tensorflow as tf
```

** Create 2 placeholders, x and y_true. Their shapes should be: **

* x shape = [None,32,32,3]
* y_true shape = [None,10]



```python
x = tf.placeholder(tf.float32,shape=[None,32,32,3])
y_true = tf.placeholder(tf.float32,shape=[None,10])
```

### Create one more placeholder called __hold_prob__. (No need for shape here.) This placeholder will just hold a single probability for the dropout.


```python
hold_prob = tf.placeholder(tf.float32)
```


```python
def init_weights(shape):
    init_w = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(init_w)

def init_bias(shape):
    init_b = tf.constant(0.1, shape=shape)
    return tf.Variable(init_b)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2by2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')

def convolutional_layer(input_x, shape):
    W = init_weights(shape)
    b = init_bias([shape[3]])
    return tf.nn.relu(conv2d(input_x, W) + b)

def full_connected_layer(input_layer, size):
    input_size = int(input_layer.get_shape()[1])
    W = init_weights([input_size, size])
    b = init_bias([size])
    return tf.matmul(input_layer, W) + b
```

### Create the Layers


```python
convo_1 = convolutional_layer(x,shape=[4,4,3,32])
convo_1_pooling = max_pool_2by2(convo_1)
```

** Create the next convolutional and pooling layers.  The last two dimensions of the convo_2 layer should be 32,64 **


```python
convo_2 = convolutional_layer(convo_1_pooling,shape=[4,4,32,64])
convo_2_pooling = max_pool_2by2(convo_2)
```

** Now create a flattened layer by reshaping the pooling layer into [-1,8 \* 8 \* 64] or [-1,4096] **


```python
8*8*64
```




    4096




```python
convo_2_flat = tf.reshape(convo_2_pooling,[-1,8*8*64])
```


```python
full_layer_one = tf.nn.relu(full_connected_layer(convo_2_flat,1024))
```

** Now create the dropout layer with tf.nn.dropout, remember to pass in your hold_prob placeholder. **


```python
full_one_dropout = tf.nn.dropout(full_layer_one,keep_prob=hold_prob)
```

** Finally set the output to y_pred by passing in the dropout layer into the normal_full_layer function. The size should be 10 because of the 10 possible labels**


```python
y_pred = full_connected_layer(full_one_dropout,10)
```

### Loss Function

** Create a cross_entropy loss function **


```python
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true,logits=y_pred))
```

### Optimizer
** Create the optimizer using an Adam Optimizer. **


```python
optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
train = optimizer.minimize(cross_entropy)
```

** Create a variable to intialize all the global tf variables. **


```python
init = tf.global_variables_initializer()
```

## Graph Session

** Perform the training and test print outs in a Tf session and run your model! **


```python
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(5000):
        batch = ch.next_batch(100)
        sess.run(train, feed_dict={x: batch[0], y_true: batch[1], hold_prob: 0.5})
        
        # PRINT OUT A MESSAGE EVERY 100 STEPS
        if i%100 == 0:
            
            print('Currently on step {}'.format(i))
            print('Accuracy is:')
            # Test the Train Model
            matches = tf.equal(tf.argmax(y_pred,1),tf.argmax(y_true,1))

            acc = tf.reduce_mean(tf.cast(matches,tf.float32))

            print(sess.run(acc,feed_dict={x:ch.test_images,y_true:ch.test_labels,hold_prob:1.0}))
            print('\n')
```

    Currently on step 0
    Accuracy is:
    0.0967
    
    
    Currently on step 100
    Accuracy is:
    0.4105
    
    
    Currently on step 200
    Accuracy is:
    0.4544
    
    
    Currently on step 300
    Accuracy is:
    0.4861
    
    
    Currently on step 400
    Accuracy is:
    0.4809
    
    
    Currently on step 500
    Accuracy is:
    0.5189
    
    
    Currently on step 600
    Accuracy is:
    0.5327
    
    
    Currently on step 700
    Accuracy is:
    0.5428
    
    
    Currently on step 800
    Accuracy is:
    0.5607
    
    
    Currently on step 900
    Accuracy is:
    0.5369
    
    
    Currently on step 1000
    Accuracy is:
    0.5768
    
    
    Currently on step 1100
    Accuracy is:
    0.5947
    
    
    Currently on step 1200
    Accuracy is:
    0.5897
    
    
    Currently on step 1300
    Accuracy is:
    0.5914
    
    
    Currently on step 1400
    Accuracy is:
    0.5965
    
    
    Currently on step 1500
    Accuracy is:
    0.6136
    
    
    Currently on step 1600
    Accuracy is:
    0.6258
    
    
    Currently on step 1700
    Accuracy is:
    0.6267
    
    
    Currently on step 1800
    Accuracy is:
    0.6182
    
    
    Currently on step 1900
    Accuracy is:
    0.626
    
    
    Currently on step 2000
    Accuracy is:
    0.6367
    
    

