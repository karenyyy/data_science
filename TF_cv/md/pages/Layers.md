
#  Layers

Each layer represents a high-level operation in the computational graph. These can be **visualized as lego blocks** can that be combined together and repeated across the architecture to form the neural network.

### Google's Inception model

<img src="https://raw.githubusercontent.com/karenyyy/data_science/master/TF_cv/images/googlenet.png" width="1500">


Common layers provided the TF-Keras `layers` module:

** Convolutional Layers**
```
tf_keras.layers.Conv1D
tf_keras.layers.Conv2D
tf_keras.layers.Conv3D
```

** Max-Pooling Layers**
```
tf_keras.layers.MaxPool1D
tf_keras.layers.MaxPool2D
tf_keras.layers.MaxPool3D
```

** Avergae Pooling Layers**
```
tf_keras.layers.AvgPool1D
tf_keras.layers.AvgPool2D
tf_keras.layers.AvgPool3D
```

** Fully-Connected layer**
```
tf_keras.layers.Dense
```

** Other Layers**
```
tf_keras.layers.Flatten
tf_keras.layers.Dropout
tf_keras.layers.BatchNormalization
```

** Activation Layers**
```
tf_keras.activations.relu
tf_keras.activations.sigmoid
tf_keras.activations.softmax
tf_keras.activations.tanh
tf_keras.activations.elu
tf_keras.activations.hard_sigmoid
tf_keras.activations.softplus
tf_keras.activations.softsign
tf_keras.activations.linear
```


```python
# output filter size
filters = 10

# feature map size
kernel_size = (3,3)

# conv1D - temporal convolution
tf_keras.layers.Conv1D(filters, kernel_size, strides=(1, 1), padding='valid',
                       activation= tf.nn.relu, use_bias=True,
                       kernel_initializer='glorot_uniform', bias_initializer='zeros')

# conv2D - spatial convolution over images
tf_keras.layers.Conv2D(filters, kernel_size, strides=(1, 1), padding='valid',
                       activation= tf.nn.relu, use_bias=True,
                       kernel_initializer='glorot_uniform', bias_initializer='zeros')

# conv3D - spatial convolution over volumes
tf_keras.layers.Conv3D(filters, kernel_size, strides=(1, 1), padding='valid',
                       activation= tf.nn.relu, use_bias=True,
                       kernel_initializer='glorot_uniform', bias_initializer='zeros')
```

- filters: the number output of filters in the convolution

- kernel_size: width and height of the 2D convolution window

- strides: the strides of the convolution along the width and height

- padding: `"valid"` or `"same"`.
    
- activation: Activation function to use.
        
- use_bias: Boolean, whether the layer uses a bias vector.


## Max-Pooling Layer



```python
# max-pooling 2D - spatial data
tf_keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2,2), padding='valid', data_format="channels_last")

# max-pooling 1D - temporal data
tf_keras.layers.MaxPool1D(pool_size=(2, 2), strides=(2,2), padding='valid', data_format="channels_last")

# max-pooling 3D - spatial or spatio-temporal
tf_keras.layers.MaxPool3D(pool_size=(2, 2), strides=(2,2), padding='valid', data_format="channels_last")
```

## Average Pooling Layer


```python
tf_keras.layers.AvgPool1D
tf_keras.layers.AvgPool2D
tf_keras.layers.AvgPool3D
```

## Dropout
Dropout consists in randomly setting
a fraction `p` of input units at each update during training to prevent overfitting.

- rate: float between 0 and 1. Fraction of the input units to drop.
   


```python
# dropout
tf_keras.layers.Dropout(rate = 0.5)
```

## Batch normalization layer

Normalize the activations of the previous layer at each batch

- axis: Integer, the axis that should be normalized (typically the features axis).
        For instance, after a `Conv2D` layer with
        `data_format="channels_first"`,
        set `axis=1` in `BatchNormalization`.
- momentum: Momentum for the moving average.
- epsilon: Small float added to variance to **avoid dividing by zero**.
- center: If True, add offset of `beta` to normalized tensor.
        If False, `beta` is ignored.
- scale: If True, multiply by `gamma`.
        If False, `gamma` is not used.
        When the next layer is linear (also e.g. `nn.relu`),
        this can be disabled since the scaling
        will be done by the next layer.
- beta_initializer: Initializer for the beta weight.
- gamma_initializer: Initializer for the gamma weight.
- moving_mean_initializer: Initializer for the moving mean.
- moving_variance_initializer: Initializer for the moving variance.



```python
f_keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True,
                                  scale=True, beta_initializer='zeros', gamma_initializer='ones',
                                  moving_mean_initializer='zeros', moving_variance_initializer='ones')

```

## Fully Connected (Dense) Layer

Fully-connected layer computes:

`output = activation(dot(input, kernel) + bias)`
where:
- `activation` is the element-wise activation function
- `kernel` is a weights matrix created by the layer
- `bias` is a bias vector created by the layer



```python
# fully connected layer
tf_keras.layers.Dense(units, activation=None, use_bias=True,
                      kernel_initializer='glorot_uniform', bias_initializer='zeros')

# flatten to vector
tf_keras.layers.Flatten()
```

## Activation Layer (increases the nonlinear properties of the decision function)


```python
tf_keras.activations.relu(inputs)
tf_keras.activations.sigmoid(inputs)
tf_keras.activations.softmax(inputs)
tf_keras.activations.tanh(inputs)
tf_keras.activations.elu(inputs)
tf_keras.activations.hard_sigmoid(inputs)
tf_keras.activations.softplus(inputs)
tf_keras.activations.softsign(inputs)
tf_keras.activations.linear(inputs)
```
