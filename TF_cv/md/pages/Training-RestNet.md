

```python
import tensorflow as tf
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
```


```python
models = tf.contrib.keras.models
layers = tf.contrib.keras.layers
initializers = tf.contrib.keras.initializers
regularizers = tf.contrib.keras.regularizers
losses = tf.contrib.keras.losses
optimizers = tf.contrib.keras.optimizers 
metrics = tf.contrib.keras.metrics
preprocessing_image = tf.contrib.keras.preprocessing.image
```

# ResNet Model


```python
def residual_block(input_tensor, 
                   filters, 
                   stage, 
                   reg=0.0, 
                   use_shortcuts=True):

    bn_name = 'bn' + str(stage)
    conv_name = 'conv' + str(stage)
    relu_name = 'relu' + str(stage)
    merge_name = 'merge' + str(stage)
    
    # sandwich model
        # 1x1
        # 3x3
        # 1x1

    # 1x1 conv
    # batchnorm-relu-conv
    # from input_filters to bottleneck_filters
    
    if stage>1: 
        # first activation is just after conv1
        x = layers.BatchNormalization(name=bn_name+'a')(input_tensor)
        x = layers.Activation('relu', name=relu_name+'a')(x)
    else:
        x = input_tensor

    x = layers.Convolution2D(
            filters[0], (1,1),
            kernel_regularizer=regularizers.l2(reg),
            use_bias=False, # since we use BN layer, thus no need for bias
            name=conv_name+'a'
        )(x)

    # 3x3 conv
    # batchnorm-relu-conv
    # from bottleneck_filters to bottleneck_filters
    x = layers.BatchNormalization(name=bn_name+'b')(x)
    x = layers.Activation('relu', name=relu_name+'b')(x)
    x = layers.Convolution2D(
            filters[1], (3,3),
            padding='same',
            kernel_regularizer=regularizers.l2(reg),
            use_bias = False,
            name=conv_name+'b'
        )(x)

    # 1x1 conv
    # batchnorm-relu-conv
    # from bottleneck_filters  to input_filters
    x = layers.BatchNormalization(name=bn_name+'c')(x)
    x = layers.Activation('relu', name=relu_name+'c')(x)
    x = layers.Convolution2D(
            filters[2], (1,1),
            kernel_regularizer=regularizers.l2(reg),
            name=conv_name+'c'
        )(x)

    # merge output with input layer (residual connection)
    if use_shortcuts:
        x = layers.add([x, input_tensor], name=merge_name)

    return x
```


```python
def ResNetPreAct(input_shape=(32,32,3), nb_classes=5, num_stages=5,
                 use_final_conv=False, reg=0.0):


    # Input
    img_input = layers.Input(shape=input_shape)

    #### Input stream ####
    # conv-BN-relu-(pool)
    x = layers.Convolution2D(
            128, (3,3), strides=(2, 2),
            padding='same',
            kernel_regularizer=regularizers.l2(reg),
            use_bias=False,
            name='conv0'
        )(img_input)
    x = layers.BatchNormalization(name='bn0')(x)
    x = layers.Activation('relu', name='relu0')(x)
    # x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='pool0')(x)

    #### Residual Blocks ####
    # pre activation
    # 1x1 conv: batchnorm-relu-conv
    # 3x3 conv: batchnorm-relu-conv
    # 1x1 conv: batchnorm-relu-conv
    for stage in range(1,num_stages+1):
        x = residual_block(x, [32,32,128], stage=stage, reg=reg)


    #### Output stream ####
    # BN-relu-(conv)-avgPool-softmax
    x = layers.BatchNormalization(name='bnF')(x)
    x = layers.Activation('relu', name='reluF')(x)

    # Optional final conv layer
    if use_final_conv:
        x = layers.Convolution2D(
                64, (3,3),
                padding='same',
                kernel_regularizer=regularizers.l2(reg),
                name='convF'
            )(x)
    
    pool_size = input_shape[0] / 2
    x = layers.AveragePooling2D((pool_size,pool_size),name='avg_pool')(x)

    x = layers.Flatten(name='flat')(x)
    x = layers.Dense(nb_classes, activation='softmax', name='fc10')(x)

    return models.Model(img_input, x, name='rnpa')
```

# Compile Model


```python
def compile_model(model):
    
    # loss
    loss = losses.categorical_crossentropy
    
    # optimizer
    # using Adam optimizer would need some time to warm up and then boost 
    # accuracy later in testing
    optimizer = optimizers.Adam(lr=0.0001)
    
    # metrics
    metric = [metrics.categorical_accuracy, metrics.top_k_categorical_accuracy]
    
    # compile model with loss, optimizer, and evaluation metrics
    model.compile(optimizer, loss, metric)
    
    return model
```

# Image Preprocessing And Augmentation


```python
train_datagen = preprocessing_image.ImageDataGenerator(
    rescale=1./255,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True)

test_datagen = preprocessing_image.ImageDataGenerator(rescale=1./255)
```


```python
BASE_DIR = "/home/karen/Downloads/data"

train_generator = train_datagen.flow_from_directory(
    os.path.join(BASE_DIR, "flower_dataset/train"),
    target_size=(32, 32),
    batch_size=32,
    class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
    os.path.join(BASE_DIR, "flower_dataset/validation"),
    target_size=(32, 32),
    batch_size=32,
    class_mode='categorical')
```

    Found 3599 images belonging to 5 classes.
    Found 2215 images belonging to 5 classes.



```python
model = ResNetPreAct(input_shape=(32, 32, 3), nb_classes=5, num_stages=5,
                     use_final_conv=False, reg=0.005)

model = compile_model(model)
```

# Train Model on Flower Dataset


```python
history = model.fit_generator(
    train_generator,
    steps_per_epoch=100,
    epochs=10,
    validation_data=validation_generator,
    validation_steps=20)
```

    Epoch 1/10
    100/100 [==============================] - 78s - loss: 4.7769 - categorical_accuracy: 0.3256 - top_k_categorical_accuracy: 1.0000 - val_loss: 4.7822 - val_categorical_accuracy: 0.2125 - val_top_k_categorical_accuracy: 1.0000
    Epoch 2/10
    100/100 [==============================] - 15s - loss: 4.2739 - categorical_accuracy: 0.5361 - top_k_categorical_accuracy: 1.0000 - val_loss: 4.6659 - val_categorical_accuracy: 0.2453 - val_top_k_categorical_accuracy: 1.0000
    Epoch 3/10
    100/100 [==============================] - 14s - loss: 3.9687 - categorical_accuracy: 0.5786 - top_k_categorical_accuracy: 1.0000 - val_loss: 4.6114 - val_categorical_accuracy: 0.2453 - val_top_k_categorical_accuracy: 1.0000
    Epoch 4/10
    100/100 [==============================] - 9s - loss: 3.7126 - categorical_accuracy: 0.5953 - top_k_categorical_accuracy: 1.0000 - val_loss: 4.5023 - val_categorical_accuracy: 0.2472 - val_top_k_categorical_accuracy: 1.0000
    Epoch 5/10
    100/100 [==============================] - 6s - loss: 3.4834 - categorical_accuracy: 0.6207 - top_k_categorical_accuracy: 1.0000 - val_loss: 4.0888 - val_categorical_accuracy: 0.3094 - val_top_k_categorical_accuracy: 1.0000
    Epoch 6/10
    100/100 [==============================] - 6s - loss: 3.3051 - categorical_accuracy: 0.6364 - top_k_categorical_accuracy: 1.0000 - val_loss: 3.5559 - val_categorical_accuracy: 0.4266 - val_top_k_categorical_accuracy: 1.0000- top_k_categorical_accuracy: 1
    Epoch 7/10
    100/100 [==============================] - 6s - loss: 3.1182 - categorical_accuracy: 0.6582 - top_k_categorical_accuracy: 1.0000 - val_loss: 3.1706 - val_categorical_accuracy: 0.5707 - val_top_k_categorical_accuracy: 1.0000
    Epoch 8/10
    100/100 [==============================] - 6s - loss: 2.9684 - categorical_accuracy: 0.6529 - top_k_categorical_accuracy: 1.0000 - val_loss: 2.9655 - val_categorical_accuracy: 0.6141 - val_top_k_categorical_accuracy: 1.0000
    Epoch 9/10
    100/100 [==============================] - 6s - loss: 2.8323 - categorical_accuracy: 0.6669 - top_k_categorical_accuracy: 1.0000 - val_loss: 2.8300 - val_categorical_accuracy: 0.6359 - val_top_k_categorical_accuracy: 1.0000
    Epoch 10/10
    100/100 [==============================] - 7s - loss: 2.7167 - categorical_accuracy: 0.6785 - top_k_categorical_accuracy: 1.0000 - val_loss: 2.7049 - val_categorical_accuracy: 0.6734 - val_top_k_categorical_accuracy: 1.0000


# Plot Accuracy And Loss Over Time


```python
def plot_accuracy_and_loss(history):
    plt.figure(1, figsize= (15, 10))
    
    # plot train and test accuracy
    plt.subplot(221)
    plt.plot(history.history['categorical_accuracy'])
    plt.plot(history.history['val_categorical_accuracy'])
    plt.title('ResNet accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')

    # plot train and test loss
    plt.subplot(222)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('ResNet loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper right')
    
    plt.show()
```


```python
plot_accuracy_and_loss(history)
```


![png](output_15_0.png)


# Save Model Weights And Configuration


```python
# save model architecture
model_json = model.to_json()
open('resnet_model.json', 'w').write(model_json)

# save model's learned weights
model.save_weights('image_classifier_resnet.h5', overwrite=True)
```
