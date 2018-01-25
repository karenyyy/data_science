
# Training and Evaluating Xception Model


```python
import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
```


```python
models = tf.contrib.keras.models
layers = tf.contrib.keras.layers
utils = tf.contrib.keras.utils
losses = tf.contrib.keras.losses
optimizers = tf.contrib.keras.optimizers 
metrics = tf.contrib.keras.metrics
preprocessing_image = tf.contrib.keras.preprocessing.image
applications = tf.contrib.keras.applications
```


```python
# load pre-trained Xception model and exclude top dense layer
base_model = applications.Xception(include_top=False,
                                   weights='imagenet',
                                   input_shape=(299,299,3),
                                   pooling='avg')
```

   Downloading data from https://github.com/fchollet/deep-learning-models/releases/download/v0.4/xception_weights_tf_dim_ordering_tf_kernels_notop.h5
    83410944/83683744 [============================>.] - ETA: 0s


```python
print("Model input shape: {}\n".format(base_model.input_shape))
print("Model output shape: {}\n".format(base_model.output_shape))
print("Model number of layers: {}\n".format(len(base_model.layers)))
```

   Model input shape: (None, 299, 299, 3)
    
   Model output shape: (None, 2048)
    
   Model number of layers: 133
    


# Fine-tune Xception Model

##  freeze weights of early layers to ease training (Important !!!)


```python
def fine_tune_Xception(base_model):
     
    # output of convolutional layers
    x = base_model.output

    # final Dense layer
    # 4 output classes
    outputs = layers.Dense(4, activation='softmax')(x)

    # define model with base_model's input
    model = models.Model(inputs=base_model.input, outputs=outputs)
    
    # freeze weights of early layers
    # to ease training
    for layer in model.layers[:40]:
        layer.trainable = False
    
    return model
```

# Compile Model


```python
def compile_model(model):
    # loss + optmizer +accuracy
    loss = losses.categorical_crossentropy
    optimizer = optimizers.RMSprop(lr=0.0001)
    metric = [metrics.categorical_accuracy]

    # compile model 
    model.compile(optimizer, loss, metric)
        
    return model
```

# Inspect Model Architecture


```python
model = fine_tune_Xception(base_model)
model = compile_model(model)
model.summary()
```

    ____________________________________________________________________________________________________
    Layer (type)                     Output Shape          Param #     Connected to                     
    ====================================================================================================
    input_1 (InputLayer)             (None, 299, 299, 3)   0                                            
    ____________________________________________________________________________________________________
    block1_conv1 (Conv2D)            (None, 149, 149, 32)  864         input_1[0][0]                    
    ____________________________________________________________________________________________________
    block1_conv1_bn (BatchNormalizat (None, 149, 149, 32)  128         block1_conv1[0][0]               
    ____________________________________________________________________________________________________
    block1_conv1_act (Activation)    (None, 149, 149, 32)  0           block1_conv1_bn[0][0]            
    ____________________________________________________________________________________________________
    block1_conv2 (Conv2D)            (None, 147, 147, 64)  18432       block1_conv1_act[0][0]           
    ____________________________________________________________________________________________________
    block1_conv2_bn (BatchNormalizat (None, 147, 147, 64)  256         block1_conv2[0][0]               
    ____________________________________________________________________________________________________
    block1_conv2_act (Activation)    (None, 147, 147, 64)  0           block1_conv2_bn[0][0]            
    ____________________________________________________________________________________________________
    block2_sepconv1 (SeparableConv2D (None, 147, 147, 128) 8768        block1_conv2_act[0][0]           
    ____________________________________________________________________________________________________
    block2_sepconv1_bn (BatchNormali (None, 147, 147, 128) 512         block2_sepconv1[0][0]            
    ____________________________________________________________________________________________________
    block2_sepconv2_act (Activation) (None, 147, 147, 128) 0           block2_sepconv1_bn[0][0]         
    ____________________________________________________________________________________________________
    block2_sepconv2 (SeparableConv2D (None, 147, 147, 128) 17536       block2_sepconv2_act[0][0]        
    ____________________________________________________________________________________________________
    block2_sepconv2_bn (BatchNormali (None, 147, 147, 128) 512         block2_sepconv2[0][0]            
    ____________________________________________________________________________________________________
    conv2d_1 (Conv2D)                (None, 74, 74, 128)   8192        block1_conv2_act[0][0]           
    ____________________________________________________________________________________________________
    block2_pool (MaxPooling2D)       (None, 74, 74, 128)   0           block2_sepconv2_bn[0][0]         
    ____________________________________________________________________________________________________
    batch_normalization_1 (BatchNorm (None, 74, 74, 128)   512         conv2d_1[0][0]                   
    ____________________________________________________________________________________________________
    add_1 (Add)                      (None, 74, 74, 128)   0           block2_pool[0][0]                
                                                                       batch_normalization_1[0][0]      
    ____________________________________________________________________________________________________
    block3_sepconv1_act (Activation) (None, 74, 74, 128)   0           add_1[0][0]                      
    ____________________________________________________________________________________________________
    block3_sepconv1 (SeparableConv2D (None, 74, 74, 256)   33920       block3_sepconv1_act[0][0]        
    ____________________________________________________________________________________________________
    block3_sepconv1_bn (BatchNormali (None, 74, 74, 256)   1024        block3_sepconv1[0][0]            
    ____________________________________________________________________________________________________
    block3_sepconv2_act (Activation) (None, 74, 74, 256)   0           block3_sepconv1_bn[0][0]         
    ____________________________________________________________________________________________________
    block3_sepconv2 (SeparableConv2D (None, 74, 74, 256)   67840       block3_sepconv2_act[0][0]        
    ____________________________________________________________________________________________________
    block3_sepconv2_bn (BatchNormali (None, 74, 74, 256)   1024        block3_sepconv2[0][0]            
    ____________________________________________________________________________________________________
    conv2d_2 (Conv2D)                (None, 37, 37, 256)   32768       add_1[0][0]                      
    ____________________________________________________________________________________________________
    block3_pool (MaxPooling2D)       (None, 37, 37, 256)   0           block3_sepconv2_bn[0][0]         
    ____________________________________________________________________________________________________
    batch_normalization_2 (BatchNorm (None, 37, 37, 256)   1024        conv2d_2[0][0]                   
    ____________________________________________________________________________________________________
    add_2 (Add)                      (None, 37, 37, 256)   0           block3_pool[0][0]                
                                                                       batch_normalization_2[0][0]      
    ____________________________________________________________________________________________________
    block4_sepconv1_act (Activation) (None, 37, 37, 256)   0           add_2[0][0]                      
    ____________________________________________________________________________________________________
    block4_sepconv1 (SeparableConv2D (None, 37, 37, 728)   188672      block4_sepconv1_act[0][0]        
    ____________________________________________________________________________________________________
    block4_sepconv1_bn (BatchNormali (None, 37, 37, 728)   2912        block4_sepconv1[0][0]            
    ____________________________________________________________________________________________________
    block4_sepconv2_act (Activation) (None, 37, 37, 728)   0           block4_sepconv1_bn[0][0]         
    ____________________________________________________________________________________________________
    block4_sepconv2 (SeparableConv2D (None, 37, 37, 728)   536536      block4_sepconv2_act[0][0]        
    ____________________________________________________________________________________________________
    block4_sepconv2_bn (BatchNormali (None, 37, 37, 728)   2912        block4_sepconv2[0][0]            
    ____________________________________________________________________________________________________
    conv2d_3 (Conv2D)                (None, 19, 19, 728)   186368      add_2[0][0]                      
    ____________________________________________________________________________________________________
    block4_pool (MaxPooling2D)       (None, 19, 19, 728)   0           block4_sepconv2_bn[0][0]         
    ____________________________________________________________________________________________________
    batch_normalization_3 (BatchNorm (None, 19, 19, 728)   2912        conv2d_3[0][0]                   
    ____________________________________________________________________________________________________
    add_3 (Add)                      (None, 19, 19, 728)   0           block4_pool[0][0]                
                                                                       batch_normalization_3[0][0]      
    ____________________________________________________________________________________________________
    block5_sepconv1_act (Activation) (None, 19, 19, 728)   0           add_3[0][0]                      
    ____________________________________________________________________________________________________
    block5_sepconv1 (SeparableConv2D (None, 19, 19, 728)   536536      block5_sepconv1_act[0][0]        
    ____________________________________________________________________________________________________
    block5_sepconv1_bn (BatchNormali (None, 19, 19, 728)   2912        block5_sepconv1[0][0]            
    ____________________________________________________________________________________________________
    block5_sepconv2_act (Activation) (None, 19, 19, 728)   0           block5_sepconv1_bn[0][0]         
    ____________________________________________________________________________________________________
    block5_sepconv2 (SeparableConv2D (None, 19, 19, 728)   536536      block5_sepconv2_act[0][0]        
    ____________________________________________________________________________________________________
    block5_sepconv2_bn (BatchNormali (None, 19, 19, 728)   2912        block5_sepconv2[0][0]            
    ____________________________________________________________________________________________________
    block5_sepconv3_act (Activation) (None, 19, 19, 728)   0           block5_sepconv2_bn[0][0]         
    ____________________________________________________________________________________________________
    block5_sepconv3 (SeparableConv2D (None, 19, 19, 728)   536536      block5_sepconv3_act[0][0]        
    ____________________________________________________________________________________________________
    block5_sepconv3_bn (BatchNormali (None, 19, 19, 728)   2912        block5_sepconv3[0][0]            
    ____________________________________________________________________________________________________
    add_4 (Add)                      (None, 19, 19, 728)   0           block5_sepconv3_bn[0][0]         
                                                                       add_3[0][0]                      
    ____________________________________________________________________________________________________
    block6_sepconv1_act (Activation) (None, 19, 19, 728)   0           add_4[0][0]                      
    ____________________________________________________________________________________________________
    block6_sepconv1 (SeparableConv2D (None, 19, 19, 728)   536536      block6_sepconv1_act[0][0]        
    ____________________________________________________________________________________________________
    block6_sepconv1_bn (BatchNormali (None, 19, 19, 728)   2912        block6_sepconv1[0][0]            
    ____________________________________________________________________________________________________
    block6_sepconv2_act (Activation) (None, 19, 19, 728)   0           block6_sepconv1_bn[0][0]         
    ____________________________________________________________________________________________________
    block6_sepconv2 (SeparableConv2D (None, 19, 19, 728)   536536      block6_sepconv2_act[0][0]        
    ____________________________________________________________________________________________________
    block6_sepconv2_bn (BatchNormali (None, 19, 19, 728)   2912        block6_sepconv2[0][0]            
    ____________________________________________________________________________________________________
    block6_sepconv3_act (Activation) (None, 19, 19, 728)   0           block6_sepconv2_bn[0][0]         
    ____________________________________________________________________________________________________
    block6_sepconv3 (SeparableConv2D (None, 19, 19, 728)   536536      block6_sepconv3_act[0][0]        
    ____________________________________________________________________________________________________
    block6_sepconv3_bn (BatchNormali (None, 19, 19, 728)   2912        block6_sepconv3[0][0]            
    ____________________________________________________________________________________________________
    add_5 (Add)                      (None, 19, 19, 728)   0           block6_sepconv3_bn[0][0]         
                                                                       add_4[0][0]                      
    ____________________________________________________________________________________________________
    block7_sepconv1_act (Activation) (None, 19, 19, 728)   0           add_5[0][0]                      
    ____________________________________________________________________________________________________
    block7_sepconv1 (SeparableConv2D (None, 19, 19, 728)   536536      block7_sepconv1_act[0][0]        
    ____________________________________________________________________________________________________
    block7_sepconv1_bn (BatchNormali (None, 19, 19, 728)   2912        block7_sepconv1[0][0]            
    ____________________________________________________________________________________________________
    block7_sepconv2_act (Activation) (None, 19, 19, 728)   0           block7_sepconv1_bn[0][0]         
    ____________________________________________________________________________________________________
    block7_sepconv2 (SeparableConv2D (None, 19, 19, 728)   536536      block7_sepconv2_act[0][0]        
    ____________________________________________________________________________________________________
    block7_sepconv2_bn (BatchNormali (None, 19, 19, 728)   2912        block7_sepconv2[0][0]            
    ____________________________________________________________________________________________________
    block7_sepconv3_act (Activation) (None, 19, 19, 728)   0           block7_sepconv2_bn[0][0]         
    ____________________________________________________________________________________________________
    block7_sepconv3 (SeparableConv2D (None, 19, 19, 728)   536536      block7_sepconv3_act[0][0]        
    ____________________________________________________________________________________________________
    block7_sepconv3_bn (BatchNormali (None, 19, 19, 728)   2912        block7_sepconv3[0][0]            
    ____________________________________________________________________________________________________
    add_6 (Add)                      (None, 19, 19, 728)   0           block7_sepconv3_bn[0][0]         
                                                                       add_5[0][0]                      
    ____________________________________________________________________________________________________
    block8_sepconv1_act (Activation) (None, 19, 19, 728)   0           add_6[0][0]                      
    ____________________________________________________________________________________________________
    block8_sepconv1 (SeparableConv2D (None, 19, 19, 728)   536536      block8_sepconv1_act[0][0]        
    ____________________________________________________________________________________________________
    block8_sepconv1_bn (BatchNormali (None, 19, 19, 728)   2912        block8_sepconv1[0][0]            
    ____________________________________________________________________________________________________
    block8_sepconv2_act (Activation) (None, 19, 19, 728)   0           block8_sepconv1_bn[0][0]         
    ____________________________________________________________________________________________________
    block8_sepconv2 (SeparableConv2D (None, 19, 19, 728)   536536      block8_sepconv2_act[0][0]        
    ____________________________________________________________________________________________________
    block8_sepconv2_bn (BatchNormali (None, 19, 19, 728)   2912        block8_sepconv2[0][0]            
    ____________________________________________________________________________________________________
    block8_sepconv3_act (Activation) (None, 19, 19, 728)   0           block8_sepconv2_bn[0][0]         
    ____________________________________________________________________________________________________
    block8_sepconv3 (SeparableConv2D (None, 19, 19, 728)   536536      block8_sepconv3_act[0][0]        
    ____________________________________________________________________________________________________
    block8_sepconv3_bn (BatchNormali (None, 19, 19, 728)   2912        block8_sepconv3[0][0]            
    ____________________________________________________________________________________________________
    add_7 (Add)                      (None, 19, 19, 728)   0           block8_sepconv3_bn[0][0]         
                                                                       add_6[0][0]                      
    ____________________________________________________________________________________________________
    block9_sepconv1_act (Activation) (None, 19, 19, 728)   0           add_7[0][0]                      
    ____________________________________________________________________________________________________
    block9_sepconv1 (SeparableConv2D (None, 19, 19, 728)   536536      block9_sepconv1_act[0][0]        
    ____________________________________________________________________________________________________
    block9_sepconv1_bn (BatchNormali (None, 19, 19, 728)   2912        block9_sepconv1[0][0]            
    ____________________________________________________________________________________________________
    block9_sepconv2_act (Activation) (None, 19, 19, 728)   0           block9_sepconv1_bn[0][0]         
    ____________________________________________________________________________________________________
    block9_sepconv2 (SeparableConv2D (None, 19, 19, 728)   536536      block9_sepconv2_act[0][0]        
    ____________________________________________________________________________________________________
    block9_sepconv2_bn (BatchNormali (None, 19, 19, 728)   2912        block9_sepconv2[0][0]            
    ____________________________________________________________________________________________________
    block9_sepconv3_act (Activation) (None, 19, 19, 728)   0           block9_sepconv2_bn[0][0]         
    ____________________________________________________________________________________________________
    block9_sepconv3 (SeparableConv2D (None, 19, 19, 728)   536536      block9_sepconv3_act[0][0]        
    ____________________________________________________________________________________________________
    block9_sepconv3_bn (BatchNormali (None, 19, 19, 728)   2912        block9_sepconv3[0][0]            
    ____________________________________________________________________________________________________
    add_8 (Add)                      (None, 19, 19, 728)   0           block9_sepconv3_bn[0][0]         
                                                                       add_7[0][0]                      
    ____________________________________________________________________________________________________
    block10_sepconv1_act (Activation (None, 19, 19, 728)   0           add_8[0][0]                      
    ____________________________________________________________________________________________________
    block10_sepconv1 (SeparableConv2 (None, 19, 19, 728)   536536      block10_sepconv1_act[0][0]       
    ____________________________________________________________________________________________________
    block10_sepconv1_bn (BatchNormal (None, 19, 19, 728)   2912        block10_sepconv1[0][0]           
    ____________________________________________________________________________________________________
    block10_sepconv2_act (Activation (None, 19, 19, 728)   0           block10_sepconv1_bn[0][0]        
    ____________________________________________________________________________________________________
    block10_sepconv2 (SeparableConv2 (None, 19, 19, 728)   536536      block10_sepconv2_act[0][0]       
    ____________________________________________________________________________________________________
    block10_sepconv2_bn (BatchNormal (None, 19, 19, 728)   2912        block10_sepconv2[0][0]           
    ____________________________________________________________________________________________________
    block10_sepconv3_act (Activation (None, 19, 19, 728)   0           block10_sepconv2_bn[0][0]        
    ____________________________________________________________________________________________________
    block10_sepconv3 (SeparableConv2 (None, 19, 19, 728)   536536      block10_sepconv3_act[0][0]       
    ____________________________________________________________________________________________________
    block10_sepconv3_bn (BatchNormal (None, 19, 19, 728)   2912        block10_sepconv3[0][0]           
    ____________________________________________________________________________________________________
    add_9 (Add)                      (None, 19, 19, 728)   0           block10_sepconv3_bn[0][0]        
                                                                       add_8[0][0]                      
    ____________________________________________________________________________________________________
    block11_sepconv1_act (Activation (None, 19, 19, 728)   0           add_9[0][0]                      
    ____________________________________________________________________________________________________
    block11_sepconv1 (SeparableConv2 (None, 19, 19, 728)   536536      block11_sepconv1_act[0][0]       
    ____________________________________________________________________________________________________
    block11_sepconv1_bn (BatchNormal (None, 19, 19, 728)   2912        block11_sepconv1[0][0]           
    ____________________________________________________________________________________________________
    block11_sepconv2_act (Activation (None, 19, 19, 728)   0           block11_sepconv1_bn[0][0]        
    ____________________________________________________________________________________________________
    block11_sepconv2 (SeparableConv2 (None, 19, 19, 728)   536536      block11_sepconv2_act[0][0]       
    ____________________________________________________________________________________________________
    block11_sepconv2_bn (BatchNormal (None, 19, 19, 728)   2912        block11_sepconv2[0][0]           
    ____________________________________________________________________________________________________
    block11_sepconv3_act (Activation (None, 19, 19, 728)   0           block11_sepconv2_bn[0][0]        
    ____________________________________________________________________________________________________
    block11_sepconv3 (SeparableConv2 (None, 19, 19, 728)   536536      block11_sepconv3_act[0][0]       
    ____________________________________________________________________________________________________
    block11_sepconv3_bn (BatchNormal (None, 19, 19, 728)   2912        block11_sepconv3[0][0]           
    ____________________________________________________________________________________________________
    add_10 (Add)                     (None, 19, 19, 728)   0           block11_sepconv3_bn[0][0]        
                                                                       add_9[0][0]                      
    ____________________________________________________________________________________________________
    block12_sepconv1_act (Activation (None, 19, 19, 728)   0           add_10[0][0]                     
    ____________________________________________________________________________________________________
    block12_sepconv1 (SeparableConv2 (None, 19, 19, 728)   536536      block12_sepconv1_act[0][0]       
    ____________________________________________________________________________________________________
    block12_sepconv1_bn (BatchNormal (None, 19, 19, 728)   2912        block12_sepconv1[0][0]           
    ____________________________________________________________________________________________________
    block12_sepconv2_act (Activation (None, 19, 19, 728)   0           block12_sepconv1_bn[0][0]        
    ____________________________________________________________________________________________________
    block12_sepconv2 (SeparableConv2 (None, 19, 19, 728)   536536      block12_sepconv2_act[0][0]       
    ____________________________________________________________________________________________________
    block12_sepconv2_bn (BatchNormal (None, 19, 19, 728)   2912        block12_sepconv2[0][0]           
    ____________________________________________________________________________________________________
    block12_sepconv3_act (Activation (None, 19, 19, 728)   0           block12_sepconv2_bn[0][0]        
    ____________________________________________________________________________________________________
    block12_sepconv3 (SeparableConv2 (None, 19, 19, 728)   536536      block12_sepconv3_act[0][0]       
    ____________________________________________________________________________________________________
    block12_sepconv3_bn (BatchNormal (None, 19, 19, 728)   2912        block12_sepconv3[0][0]           
    ____________________________________________________________________________________________________
    add_11 (Add)                     (None, 19, 19, 728)   0           block12_sepconv3_bn[0][0]        
                                                                       add_10[0][0]                     
    ____________________________________________________________________________________________________
    block13_sepconv1_act (Activation (None, 19, 19, 728)   0           add_11[0][0]                     
    ____________________________________________________________________________________________________
    block13_sepconv1 (SeparableConv2 (None, 19, 19, 728)   536536      block13_sepconv1_act[0][0]       
    ____________________________________________________________________________________________________
    block13_sepconv1_bn (BatchNormal (None, 19, 19, 728)   2912        block13_sepconv1[0][0]           
    ____________________________________________________________________________________________________
    block13_sepconv2_act (Activation (None, 19, 19, 728)   0           block13_sepconv1_bn[0][0]        
    ____________________________________________________________________________________________________
    block13_sepconv2 (SeparableConv2 (None, 19, 19, 1024)  752024      block13_sepconv2_act[0][0]       
    ____________________________________________________________________________________________________
    block13_sepconv2_bn (BatchNormal (None, 19, 19, 1024)  4096        block13_sepconv2[0][0]           
    ____________________________________________________________________________________________________
    conv2d_4 (Conv2D)                (None, 10, 10, 1024)  745472      add_11[0][0]                     
    ____________________________________________________________________________________________________
    block13_pool (MaxPooling2D)      (None, 10, 10, 1024)  0           block13_sepconv2_bn[0][0]        
    ____________________________________________________________________________________________________
    batch_normalization_4 (BatchNorm (None, 10, 10, 1024)  4096        conv2d_4[0][0]                   
    ____________________________________________________________________________________________________
    add_12 (Add)                     (None, 10, 10, 1024)  0           block13_pool[0][0]               
                                                                       batch_normalization_4[0][0]      
    ____________________________________________________________________________________________________
    block14_sepconv1 (SeparableConv2 (None, 10, 10, 1536)  1582080     add_12[0][0]                     
    ____________________________________________________________________________________________________
    block14_sepconv1_bn (BatchNormal (None, 10, 10, 1536)  6144        block14_sepconv1[0][0]           
    ____________________________________________________________________________________________________
    block14_sepconv1_act (Activation (None, 10, 10, 1536)  0           block14_sepconv1_bn[0][0]        
    ____________________________________________________________________________________________________
    block14_sepconv2 (SeparableConv2 (None, 10, 10, 2048)  3159552     block14_sepconv1_act[0][0]       
    ____________________________________________________________________________________________________
    block14_sepconv2_bn (BatchNormal (None, 10, 10, 2048)  8192        block14_sepconv2[0][0]           
    ____________________________________________________________________________________________________
    block14_sepconv2_act (Activation (None, 10, 10, 2048)  0           block14_sepconv2_bn[0][0]        
    ____________________________________________________________________________________________________
    global_average_pooling2d_1 (Glob (None, 2048)          0           block14_sepconv2_act[0][0]       
    ____________________________________________________________________________________________________
    dense_2 (Dense)                  (None, 4)             8196        global_average_pooling2d_1[0][0] 
    ====================================================================================================
    Total params: 20,869,676
    Trainable params: 19,170,396
    Non-trainable params: 1,699,280
    ____________________________________________________________________________________________________


# Image Preprocessing


```python
def preprocess_image(x):
    # make sure to do the calculations with floats
    x /= 255.0 # normalization
    x -= 0.5
    x *= 2.0
    
    # 'RGB'->'BGR'
    x = x[..., ::-1]
    # Zero-center by mean pixel
    x[..., 0] -= 103.939
    x[..., 1] -= 116.779
    x[..., 2] -= 123.68
    return x


train_datagen = preprocessing_image.ImageDataGenerator(
    preprocessing_function=preprocess_image, # function that will be implied on each input, will run before any other modification on it
    shear_range=0.2, # data augmentation
    zoom_range=0.2,  # data augmentation
    horizontal_flip=True)

test_datagen = preprocessing_image.ImageDataGenerator(preprocessing_function=preprocess_image)
```


```python
BASE_DIR = "/home/karen/Downloads/data/ImageNet_Utils/"

train_generator = train_datagen.flow_from_directory(
    os.path.join(BASE_DIR, "imageNet_dataset/train"),
    target_size=(299, 299),
    batch_size=32,
    class_mode='categorical',
    shuffle=True)

validation_generator = test_datagen.flow_from_directory(
    os.path.join(BASE_DIR, "imageNet_dataset/validation"),
    target_size=(299, 299),
    batch_size=32,
    class_mode='categorical',
    shuffle=True)
```

   Found 2063 images belonging to 4 classes.
   Found 881 images belonging to 4 classes.


# Train Model on ImageNet Dataset


```python
history = model.fit_generator(
    train_generator,
    steps_per_epoch=100,
    epochs=10,
    validation_data=validation_generator,
    validation_steps=20)
```

    Epoch 1/10
    100/100 [==============================] - 88s - loss: 0.1813 - categorical_accuracy: 0.9575 - val_loss: 7.4058 - val_categorical_accuracy: 0.4062
    Epoch 2/10
    100/100 [==============================] - 70s - loss: 0.0269 - categorical_accuracy: 0.9937 - val_loss: 8.6131 - val_categorical_accuracy: 0.4224
    Epoch 3/10
    100/100 [==============================] - 68s - loss: 0.0119 - categorical_accuracy: 0.9969 - val_loss: 7.6911 - val_categorical_accuracy: 0.4064
    Epoch 4/10
    100/100 [==============================] - 68s - loss: 0.0030 - categorical_accuracy: 0.9991 - val_loss: 3.9173 - val_categorical_accuracy: 0.4256
    Epoch 5/10
    100/100 [==============================] - 69s - loss: 0.0048 - categorical_accuracy: 0.9984 - val_loss: 0.3217 - val_categorical_accuracy: 0.9440
    Epoch 6/10
    100/100 [==============================] - 68s - loss: 0.0044 - categorical_accuracy: 0.9987 - val_loss: 0.1464 - val_categorical_accuracy: 0.9797
    Epoch 7/10
    100/100 [==============================] - 69s - loss: 7.7943e-04 - categorical_accuracy: 0.9997 - val_loss: 0.0744 - val_categorical_accuracy: 0.9875
    Epoch 8/10
    100/100 [==============================] - 68s - loss: 0.0061 - categorical_accuracy: 0.9984 - val_loss: 0.1788 - val_categorical_accuracy: 0.9719
    Epoch 9/10
    100/100 [==============================] - 68s - loss: 0.0020 - categorical_accuracy: 0.9994 - val_loss: 0.1850 - val_categorical_accuracy: 0.9781
    Epoch 10/10
    100/100 [==============================] - 67s - loss: 0.0036 - categorical_accuracy: 0.9981 - val_loss: 0.2001 - val_categorical_accuracy: 0.9696



```python
def plot_accuracy_and_loss(history):
    plt.figure(1, figsize= (15, 10))
    
    # plot train and test accuracy
    plt.subplot(221)
    plt.plot(history.history['categorical_accuracy'])
    plt.plot(history.history['val_categorical_accuracy'])
    plt.title('XceptionNet accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')

    # plot train and test loss
    plt.subplot(222)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('XceptionNet loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper right')
    
    plt.show()
```


```python
plot_accuracy_and_loss(history)
```


![png](https://raw.githubusercontent.com/karenyyy/data_science/master/TF_cv/images/output_17_0.png)


## Notes:

### If the train accuracy curve presents a stable trend and tends to converge smoothly, while the test accuracy curve oscilliates heavily and tends to be unstable, that indicates it is overfitting

- should train with more data
- apply regularization (like L2)
- or train again with lower learning rate

# Save Model Weights And Configuration


```python
# save model architecture
model_json = model.to_json()
open('xception_model.json', 'w').write(model_json)

# save model's learned weights
model.save_weights('image_classifier_xception.h5', overwrite=True)
```
