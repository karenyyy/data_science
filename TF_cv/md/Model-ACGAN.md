

```python
import tensorflow as tf
import numpy as np
from collections import defaultdict
```


```python
models = tf.contrib.keras.models
layers = tf.contrib.keras.layers
utils = tf.contrib.keras.utils
losses = tf.contrib.keras.losses
optimizers = tf.contrib.keras.optimizers 
metrics = tf.contrib.keras.metrics
```


```python
preprocessing_image = tf.contrib.keras.preprocessing.image
datasets = tf.contrib.keras.datasets
```

# Generator
<img src="https://raw.githubusercontent.com/karenyyy/data_science/master/TF_cv/images/acgan-3.png" width="800">


```python
 def up_sampling_block(x, filter_number):
        # upsample block
        # factor = stride = 2
        x = layers.UpSampling2D(size=(2,2))(x)
        x = layers.Conv2D(filter_number, (5,5), padding='same', activation='relu')(x)
        return x
```

### Notes: 

- #### `Glorot normal initializer`, also called Xavier normal initializer.

    - It draws samples from a `truncated normal distribution` centered on 0 with 
    - $$stddev = \sqrt{\frac{2}{fan_{in} + fan_{out}}}$$ 

        - fan_in is the `number of input units in the weight tensor` 
        - fan_out is the `number of output units in the weight tensor`


- #### `Hadamard product` (also known as the Schur product or the entrywise product)

Hadamard product is a binary operation that `takes two matrices of the same dimensions, and produces another matrix with the same dimension`, where each element i,j is the product of elements i,j of the original two matrices.


```python
def generator(latent_size, classes=10):
    
    #######################
    ####### Input 1########
    
    # image class label
    image_class = layers.Input(shape=(1,), dtype='int32')
    
    # class embeddings
    # reconstruct: 10 => 100
    emb = layers.Embedding(classes, latent_size,
                           embeddings_initializer='glorot_normal')(image_class)
    
    # 10 classes in MNIST
    fc_embedding = layers.Flatten()(emb)
    
    #######################
    ####### Input 2########
    
    # latent noise vector
    latent_input = layers.Input(shape=(latent_size,))
    
    # hadamard product between latent embedding and a class conditional embedding
    h = layers.multiply([latent_input, fc_embedding])
    
    #########################################################
    ####### generator part 1: dense layer and reshape########
    
    x = layers.Dense(1024, activation='relu')(h)
    x = layers.Dense(128 * 7 * 7, activation='relu')(x)
    x = layers.Reshape((7, 7, 128))(x)
    
    #############################################################
    ####### generator part 2: upsampling and reconstruct ########
    
    # upsample to (14, 14, 128)
    x = up_sampling_block(x, 128)
    
    # upsample to (28, 28, 256)
    x = up_sampling_block(x, 256)
    
    ############################################################
    ####### generator part 3: conv layer and reduce dim ########
    
    # reduce channel into binary image (28, 28, 1)
    generated_img = layers.Conv2D(1, (2,2), padding='same', activation='tanh')(x)
    
    return models.Model(inputs=[latent_input, image_class], # here: since ACGAN is GAN with conditional label attached
                        outputs=generated_img,
                        name='generator') 
```


```python
g = generator(latent_size = 100, classes=10)
g.summary()
```

    ____________________________________________________________________________________________________
    Layer (type)                     Output Shape          Param #     Connected to                     
    ====================================================================================================
    input_4 (InputLayer)             (None, 1)             0                                            
    ____________________________________________________________________________________________________
    embedding_3 (Embedding)          (None, 1, 100)        1000        input_4[0][0]                    
    ____________________________________________________________________________________________________
    input_5 (InputLayer)             (None, 100)           0                                            
    ____________________________________________________________________________________________________
    flatten_3 (Flatten)              (None, 100)           0           embedding_3[0][0]                
    ____________________________________________________________________________________________________
    multiply_3 (Multiply)            (None, 100)           0           input_5[0][0]                    
                                                                       flatten_3[0][0]                  
    ____________________________________________________________________________________________________
    dense_5 (Dense)                  (None, 1024)          103424      multiply_3[0][0]                 
    ____________________________________________________________________________________________________
    dense_6 (Dense)                  (None, 6272)          6428800     dense_5[0][0]                    
    ____________________________________________________________________________________________________
    reshape_3 (Reshape)              (None, 7, 7, 128)     0           dense_6[0][0]                    
    ____________________________________________________________________________________________________
    up_sampling2d_5 (UpSampling2D)   (None, 14, 14, 128)   0           reshape_3[0][0]                  
    ____________________________________________________________________________________________________
    conv2d_7 (Conv2D)                (None, 14, 14, 128)   409728      up_sampling2d_5[0][0]            
    ____________________________________________________________________________________________________
    up_sampling2d_6 (UpSampling2D)   (None, 28, 28, 128)   0           conv2d_7[0][0]                   
    ____________________________________________________________________________________________________
    conv2d_8 (Conv2D)                (None, 28, 28, 256)   819456      up_sampling2d_6[0][0]            
    ____________________________________________________________________________________________________
    conv2d_9 (Conv2D)                (None, 28, 28, 1)     1025        conv2d_8[0][0]                   
    ====================================================================================================
    Total params: 7,763,433
    Trainable params: 7,763,433
    Non-trainable params: 0
    ____________________________________________________________________________________________________


# Discriminator
<img src="https://raw.githubusercontent.com/karenyyy/data_science/master/TF_cv/images/acgan-3.png" width="800">


```python
def conv_block(x, filter_number, stride):
    x = layers.Conv2D(filter_number, (3,3), padding='same', strides=stride)(x)
    x = layers.LeakyReLU()(x)
    x = layers.Dropout(0.3)(x)
    return x
```


```python
def discriminator(input_shape=(28, 28, 1)):

    input_img = layers.Input(shape=input_shape)
    
    # discriminator network
    x = conv_block(input_img, 32, (2,2))
    x = conv_block(input_img, 64, (1,1))
    x = conv_block(input_img, 128, (2,2))
    x = conv_block(input_img, 256, (1,1))
    
    features = layers.Flatten()(x)
    
    # binary classifier, image fake or real
    fake = layers.Dense(1, activation='sigmoid', name='generation')(features)
    
    # multi-class classifier, image digit class
    aux = layers.Dense(10, activation='softmax', name='auxiliary')(features)
    
    return models.Model(inputs=input_img, outputs=[fake, aux], name='discriminator')
```


```python
d = discriminator(input_shape=(28, 28, 1))
d.summary()
```

    ____________________________________________________________________________________________________
    Layer (type)                     Output Shape          Param #     Connected to                     
    ====================================================================================================
    input_10 (InputLayer)            (None, 28, 28, 1)     0                                            
    ____________________________________________________________________________________________________
    conv2d_24 (Conv2D)               (None, 28, 28, 256)   2560        input_10[0][0]                   
    ____________________________________________________________________________________________________
    leaky_re_lu_12 (LeakyReLU)       (None, 28, 28, 256)   0           conv2d_24[0][0]                  
    ____________________________________________________________________________________________________
    dropout_12 (Dropout)             (None, 28, 28, 256)   0           leaky_re_lu_12[0][0]             
    ____________________________________________________________________________________________________
    flatten_7 (Flatten)              (None, 200704)        0           dropout_12[0][0]                 
    ____________________________________________________________________________________________________
    generation (Dense)               (None, 1)             200705      flatten_7[0][0]                  
    ____________________________________________________________________________________________________
    auxiliary (Dense)                (None, 10)            2007050     flatten_7[0][0]                  
    ====================================================================================================
    Total params: 2,210,315
    Trainable params: 2,210,315
    Non-trainable params: 0
    ____________________________________________________________________________________________________


# Combine Generator with Discriminator
<img src="https://raw.githubusercontent.com/karenyyy/data_science/master/TF_cv/images/acgan-2.png" width="300">


```python
# Adam parameters pretrained
adam_lr = 0.0002
adam_beta_1 = 0.5

def ACGAN(latent_size = 100):
    # build the discriminator
    d_model = discriminator()
    d_model.compile(
        optimizer=optimizers.Adam(lr=adam_lr, beta_1=adam_beta_1),
        loss=['binary_crossentropy', 'sparse_categorical_crossentropy']
    )

    # build the generator
    g_model = generator(latent_size)
    g_model.compile(optimizer=optimizers.Adam(lr=adam_lr, beta_1=adam_beta_1),
                      loss='binary_crossentropy')

    # Inputs
    latent = layers.Input(shape=(latent_size, ), name='latent_noise')
    image_class = layers.Input(shape=(1,), dtype='int32', name='image_class')

    # Get a fake image
    fake_img = g_model([latent, image_class])

    # Only train generator in combined model
    d_model.trainable = False
    fake_or_real, label = d_model(fake_img)
    acgan = models.Model(inputs=[latent, image_class],
                            outputs=[fake_or_real, label],
                            name='ACGAN')

    acgan.compile(
        optimizer=optimizers.Adam(lr=adam_lr, beta_1=adam_beta_1),
        loss=['binary_crossentropy', 'sparse_categorical_crossentropy']
    )
    
    return acgan, g_model, d_model
```


```python
acgan,_,_ = ACGAN(latent_size = 100)
acgan.summary()
```

    ____________________________________________________________________________________________________
    Layer (type)                     Output Shape          Param #     Connected to                     
    ====================================================================================================
    latent_noise (InputLayer)        (None, 100)           0                                            
    ____________________________________________________________________________________________________
    image_class (InputLayer)         (None, 1)             0                                            
    ____________________________________________________________________________________________________
    generator (Model)                (None, 28, 28, 1)     7763433     latent_noise[0][0]               
                                                                       image_class[0][0]                
    ____________________________________________________________________________________________________
    discriminator (Model)            [(None, 1), (None, 10 2210315     generator[1][0]                  
    ====================================================================================================
    Total params: 9,973,748
    Trainable params: 7,763,433
    Non-trainable params: 2,210,315
    ____________________________________________________________________________________________________



```python
# reshape to (..., 28, 28, 1)
# normalize dataset with range [-1, 1]
(X_train, y_train), (X_test, y_test) = datasets.mnist.load_data()

# normalize and reshape train set
X_train = (X_train.astype(np.float32) - 127.5) / 127.5
X_train = np.expand_dims(X_train, axis=-1)

# normalize and reshape test set
X_test = (X_test.astype(np.float32) - 127.5) / 127.5
X_test = np.expand_dims(X_test, axis=-1)

train_size, test_size = X_train.shape[0], X_test.shape[0]
```


```python
def generate_batch_noise_and_labels(batch_size, latent_size):

    # generate a new batch of noise
    noise = np.random.uniform(-1, 1, (batch_size, latent_size))

    # sample some labels
    sampled_labels = np.random.randint(0, 10, batch_size)

    return noise, sampled_labels
```


```python
epochs = 50
batch_size = 100

train_history = defaultdict(list)
test_history = defaultdict(list)

latent_size = 100

acgan, g_model, d_model = ACGAN(latent_size)

for epoch in range(epochs):
    print('Epoch {} of {}'.format(epoch + 1, epochs))

    batches = int(X_train.shape[0] / batch_size)
    progress_bar = utils.Progbar(target=batches)

    epoch_gen_loss = []
    epoch_dis_loss = []
    
    ############################### batches training start #######################################
    
    for index in range(batches):
        progress_bar.update(index)
        
        ###############################################################
        ######################## Train Discriminator ##################
        
        # generate noise and labels
        noise, sampled_labels = generate_batch_noise_and_labels(batch_size, latent_size)
        
        # generate a batch of fake images, using the generated labels as a conditioner
        generated_images = g_model.predict([noise, sampled_labels.reshape((-1, 1))], verbose=0)
        
        # get a batch of real images
        image_batch = X_train[index * batch_size:(index + 1) * batch_size]
        label_batch = y_train[index * batch_size:(index + 1) * batch_size]

        # construct discriminator dataset
        X = np.concatenate((image_batch, generated_images))
        binary_y = np.array([1] * batch_size + [0] * batch_size)
        multiclass_y = np.concatenate((label_batch, sampled_labels), axis=0)

        # train discriminator
        epoch_dis_loss.append(d_model.train_on_batch(X, [binary_y, multiclass_y])) # acgan with 2 tasks
        
        ##################################################################
        ######################### Train Generator ########################
        
        # generate 2 * batch size here such that we have
        # the generator optimize over an identical number of images as the
        # discriminator       
        noise, sampled_labels = generate_batch_noise_and_labels(2 * batch_size, latent_size)
        
        # here: np.ones(2 * batch_size) --> 
        # all label '1' aims to trick discrimintor to think all generated images are all real(which is labeled as '1')
        epoch_gen_loss.append(acgan.train_on_batch(
            [noise, sampled_labels.reshape((-1, 1))], [np.ones(2 * batch_size), sampled_labels]))
    print('\nTesting for epoch {}:'.format(epoch + 1))
    
    
    ####################################### training end ##################################################
    
    
    ####################################### evaluation start ################################################
    
    
    ################################################################
    ##################### Evaluate Discriminator ###################

    # generate a new batch of noise
    noise, sampled_labels = generate_batch_noise_and_labels(test_size, latent_size)
    
    # generate images
    generated_images = g_model.predict(
        [noise, sampled_labels.reshape((-1, 1))], verbose=False)
    
    # construct discriminator evaluation dataset
    X = np.concatenate((X_test, generated_images))
    binary_y = np.array([1] * test_size + [0] * test_size)
    multiclass_y = np.concatenate((y_test, sampled_labels), axis=0)

    # evaluate discriminator
    # test loss
    discriminator_test_loss = d_model.evaluate(X, [binary_y, multiclass_y], verbose=False)
    # train loss
    discriminator_train_loss = np.mean(np.array(epoch_dis_loss), axis=0)
    
    ################################################################
    ######################## Evaluate Generator ####################

    # make new noise
    noise, sampled_labels = generate_batch_noise_and_labels(2 * test_size, latent_size)

    # evaluate generator : evaluate([input], [output], ...)
    # test loss
    generator_test_loss = acgan.evaluate(
        [noise, sampled_labels.reshape((-1, 1))],
        [np.ones(2 * test_size), sampled_labels], verbose=False)

    # train loss
    generator_train_loss = np.mean(np.array(epoch_gen_loss), axis=0)
    
    
    ####################################### evaluation end ################################################
    
    

    ###############################################################
    #################### Save Losses per Epoch ####################
    
    
    # append train losses
    train_history['generator'].append(generator_train_loss)
    train_history['discriminator'].append(discriminator_train_loss)

    # append test losses
    test_history['generator'].append(generator_test_loss)
    test_history['discriminator'].append(discriminator_test_loss)
    
    # save weights every epoch
    g_model.save_weights(
        '../logs/params_generator_epoch_{0:03d}.hdf5'.format(epoch), True)
    d_model.save_weights(
        '../logs/params_discriminator_epoch_{0:03d}.hdf5'.format(epoch), True)

    
################################### epoch pass end ################################################

# Save train test loss history
pickle.dump({'train': train_history, 'test': test_history},
            open('/home/karen/Downloads/data/logs/acgan-history.pkl', 'wb'))
```

    Epoch 1 of 50
      0/600 [..............................] - ETA: 0s


    ---------------------------------------------------------------------------




```python
hist = pickle.load(open('/home/karen/Downloads/data/logs/acgan-history.pkl'))

for p in ['train', 'test']:
    for g in ['discriminator', 'generator']:
        hist[p][g] = pd.DataFrame(hist[p][g], columns=['loss', 'generation_loss', 'auxiliary_loss'])
        plt.plot(hist[p][g]['generation_loss'], label='{} ({})'.format(g, p))

# get the NE and show as an equilibrium point
plt.hlines(-np.log(0.5), 0, hist[p][g]['generation_loss'].shape[0], label='Nash Equilibrium')
plt.legend()
plt.title(r'$L_s$ (generation loss) per Epoch')
plt.xlabel('Epoch')
plt.ylabel(r'$L_s$')
plt.show()
```


```python
for g in ['discriminator', 'generator']:
    for p in ['train', 'test']:
        plt.plot(hist[p][g]['auxiliary_loss'], label='{} ({})'.format(g, p))

plt.legend()
plt.title(r'$L_c$ (classification loss) per Epoch')
plt.xlabel('Epoch')
plt.ylabel(r'$L_c$')
plt.semilogy()
plt.show()
```


```python
# load the weights from the last epoch
gen.load_weights(sorted(glob('/home/karen/Downloads/data/logs/params_generator*'))[-1])

# construct batch of noise and labels
noise = np.tile(np.random.uniform(-1, 1, (10, latent_size)), (10, 1))
sampled_labels = np.array([[i] * 10 for i in range(10)]).reshape(-1, 1)

# generate digits
generated_images = gen.predict([noise, sampled_labels], verbose=0)

# arrange them into a grid and un-normalize the pixels
img = (np.concatenate([r.reshape(-1, 28)
                       for r in np.split(generated_images, 10)
                       ], axis=-1) * 127.5 + 127.5).astype(np.uint8)

# plot images
plt.imshow(img, cmap='gray')
_ = plt.axis('off')
```
