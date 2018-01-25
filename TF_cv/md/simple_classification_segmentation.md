
## Tensorflow Slim 101


```python
import sys
import os

sys.path.append("/home/karen/workspace/py/models/slim")
```

- download the VGG-16 model which we will use for classification of images and segmentation. 
- can also use networks that will consume less memory(for example, AlexNet). 
- For more models look [here](https://github.com/tensorflow/models/tree/master/research/slim#pre-trained-models). 


```python
from datasets import dataset_utils
import tensorflow as tf

url = "http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz"

# Specify where you want to download the model to
checkpoints_dir = '/home/karen/Downloads/data/VGG_checkpoints'

if not tf.gfile.Exists(checkpoints_dir):
    tf.gfile.MakeDirs(checkpoints_dir)

dataset_utils.download_and_uncompress_tarball(url, checkpoints_dir)
```

    >> Downloading vgg_16_2016_08_28.tar.gz 100.0%
    Successfully downloaded vgg_16_2016_08_28.tar.gz 513324920 bytes.


### Image Classification

The model that we have just downloaded was trained to be able to classify images
into [ImageNetLSVPR: 1000 classes](http://image-net.org/challenges/LSVRC/2014/browse-synsets).
The set of classes is very diverse. 


```python
%matplotlib inline

from matplotlib import pyplot as plt

import numpy as np
import os
import tensorflow as tf
import urllib

from datasets import imagenet
from nets import vgg
from preprocessing import vgg_preprocessing

slim = tf.contrib.slim

# default size of image for a particular network

image_size = vgg.vgg_16.default_image_size
```


```python
def create_readable_names_for_imagenet_labels():
    """Create a dict mapping label id to human readable string.
            one synset one per line, eg:
              #   n01440764
              #   n01443537
            synset in Imagenet, eg:
              #   n02119247    black fox
              #   n02119359    silver fox
      
      Code is based on
      https://github.com/tensorflow/models/blob/master/research/inception/inception/data/build_imagenet_data.py#L463
    """
    base_url = 'https://raw.githubusercontent.com/tensorflow/serving/master/tensorflow_serving/example'
    synset_url = '{}/imagenet_lsvrc_2015_synsets.txt'.format(base_url)
    synset_to_human_url = '{}/imagenet_metadata.txt'.format(base_url)

    filename, _ = urllib.request.urlretrieve(synset_url)
    synset_list = [s.strip() for s in open(filename).readlines()]
    num_synsets_in_ilsvrc = len(synset_list)
    assert num_synsets_in_ilsvrc == 1000

    filename, _ = urllib.request.urlretrieve(synset_to_human_url)
    synset_to_human_list = open(filename).readlines()
    num_synsets_in_all_imagenet = len(synset_to_human_list)
    assert num_synsets_in_all_imagenet == 21842

    synset_to_human = {}
    for s in synset_to_human_list:
        parts = s.strip().split('\t')
        assert len(parts) == 2
        synset = parts[0]
        human = parts[1]
        synset_to_human[synset] = human

    label_index = 1
    labels_to_names = {0: 'background'}
    for synset in synset_list:
        name = synset_to_human[synset]
        labels_to_names[label_index] = name
        label_index += 1

    return labels_to_names
```


```python
with tf.Graph().as_default():
    
    url = ("https://upload.wikimedia.org/wikipedia/commons/d/d9/"
           "First_Student_IC_school_bus_202076.jpg")
    
    image_string = urllib.request.urlopen(url).read()
    
    # Decode string into matrix with intensity values
    image = tf.image.decode_jpeg(image_string, channels=3)
    
    # Resize the input image, preserving the aspect ratio
    # and make a central crop of the resulted image.
    
    processed_image = vgg_preprocessing.preprocess_image(image,
                                                         image_size,
                                                         image_size,
                                                         is_training=False)
    
    processed_images  = tf.expand_dims(processed_image, 0)
    
    # Create the model, use the default arg scope to configure
    # the batch norm parameters, can define default
    # parameters for layers -- like stride, padding etc.
    with slim.arg_scope(vgg.vgg_arg_scope()):
        logits, _ = vgg.vgg_16(processed_images,
                               num_classes=1000,
                               is_training=False)
    
    # In order to get probabilities we apply softmax on the output.
    probabilities = tf.nn.softmax(logits)
    
    # reads the network weights
    # from the checkpoint file downloaded
    init_fn = slim.assign_from_checkpoint_fn(
        os.path.join(checkpoints_dir, 'vgg_16.ckpt'),
        slim.get_model_variables('vgg_16'))
    
    with tf.Session() as sess:
        init_fn(sess)
        
        np_image, network_input, probabilities = sess.run([image,
                                                           processed_image,
                                                           probabilities])
        probabilities = probabilities[0, 0:]
        sorted_inds = [i[0] for i in sorted(enumerate(-probabilities),
                                            key=lambda x:x[1])]
    
    # Show the downloaded image
    plt.figure()
    plt.imshow(np_image.astype(np.uint8))
    plt.suptitle("Downloaded image", fontsize=14, fontweight='bold')
    plt.axis('off')
    plt.show()
    
    # Show the image that is actually being fed to the network
    # The image was normalized first
    plt.imshow( network_input / (network_input.max() - network_input.min()) )
    plt.suptitle("Resized, Cropped and Mean-Centered input to the network",
                 fontsize=14, fontweight='bold')
    plt.axis('off')
    plt.show()

    names = create_readable_names_for_imagenet_labels()
    for i in range(5):
        index = sorted_inds[i]
        # Now we print the top-5 predictions 
        print('Probability %0.2f => [%s]' % (probabilities[index], names[index+1]))
        
    res = slim.get_model_variables()
```


![png](output_8_0.png)



![png](output_8_1.png)


    Probability 1.00 => [school bus]
    Probability 0.00 => [minibus]
    Probability 0.00 => [passenger car, coach, carriage]
    Probability 0.00 => [trolleybus, trolley coach, trackless trolley]
    Probability 0.00 => [cab, hack, taxi, taxicab]


### Image Annotation and Segmentation



As you can see from the previous example, ``only a certain part 
of the original image is being processed by the network. This is good only
for cases when we want to get a single prediction for an image.``

Sometimes we want to get more information from an image. For example,
it would be great to know about all the objects that are present in the
image. For example, network would tell us that it found a school bus,
other cars and building. This can be seen as a simple case of `Image Annotation`.

But what if we also want to get `spatial information` about the objects locations.
Can the network tell us that it sees a bus in the center of the image and building
on the top-right corner? 


There are cases when we need to classify each pixel of the image, also know as the task
of ``Segmentation``. 

> Imagine, that we have a huge dataset with pictures and we want to blur
faces of people there, so that we don't have to get their permission to publish these 
pictures. For example, you can see people's faces being blured in Google Street View. But
we only need to blur faces and not other content that might be important. _Segmentation_ can
help us in this case. We can get pixels that belong to faces and blur only them.

Next , we implement segmentation ``using an existing Convolutional
Neural Network by applying it in a Fully Convolutional manner, by casting the
Fully Connected Layers of a network into Convolutional`` 


```python
# print segmentation results with
# colorbar showing class names
def discrete_matshow(data, labels_names=[], title=""):
    #get discrete colormap
    cmap = plt.get_cmap('Paired', np.max(data)-np.min(data)+1)
    # set limits .5 outside true range
    mat = plt.matshow(data,
                      cmap=cmap,
                      vmin = np.min(data)-.5,
                      vmax = np.max(data)+.5)
    #tell the colorbar to tick at integers
    cax = plt.colorbar(mat,
                       ticks=np.arange(np.min(data),np.max(data)+1))
    
    # The names to be printed aside the colorbar
    if labels_names:
        cax.ax.set_yticklabels(labels_names)
    
    if title:
        plt.suptitle(title, fontsize=14, fontweight='bold')

```


```python
from preprocessing import vgg_preprocessing

from preprocessing.vgg_preprocessing import (_mean_image_subtraction,
                                            _R_MEAN, _G_MEAN, _B_MEAN)

with tf.Graph().as_default():
    
    url = ("https://upload.wikimedia.org/wikipedia/commons/d/d9/"
           "First_Student_IC_school_bus_202076.jpg")
    
    image_string = urllib2.urlopen(url).read()
    image = tf.image.decode_jpeg(image_string, channels=3)
    
    # Convert image to float32 before subtracting the
    # mean pixel value
    image_float = tf.to_float(image)
    
    # Subtract the mean pixel value from each pixel
    processed_image = _mean_image_subtraction(image_float,
                                              [_R_MEAN, _G_MEAN, _B_MEAN])

    input_image = tf.expand_dims(processed_image, 0)
    
    with slim.arg_scope(vgg.vgg_arg_scope()):
        
        # spatial_squeeze option enables to use network in a fully
        # convolutional manner
        logits, _ = vgg.vgg_16(input_image,
                               num_classes=1000,
                               is_training=False,
                               spatial_squeeze=False)
    
    # For each pixel we get predictions for each class
    # out of 1000. We need to pick the one with the highest
    # probability
    
    # But these are not probabilities,
    # because we didn't apply softmax. But if we pick a class
    # with the highest value it will be equivalent to picking
    # the highest value after applying softmax
    pred = tf.argmax(logits, dimension=3)
    
    init_fn = slim.assign_from_checkpoint_fn(
        os.path.join(checkpoints_dir, 'vgg_16.ckpt'),
        slim.get_model_variables('vgg_16'))
    
    with tf.Session() as sess:
        init_fn(sess)
        segmentation, np_image = sess.run([pred, image])

# Remove the first empty dimension
segmentation = np.squeeze(segmentation)

# Let's get unique predicted classes (from 0 to 1000) and
# relable the original predictions so that classes are
# numerated starting from zero
unique_classes, relabeled_image = np.unique(segmentation,
                                            return_inverse=True)

segmentation_size = segmentation.shape

relabeled_image = relabeled_image.reshape(segmentation_size)

labels_names = []

for index, current_class_number in enumerate(unique_classes):

    labels_names.append(str(index) + ' ' + names[current_class_number+1])

# Show the downloaded image
plt.figure()
plt.imshow(np_image.astype(np.uint8))
plt.suptitle("Input Image", fontsize=14, fontweight='bold')
plt.axis('off')
plt.show()

discrete_matshow(data=relabeled_image, labels_names=labels_names, title="Segmentation")


```


![png](output_12_0.png)



![png](output_12_1.png)


The segmentation that was obtained shows that network:

- able to find the school bus
- traffic sign in the left-top corner that can't be clearly seen in the image
- able to locate windows at the top-left corner and even made a hypothesis that it is a library (we don't know if that is true). 
- It also made a certain number of not so correct predictions. 

``Those are usually caused by the fact that the network can only see a part of image when it is centered at a pixel.`` The characteristic of a network that represents it is called G__receptive field__. 

Receptive
field of the network that we use in this blog is _404_ pixels. So when network can
only see a part of the school bus, it confuses it with taxi or pickup truck. You can
see that in the bottom-left corner of segmentation results. 

`It is not very precise because the network was originally trained to perform classification
and not segmentation.` 

Performing Segmentation using Convolutional Neural Networks can be seen as ``performing classification
at different parts of an input image``. We center network at a particular pixel, make prediction and
assign label to that pixel. This way we add spatial information to our classification and get
segmentation.
