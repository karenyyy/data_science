

# VGG Paper

[_Very Deep Convolutional Networks for Large-Scale Image Recognition_](https://arxiv.org/pdf/1409.1556.pdf) 
<img src="../../https://raw.githubusercontent.com/karenyyy/data_science/master/TF_cv/images/VGG-paper.png" width="800">

# VGG Result

<img src="https://raw.githubusercontent.com/karenyyy/data_science/master/TF_cv/images/VGG-result.png" width="800">


# VGG Architecture (two size “VGG16” and “VGG19”)

- extremely homogeneous architecture:
    - 3x3 convolutional layers
    - 2x2 max pooling
    - 2 fully-connected layers
    - softmax classifier. 


<img src="https://raw.githubusercontent.com/karenyyy/data_science/master/TF_cv/images/VGGNet.png" width="500">

## VGG16 Architecture

<img src="https://raw.githubusercontent.com/karenyyy/data_science/master/TF_cv/images/vgg16.png" width="500">


**Main Points of the VGG architecture:**

- The use of only 3x3 sized filters, which is small compared to previous models that used 11x11 and 7x7 filter size. One of the benefits is a __decrease in the number of parameters__. 


- __3 conv layers back to back have an effective receptive field of 7x7__


- As the spatial size of the input volumes at each layer decrease, the depth of the volumes increase due to the increased number of filters.


- __The number of filters doubles after each maxpool layer__. This reinforces the idea of shrinking spatial dimensions, but growing depth.


- Works well on both image classification and localization tasks. __Localization is treated as a regression task.__


** Down side of VGG architecture:**

- it can be slow to train on large dataset because __the number of model parameters is quite large, due to its depth and its large fully-connected layers.__  (Smaller network architectures have been since proposed with comparable performance, such as SqueezeNet.)
