
# Deep ResNet

# Deep Residual Networks (ResNets)


### Motivation
- Network depth is of crucial importance in neural network architectures
- Deeper networks are more difficult to train.
- Residual learning eases the training
- Enables them to be substantially deeper  with improved performance

<img src="https://raw.githubusercontent.com/karenyyy/data_science/master/TF_cv/images/resnet-results.png" width="700">

<img src="https://raw.githubusercontent.com/karenyyy/data_science/master/TF_cv/images/resnet-coco.png" width="700">

# Training Increasingly Deeper Networks

### Common Practices

**Initialization**
- Careful __initialization of model weights__
    - __Avoid exploding or vanishing gradients__ during training

**Batch Normalization**
- Batch Normalization of each layer for each training mini-batch
    - Accelerates training
    - __Less sensitive to initialization__, for more stable training
    - __Improves regularization__ of model (better generalization)

<img src="https://raw.githubusercontent.com/karenyyy/data_science/master/TF_cv/images/bn.png" width="700">

# Simply stacking more layers does not improve performance

- 56-layer	CNN	has	higher	**training error and	test	error**	than	20-layer	CNN
- __accuracy gets saturated and then starts degrading__

<img src="https://raw.githubusercontent.com/karenyyy/data_science/master/TF_cv/images/stacking.png" width="900">

# ResNet Introduces Residual Learning
Original Paper:
[Deep Residual Learning for Image Recognition](https://arxiv.org/pdf/1512.03385.pdf)

** Plain Layer Stacking**

<img src="https://raw.githubusercontent.com/karenyyy/data_science/master/TF_cv/images/plain-stack.png" width="300">

** Stacking with Residual Connection**

- add an identity skip connection
- layer learns residual mapping instead
- makes optimization easier expecially for deeper layers
- helps propagate signal deep into the network with less degradation

**New assumption:**
- optimal function is closer to an identity mapping than to a zero mapping
- easier for network to learn residual error
- each layer is responsible for fine-tuning the output of a previous block (instead of having to generate the desired output from scratch)

<img src="https://raw.githubusercontent.com/karenyyy/data_science/master/TF_cv/images/residual-stack.png" width="400">


# Bottleneck Design

** Practical design for going deeper**
- __sandwich 3x3 conv layer with two 1x1 conv layers__
- similar complexity
- better representation
- 1x1 conv reduce tensor dimensonality for 3x3 conv layer

<img src="https://raw.githubusercontent.com/karenyyy/data_science/master/TF_cv/images/bottleneck.png" width="700">


# ResNet Provides Improvements in 3 Key Concepts

**Representation**
- training of much deeper networks
- larger feature space allows for better representation

**Optimization**
- Enable very smooth forward propagation of signal and backward propagation of error
- Greatly	ease	optimizing	deeper models

**Generalization**
- Does not overfit on training data
- Maintains good performance on test data

# ResNet Improvement: Pre-Activation

Improvements on ResNet design: [Identity Mappings in Deep Residual Networks](https://arxiv.org/pdf/1603.05027.pdf)

<img src="https://raw.githubusercontent.com/karenyyy/data_science/master/TF_cv/images/pre-activation.png" width="400">

- ReLU	could	block	prop	when	there	are	1000	layers
- __pre-activation	design	eases	optimization	and	improves	generalization__


<img src="https://raw.githubusercontent.com/karenyyy/data_science/master/TF_cv/images/pre-activation-results.png" width="600">

