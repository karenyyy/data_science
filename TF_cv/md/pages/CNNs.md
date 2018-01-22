
# CNN

- Each neuron receives some inputs, performs a dot product and optionally follows it with a non-linearity function. 
- The whole network represents **a single differentiable score function.**
- This function takes in as input raw image pixels on one end and computes class scores at the other. 
- The output scores are fed into a loss function to compute the multinomial class probabilities.



### Main components

- inputs matrix - image 4D tensor
- output class scores
- convolutional layers
- fully-connected layers
- activation functions
- max-pooling layers
- dropout layers
- softmax layer for multinomial class probabilities
- loss function




<img src="https://raw.githubusercontent.com/karenyyy/data_science/master/TF_cv/images/convnet.jpeg" width="800">



## Fully Connected (Dense) Layer (weighted sum)

<div style="float:right;margin-right:5px;">
    <img src="https://raw.githubusercontent.com/karenyyy/data_science/master/TF_cv/images/SingleNeuron.png" width="300" />
    <p style="text-align:center;">*Single feedforward neuron*</p>
</div>
<br>
**Feedforward computation**

$$\textstyle h_{W,b}(x) = f(W^Tx) = f(\sum_{i=1}^3 W_{i}x_i +b)$$ <br>

f = activation function <br>
W = weight vector/matrix <br>
b = bias scalar/vector <br>


**Fully connected layers are not spatially located since there is no weight sharing. Therefore the input to a fully connect layer must be reshaped to a vector.**


### Fully Connected Neural Network
<br>
<img src="https://raw.githubusercontent.com/karenyyy/data_science/master/TF_cv/images/NN1.gif" width="500">

## Convolutional Layer

#### CNN components
   - learnable filters (or kernels)

       - have a small receptive field
       - extend through the full depth of the input volume
       - during the forward pass, each filter is convolved across the width and height of the input volume, computing the dot product between the entries of the filter and the input and producing a 2-dimensional activation map of that filter.
       - as a result, the network learns filters that activate when it detects some specific type of feature at some spatial position in the input.
       - stacking the activation maps for all filters along the depth dimension forms the full output volume of the convolution layer.


<br>
<div style="float:left;margin-right:5px;">
    <img src="https://raw.githubusercontent.com/karenyyy/data_science/master/TF_cv/images/Conv3.jpeg" width="300" />
    <p style="text-align:center;">*2D Convolution on color image*</p>
</div>
<div style="float:center;margin-right:5px;">
    <img src="https://raw.githubusercontent.com/karenyyy/data_science/master/TF_cv/images/neuron_model.jpeg" width="350" />
    <p style="text-align:center;">*A Neural Network "neuron"*</p>
</div>



#### Benefits of CNN:

1. **Location Invariance** -  because of the sliding filters, **the exact location of important features is not important**, which allows the model to generalize better to unseen images (pooling also provides invariance)

2. **Local connectivity** - Convolutional networks exploit spatially local correlation by **enforcing a local connectivity pattern (receptive field) between neurons of adjacent layers.** This is in contrast to fully connected layers that do not take into account the spatial structure of the input.

3. **Compositionality** -  CNN layers are generally stacked on top of eachother. Allowing the model to construct incrementally higher-level representation of the image, making the classification task easier at the last layer.


<img src="https://raw.githubusercontent.com/karenyyy/data_science/master/TF_cv/images/Convolution.gif" width="400">

## Activation Layer

- (preferable, faster, generalization accuracy) **ReLu** (Rectified Linear Unit) $$f(x)=\max(0,x)$$

- tanh $$f(x)=\tanh(x)$$ 

- sigmoid $$f(x)=(1+e^{-x})^{-1}$$


<img src="https://raw.githubusercontent.com/karenyyy/data_science/master/TF_cv/images/activations.png" width="600">

## Pooling Layer (downsampling)

- Max-pooling 
- Average pooling
- L2-norm pooling.

### Intuitive reasoning behind this layer

- Once we know that a specific feature is in the original input volume (there will be a high activation value), **its exact location is not as important as its relative location to the other features.** 
- This layer drastically reduces the spatial dimension (the length and the width change but not the depth) of the input tensor.

    - 1. Reduce the amount of parameters and computation in the network
    - 2. Control overfitting (high train accuracy but low test accuracy)


<br>
<div style="float:left;margin-right:5px;">
    <img src="https://raw.githubusercontent.com/karenyyy/data_science/master/TF_cv/images/pool.jpeg" width="300" />
    <p style="text-align:center;">*Spatial downsampling with filter size 2, stride 2*</p>
</div>
<div style="float:center;margin-right:5px;">
    <img src="https://raw.githubusercontent.com/karenyyy/data_science/master/TF_cv/images/maxpool.jpeg" width="400" />
    <p style="text-align:center;">*Maxpooling operation*</p>
</div>
 

## Dropout Layer

<img src="https://raw.githubusercontent.com/karenyyy/data_science/master/TF_cv/images/dropout1.png" width="600">

Overly complex models can lead to the problem of overfitting. 

The solution is to apply regularization. 

   - Add the L2-norm of the model's weights to the cost function
        - penalize peaky weight vectors 
        - prefer diffuse weight vectors
        - force the network to use all of its inputs a little rather that some of its inputs a lot.



   - Dropout (complements the other regularization methods)
        - this layer “drops out” a random set of activations in that layer **by setting them to zero in the forward pass*.  
        - it forces the network to provide the right classification or output for a specific example even if some of the activations are dropped out. 
        - **during testing there is no dropout applied**
        
       