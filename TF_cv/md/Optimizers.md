
# Optimizers



## Basic Gradient Descent
- minimize an objective function $J(\theta)$ parameterized by a model's parameters $\theta \in \mathbb{R}^d$ **by updating the parameters in the opposite direction of the gradient of the objective function** $\nabla_\theta J(\theta)$. 
- **The learning rate $\eta$ determines the size of the steps to reach a (local) minimum.** In other words, we follow **the direction of the slope** of the surface created by the objective function downhill until we reach a valley.

<img src="https://raw.githubusercontent.com/karenyyy/data_science/master/TF_cv/images/grad.png" width="600">

TensorFlow train module **`tf.train`**

    tf.train.GradientDescentOptimizer
    tf.train.MomentumOptimizer
    tf.train.RMSPropOptimizer
    tf.train.AdadeltaOptimizer
    tf.train.AdagradOptimizer
    tf.train.AdamOptimizer
    tf.train.AdagradDAOptimizer
    
TF-Keras optimizer module **`tf.contrib.keras.optimizers`**

    tf_keras.optimizers.SGD
    tf_keras.optimizers.Adadelta
    tf_keras.optimizers.Adagrad
    tf_keras.optimizers.RMSprop
    tf_keras.optimizers.Adamax
    tf_keras.optimizers.Nadam
    tf_keras.optimizers.Optimizer

## Gradient Descent Variants


$$\theta = \theta - \eta \cdot \nabla_\theta J( \theta; x; y)$$

- **Batch gradient descent**
    - computes the gradient of the cost function to the parameters θ for the entire training dataset.
- **Stochastic gradient descent**
    - computes a parameter update for each training example x and label y
- **Mini-batch gradient descent**
    - computes an update for every mini-batch of nn training examples
    
Mini-batch gradient descent is typically the algorithm of choice

1. updates have lower variances than  stochastic gradient descent, the model converges better.
2. the dataset doesn't need to fit in memory like for batch gradient descent
3. gradient computation step is faster, because
    - the batch is smaller
    - make use of highly optimized matrix multiplication operation common to state-of-the-art deep learning libraries.
    
Short Comings of Mini-Batch Gradient Descent

<img src="https://raw.githubusercontent.com/karenyyy/data_science/master/TF_cv/images/learning_rates.jpg" width="400">

- **Learning rate magnitude**
    - learning rate too large:
        - loss function fluctuates around the minimum preventing convergence
    - learning rate too small: 
        - training takes too long to reach convergence

- **Challenge of minimizing highly non-convex error functions**
    - escaping suboptimal local minima or saddle points (gradients close to zero in most dimensions) can be difficult


```python
# model Mini-Batch gradient descent optimizer
# Tensorflow optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate)

# TF-Keras optimzer
optimizer = tf_keras.optimizers.SGD(lr = learning_rate)
```

## Momentum Optimizer

<b>SGD tends to oscillate when the loss surface curves much more steeply in one dimension than in another, which are common around local optima.</b>

<img src="https://raw.githubusercontent.com/karenyyy/data_science/master/TF_cv/images/momentum.png" width="600">

Momentum : almost always enjoys better converge rates on deep networks. 
- In a sense the Momentum optimizer gives potential energy to the update step. 
    - allows the parameter vector to build up velocity in any direction that has **consistent gradient**. Eg: pushing a ball down a hill. The ball accumulates momentum as it rolls downhill, becoming faster and faster on the way.
- As a result, we gain faster convergence and reduced oscillation.

Update step:

$$v_t = \gamma v_{t-1} + \eta \nabla_\theta J( \theta)$$
$$\theta = \theta - v_t$$


**The momentum term (friction term) $\gamma$ is usually set to 0.9 or a similar value.**



```python
# Momentum Optimizers

# Tensorflow optimizer
optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=momentum)

# TF-Keras optimizer
optimizer = tf_keras.optimizers.SGD(lr=learning_rate, momentum=momentum)
```

## Nesterov Accelerated Gradient (NAG) Optimizer
 

<img src="https://raw.githubusercontent.com/karenyyy/data_science/master/TF_cv/images/nesterov.jpeg" width="800">

Instead of evaluating gradient at the current position (red circle), we know that *our momentum is about to carry us to the tip of the green arrow*. **With Nesterov momentum we therefore instead evaluate the gradient at this "looked-ahead" position.**

$$v_t = \gamma v_{t-1} + \eta \nabla_\theta J( \theta - \gamma v_{t-1} )$$
$$\theta = \theta - v_t$$

### This anticipatory update prevents us from going too fast (overshooting), which has significantly increased the performance of RNNs on a number of tasks.


```python
# Nesterov Optimizers

# Tensorflow optimizer
optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=momentum, use_nesterov=True)

# TF-Keras optimizer
optimizer = tf_keras.optimizers.SGD(lr=learning_rate, momentum=momentum, nesterov=True)
```

## Adagrad Optimizer

In Adagrad the **learning rate is normalized by the sum of squared gradients on a per-parameter basis.** 
   - well-suited for dealing with sparse data:
       - larger updates for infrequent parameters
       - smaller updates for frequent parameters
    



```python
# Simplified update step:

# Assume the gradient dx and parameter vector x
cache += dx**2
x += - learning_rate * dx / (np.sqrt(cache) + eps)

```


```python
# Adagrad Optimizers

# Tensorflow optimizer
optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate)


# TF-keras optimizer
optimizer = tf_keras.optimizers.Adagrad(lr=learning_rate)
```

## AdaDelta and RMSprop

AdaDelta and RMSprop are both very similar optimizers that improves on Adagrad;

- by **normalizing the learning rate** with **the moving average of squared gradients**
- this reduces the aggressive, monotonically decreasing learning rate found in Adagrad.



```python
#Simplified update step:

cache = decay_rate * cache + (1 - decay_rate) * dx**2
x += - learning_rate * dx / (np.sqrt(cache) + eps)
```


```python
# Adadelta
optimizer = tf.train.AdadeltaOptimizer(learning_rate=learning_rate)

# RMSprop
optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)


# TF-keras optimizer

# Adadelta
optimizer = tf_keras.optimizers.Adadelta(lr=learning_rate)

# RMSprop
optimizer = tf_keras.optimizers.RMSprop(lr=learning_rate)
```

## Adaptive Moment Estimation (Adam) Optimizer
### the most recommended optimizer

- **It combines idea from both RMSProp the Momentum method.** 
- It generates a "smooth" estimate (exponentially decaying average) of the gradients' mean and variance. 
- Giving you the best a less noisy gradient and an adaptive learning rates for each parameter.


```python
# Simplified update step:

m = beta1*m + (1-beta1)*dx
v = beta2*v + (1-beta2)*(dx**2)
x += - learning_rate * m / (np.sqrt(v) + eps)
```
