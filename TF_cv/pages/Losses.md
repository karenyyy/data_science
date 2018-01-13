
# TensorFlow Loss Functions

**Losses Functions**


TensorFlow loss module **`tf.losses`**
    
    tf.losses.sigmoid_cross_entropy
    tf.losses.softmax_cross_entropy
    tf.losses.sparse_softmax_cross_entropy
    tf.losses.cosine_distance
    tf.losses.hinge_loss
    tf.losses.log_loss
    tf.losses.mean_squared_error
    tf.losses.mean_pairwise_squared_error
    
TF-Keras loss module **`tf.contrib.keras.losses`**

    tf_keras_losses.mean_squared_error
    tf_keras_losses.mean_absolute_error
    tf_keras_losses.binary_crossentropy
    tf_keras_losses.categorical_crossentropy
    tf_keras_losses.sparse_categorical_crossentropy
    tf_keras_losses.cosine_proximity
    tf_keras_losses.hinge
    tf_keras_losses.squared_hinge

All of the loss functions take a pair of predictions and ground truth labels, from which the loss is computed.

$$LossScore = Loss(y_{true}, y_{pred})$$ 

## Binary Classification


**Sigmoid cross-entropy loss** measures the probability error of the correct class.

**Sigmoid function** or logistic function only ouputs a single value (between 0 and 1), which represent the prediction probability of the positive class. It is often used as an activation function with saturation points at both extremes. It is also used for binary classification.

<img src="../../images/sigmoid.jpg" width="200">

$$sigmoid(z)=\frac {1}{1+e^{-z_j}}$$

**Sigmoid function** can be thought of has a special case of softmax where the number of class equals to 2.  Sigmoid functions takes in a single output neuron and the prediction is defined by an arbitrary threshold (often 0.5).

**Cross Entropy Loss** for binary classification is equivalent to the negative log-likelihood of the true labels given a probabilistic classifier’s predictions. It is also refered as **log-loss**.

$$sigmoidCrossEntropyLoss(z)=-\log\left(\frac {1}{1+e^{-z_j}}\right)$$

Both core TensorFlow and TF-Keras provide their version of a binary loss:


```python
# your binary model predictions
class_score = BinaryClassificationModel(inputs)

# TensorFlow Core loss
cross_entropy_loss = tf.losses.sigmoid_cross_entropy(binary_labels, class_score)
 
# Or TF-Keras loss
cross_entropy_loss = tf_keras_losses.binary_crossentropy(binary_labels, class_score)

# Or custom TensorFlow operation
class_probability = tf.nn.sigmoid(class_score)
cross_entropy_loss = - tf.losses.log_loss(binary_labels, class_probability)

# model's loss plus any regularization losses.
total_loss = tf.losses.get_total_loss(add_regularization_losses=True)
```

# Multi-Class Classification


**Softmax Cross-Entropy loss** measures the probability error for discrete multi-classification tasks.

A common classfier choice for multi-class task is the **Softmax classifier**.
The softmax classifier is a generalization of a binary classifier to multiple classes.
The softmax classifier has an interpretable output of normalized class probabilities.



**Softmax function**
$$softmax(z) = \frac{e^{z_j}}{\sum_k e^{z_k}}$$

**Softmax function** takes a vector of arbitrary real-valued scores and squashes it to a vector of values between zero and one that sum to one. Therefore, it guarantees that the sum of all class probabilities is 1.That's why it's used for multi-class classification **because you expect your samples to belong to a single class at the time.**



**Cross-Entropy Loss**
$$CrossEntropyLoss = -\log\left(\frac{e^{z_j}}{\sum_k e^{z_k}}\right)$$

The Softmax classifier minimizes the **cross-entropy between the estimated class probabilities and the “true” distribution**, which can be a **soft class distribution or a one-hot encoding of the target labels**.

If the true distribution is **one-hot encoding** then the loss simplifies to the **sparse cross-entropy** because the probability of a given label is considered exclusive.

This loss is **equivalent to minimizing the KL divergence** (distance) between the two distributions.

$$H(p,q) = - \sum_x p(x) \log q(x)$$




**Probabilistic interpretation**
given the image $x_i$ and parameterized by $W$, softmax classifier gives the probability assigned to the correct label $y_i$ .

$$P(y_i \mid x_i; W) = \frac{e^{f_{y_i}}}{\sum_j e^{f_j} }$$

We are therefore minimizing the negative log likelihood of the correct class, which can be interpreted as performing **Maximum Likelihood Estimation (MLE)**. With added L2 regularization (which equates to a Gaussian prior over the weight matrix $W$), we are instead performing the **Maximum a posteriori (MAP)** estimation.


```python
# your multi-class model predictions
class_scores = MultiClassificationModel(inputs) # vector of class scores

# TensorFlow Core loss
cross_entropy_loss = tf.losses.softmax_cross_entropy(one_hot_labels, class_scores)

# sparse implementation
cross_entropy_loss = tf.losses.sparse_softmax_cross_entropy(one_hot_labels, class_scores)

# TF-Keras loss
cross_entropy_loss = tf_keras_losses.categorical_crossentropy(one_hot_labels, class_scores)

# sparse implementation
cross_entropy_loss = tf_keras_losses.sparse_categorical_crossentropy(one_hot_labels, class_scores)

# Or custom TensorFlow operation
class_probabilities = tf.nn.softmax(class_scores)
cross_entropy_loss = tf.losses.log_loss(one_hot_labels, class_probabilities)

# model's loss plus any regularization losses.
total_loss = tf.losses.get_total_loss(add_regularization_losses=True)
```

## Multi-Label Classification

**Multi-Label Classification is when a sample observation can belong to multiple classes at the same time**.

We can rephrase multi-label learning as the problem of finding a model that **maps inputs x to binary vectors y, rather than scalar outputs as in the ordinary classification problem.**
With this interpretation, the solution is to apply an independent sigmoid function for each label.


**Multi-Label Sigmoid Cross-Entropy** measures the probability error in discrete classification tasks in which each class is independent and not mutually exclusive.


```python
# your multi-label model predictions
class_scores = MultiLabelClassificationModel(inputs) # vector of class scores

# TensorFlow core loss
loss = tf.losses.sigmoid_cross_entropy(multi_class_labels, class_scores)

# model's loss plus any regularization losses.
total_loss = tf.losses.get_total_loss(add_regularization_losses=True)
```

## Hinge Loss
**Hinge Loss is used in Multiclass Support Vector Machine (SVM) loss. The SVM loss is set up so that the SVM “wants” the correct class for each image to a have a score higher than the incorrect classes by some fixed margin $\Delta$.**

For the score function $s_j = f(x_i, W)_j$. The Multiclass SVM loss for the i-th example is then formalized as follows:

$$HingeLoss = \sum_{j\neq y_i} \max(0, s_j - s_{y_i} + \Delta)$$

**Squared Hinge Loss** SVM (or L2-SVM), which simply squares the margin error. It penalizes the violated margins more strongly (quadratically instead of linearly). The unsquared version is more standard, but in some datasets the squared hinge loss can work better. This can be determined during cross-validation.

$$SquaredHingeLoss = \sum_{j\neq y_i} \max(0, s_j - s_{y_i} + \Delta)^2$$


```python
# your multi-class model predictions
class_scores = MultiClassificationModel(inputs) # vector of class scores

# TensorFlow core loss
loss = tf.losses.hinge_loss(one_hot_labels, class_scores)

# TF-Keras loss
loss = tf_keras_losses.hinge(one_hot_labels, class_scores)

# TF-Keras square hinge loss
loss = tf_keras_losses.squared_hinge(one_hot_labels, class_scores)

# model's loss plus any regularization losses.
total_loss = tf.losses.get_total_loss(add_regularization_losses=True)
```

## Regression Loss

**Not all predictive tasks involve outputing distinct labels, sometimes need to predict a continuous variable**

**Mean squared error (MSE)** measures the average of the squares of the errors or deviations. In the context of regression analysis, it measures the quality of an estimator—it is always non-negative, and values closer to zero are better.


$$\operatorname {MSE}={\frac{1}{n}}\sum_{{i=1}}^{n}({\hat{Y_{i}}}-Y_{i})^{2}$$


**Mean absolute error (MAE)** measures the average absolute errors. This loss is used to measure how close forecasts or predictions are to the eventual outcomes.


$$\operatorname {MAE}={\frac{1}{n}}\sum_{{i=1}}^{n}|{\hat{Y_{i}}}-Y_{i}|$$


**Mean pairwise squared error(MPSE)** Unlike `mean_squared_error`, which is a measure of the differences between
corresponding elements of `predictions` and `labels`,
`mean_pairwise_squared_error` is a measure of the differences between pairs of
corresponding elements of `predictions` and `labels`.

For example:
if `labels = [a, b, c]` and `predictions = [x, y, z]`, there are
three pairs of differences are summed to compute the loss:
 
$$\operatorname {MPSE} = \frac{((a-b) - (x-y))^2 + ((a-c) - (x-z))^2 + ((b-c) - (y-z))^2}{3}$$


```python
# your multi-class model predictions
y_pred = RegressionModel(inputs)

# TensorFlow core loss
loss = tf.losses.mean_squared_error(y_true, y_pred)

# TF-Keras loss
loss = tf_keras_losses.mean_squared_error(y_true, y_pred)

# Mean absolute error
loss = tf_keras_losses.mean_absolute_error(y_true, y_pred)

# Main pairwise squared error
loss = tf.losses.mean_pairwise_squared_error(y_true, y_pred)

# model's loss plus any regularization losses.
total_loss = tf.losses.get_total_loss(add_regularization_losses=True)
```
