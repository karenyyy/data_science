
# Evaluation Metrics


    A summary of the most common evaluation metrics:

<img src="images/metrics.png" width="1000">


TensorFlow metrics module **`tf.metrics`**

    tf.metrics.accuracy
    tf.metrics.auc
    tf.metrics.precision
    tf.metrics.recall
    tf.metrics.recall_at_k
    tf.metrics.true_positives
    tf.metrics.false_negatives
    tf.metrics.false_positives
    tf.metrics.mean_per_class_accuracy
    tf.metrics.precision_at_thresholds
    tf.metrics.recall_at_thresholds
    
TF-Keras metrics module **`tf.contrib.keras.metrics`**

    tf_keras_metrics.binary_accuracy
    tf_keras_metrics.categorical_accuracy
    tf_keras_metrics.top_k_categorical_accuracy

# Accuracy and Precision

**Precision** is a description of **random errors**, a measure of **statistical variability**. In other words, the closeness of two or more measurements to each other.

**Accuracy** is a description of **systematic errors**, a measure of **statistical bias**. In other words, the closeness of a measured value to the true value.


<i>**Precision and Accuracy** are related to the **Bias-variance trade-off** found in Machine Learning models.</i>


**High variance** can be caused by the model **overfitting** (low precision)

*Solution:*
- add regularization to the model
- collect more data
- decrease model expressiveness (complexity)
- bagging (Bootstrap Aggregating) or other resampling techniques (random forest) 

**High bias** can be caused by **under-fitting** the model. (low accuracy)

*Solution:*
- increase model expressiveness (complexity)
- collect more data


```python
# model predictions
predictions = DeepLearningModel(inputs)

# TensorFlow core accuracy metric
accuracy = tf.metrics.accuracy(labels, predictions)

# TF-Keras accuracy metrics
accuracy = tf_keras_metrics.binary_accuracy(labels, predictions)

# multi-class accuracy
accuracy = tf_keras_metrics.categorical_accuracy(labels, predictions)

# Top-K accuracy
accuracy = tf_keras_metrics.top_k_categorical_accuracy(labels, predictions)

# mean per class accuracy
accuracy = tf.metrics.mean_per_class_accuracy(labels, predictions)
```

# Recall and F-1 Score

<img src="images/recall.png" width="200">

$$Recall = \frac{TruePositive}{PositiveSamples} = \frac{TruePositive}{TruePositive + FalseNegative} $$


```python
# model predictions
predictions = DeepLearningModel(inputs)

# TensorFlow core recall metric
recall = tf.metrics.recall(labels, predictions)

# Recall at k
recall = tf.metrics.recall_at_k
```

# Custom Evaluation Metrics

**F1 score is the harmonic mean of precision and recall**

$${\displaystyle F_{1}=2\cdot {\frac {1}{{\tfrac {1}{\mathrm {recall} }}+{\tfrac {1}{\mathrm {precision} }}}}=2\cdot {\frac {\mathrm {precision} \cdot \mathrm {recall} }{\mathrm {precision} +\mathrm {recall} }}}$$


```python
# F1 score metric
def F1_score(labels, predictions):
    precision = tf.metrics.precision(labels, predictions)
    recall = tf.metrics.recall(labels, predictions)
    return 2 * tf.multiply(precision, recall) / tf.add(precision, recall)

# recall metric
def recall(labels, predictions):
    TP = tf.metrics.true_positives(labels, predictions)
    FN = tf.metrics.false_negatives(labels, predictions)
    return TP / tf.add(TP,FN)
```

# ROC AUC

- An **ROC curve** is the most commonly used way to visualize the performance of a **binary classifier**. The curve is created by plotting the **true positive rate (TPR**) against the **false positive rate (FPR)** at various threshold settings. 

- **AUC** is a good way summarize the **classifier's performance** in a single number. This number is between 0.5 and 1. The **Area Under the Curve (AUC)** is literally just the percentage of the box that is under the curve. This metric quantifies the performance of a classifier into one number for model comparaison.


<img src="images/roc1.png" width="300">


- think of AUC as representing the **probability that a classifier will rank a randomly chosen positive observation higher than a randomly chosen negative observation**, and thus it is a useful metric even for datasets with **highly unbalanced classes**.


```python
# model predictions
predictions = DeepLearningModel(inputs)

# TensorFlow core AUC metric
auc = tf.metrics.auc(labels, predictions)
```

# TensorFlow Streaming Metrics -defining Multiple Metrics

**Dictionary Aggregation**



```python
# model predictions
predictions = DeepLearningModel(inputs)


# Aggregates the value and update ops into dictionary:
names_to_values, names_to_updates = tf.contrib.slim.metrics.aggregate_metric_map({
    'eval/Accuracy': tf.metrics.accuracy(labels, predictions),
    'eval/Precision': tf.metrics.precision(labels, predictions),
    'eval/Recall': tf.metrics.recall(labels, predictions)
})


# Evaluate the model using 1000 batches of data:
num_batches = 1000

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    # run metrics over multiple batches
    for batch_id in range(num_batches):
        sess.run(names_to_updates.values())

    # Get each metric end value
    metric_values = sess.run(name_to_values.values())
    for metric, value in zip(names_to_values.keys(), metric_values):
        print('Metric %s has value: %f' % (metric, value))
```
