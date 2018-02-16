import matplotlib
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

diabetes = pd.read_csv("pima-indians-diabetes.csv", names=(
    "Pregnant", "Plasma_Glucose", "Dias_BP", "Triceps_Skin", "Serum_Insulin", "BMI", "DPF", "Age", "Diabetes"))
print(diabetes.head())

cols_to_norm = ["Pregnant", "Plasma_Glucose", "Dias_BP", "Triceps_Skin", "Serum_Insulin", "BMI", "DPF",
                "Diabetes"]

diabetes[cols_to_norm] = diabetes[cols_to_norm].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
print(diabetes.head())

num_preg = tf.feature_column.numeric_column("Pregnant")
plasma = tf.feature_column.numeric_column("Plasma_Glucose")
dias_bp = tf.feature_column.numeric_column("Dias_BP")
skin = tf.feature_column.numeric_column("Triceps_Skin")
insulin = tf.feature_column.numeric_column("Serum_Insulin")
bmi = tf.feature_column.numeric_column("BMI")
dpf = tf.feature_column.numeric_column("DPF")
age = tf.feature_column.numeric_column("Age")

# assigned_group=tf.feature_column.categorical_column_with_vocabulary_list("Group", ['A','B','C','D'])
#assigned_group = tf.feature_column.categorical_column_with_hash_bucket("Group", hash_bucket_size=10)
diabetes["Age"].hist(bins=20)
# plt.show()

age_bucket = tf.feature_column.bucketized_column(age, boundaries=[20, 30, 40, 50, 60, 70, 80])

feat_cols = [num_preg, plasma, dias_bp, skin, insulin, bmi, dpf, age_bucket]

y_label = diabetes["Diabetes"]
print(y_label)

x_data = diabetes.drop("Diabetes", axis=1)
print(x_data.head())

from sklearn.cross_validation import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x_data, y_label, test_size=0.3, random_state=2018)

# input_func = tf.estimator.inputs.pandas_input_fn(x=x_train,
#                                                  y=y_train,
#                                                  batch_size=8,
#                                                  num_epochs=None,
#                                                  shuffle=True)
train_func = tf.estimator.inputs.pandas_input_fn(x=x_train,
                                                 y=y_train,
                                                 batch_size=8,
                                                 num_epochs=1000,
                                                 shuffle=False)
eval_func = tf.estimator.inputs.pandas_input_fn(x=x_test,
                                                y=y_test,
                                                batch_size=8,
                                                num_epochs=1,
                                                shuffle=False)

model=tf.estimator.LinearClassifier(feature_columns=feat_cols, n_classes=2)
model.train(input_fn=train_func, steps=1000)

eval_metric=model.evaluate(input_fn=eval_func)
print(eval_metric)


estimated=model.predict(input_fn=eval_func)
print(list(estimated))


dnn_model=tf.estimator.DNNClassifier(hidden_units=[10, 10, 10], feature_columns=feat_cols, n_classes=2)
dnn_model.train(input_fn=train_func, steps=1000)