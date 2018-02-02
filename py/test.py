from sklearn.feature_extraction import DictVectorizer
data = [
    {'price': 850000, 'rooms': 4, 'neighborhood': 'Queen Anne'},
    {'price': 700000, 'rooms': 3, 'neighborhood': 'Fremont'},
    {'price': 650000, 'rooms': 3, 'neighborhood': 'Wallingford'},
    {'price': 600000, 'rooms': 2, 'neighborhood': 'Fremont'}
]

vec=DictVectorizer(sparse=True, dtype=int)
c=vec.fit_transform(data)
print(c)

import tensorflow as tf

from tensorflow.contrib import rnn

import numpy as np

x=np.random.randint(1, 10, 4, dtype=int)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

model=make_pipeline(TfidfVectorizer(), MultinomialNB())
model.fit(t)

from sklearn.metrics import confusion_matrix
confusion_matrix(test.ta)

import seaborn as sns

sns.pairplot()