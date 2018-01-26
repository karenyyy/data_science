
# How HDBSCAN Works

HDBSCAN is a clustering algorithm developed by [Campello, Moulavi, and Sander](http://link.springer.com/chapter/10.1007%2F978-3-642-37456-2_14). 

`It extends DBSCAN by converting it into a hierarchical clustering algorithm`, and then using a technique to `extract a flat clustering based in the stability of clusters.`

### First a quick mention of DBSCAN


```python
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler

sns.set_context('poster')
sns.set_style('white')
sns.set_color_codes()
plt.rcParams["figure.figsize"] = (8,7)
plot_kwds = {'alpha' : 0.5, 's' : 80, 'linewidths':0}
```


```python
# #############################################################################
# Generate sample data
centers = [[1, 1], [-1, -1], [1, -1], [-1, 1]]
X, labels_true = make_blobs(n_samples=750, centers=centers, cluster_std=0.4,
                            random_state=0)

X = StandardScaler().fit_transform(X)
X.shape
```




    (750, 2)




```python
plt.hist(X[:,0])
```




    (array([  17.,   69.,  158.,  105.,   27.,   26.,  134.,  155.,   49.,   10.]),
     array([-2.02354988, -1.61121034, -1.19887079, -0.78653124, -0.3741917 ,
             0.03814785,  0.4504874 ,  0.86282694,  1.27516649,  1.68750603,
             2.09984558]),
     <a list of 10 Patch objects>)




![png](output_4_1.png)



```python
plt.hist(X[:,1])
```




    (array([   9.,   63.,  148.,  115.,   37.,   16.,   84.,  165.,   91.,   22.]),
     array([-2.02639932, -1.63402909, -1.24165886, -0.84928863, -0.4569184 ,
            -0.06454817,  0.32782207,  0.7201923 ,  1.11256253,  1.50493276,
             1.89730299]),
     <a list of 10 Patch objects>)




![png](output_5_1.png)



```python
# #############################################################################
# Compute DBSCAN
db = DBSCAN(eps=0.3, min_samples=10).fit(X)
db.labels_
```




    array([ 0,  0,  0,  0,  0,  1,  1,  0,  0,  0,  0,  1,  0,  0,  0,  0, -1,
            1, -1,  0,  0,  0,  0,  0,  1,  1,  1,  0,  0,  1,  0,  0,  0,  0,
            0,  0,  0,  0,  0, -1,  1,  1,  0,  1,  0,  1,  0,  0,  0,  1,  0,
            1,  1,  0,  0,  0,  1,  0,  1,  1,  1,  1,  1,  0,  0,  0,  0,  1,
            0,  0,  0,  0,  0,  0,  0,  0, -1,  0,  0,  0,  0,  0,  0,  0,  0,
            0,  0,  1,  0,  0,  1,  0,  1,  0,  1, -1,  1,  1,  0,  0,  0,  1,
            0,  0,  0,  1,  0,  0, -1,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,
            1,  0,  1,  0,  0,  1,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0, -1,
            1,  0,  0,  1,  0,  1,  0,  0,  0,  0, -1,  1,  1, -1,  0,  1,  1,
            0,  0,  0, -1,  1,  0,  1,  0,  0,  0,  0,  0, -1,  0,  1,  1,  0,
            0,  0,  0,  1,  1,  0,  0,  1,  0,  0,  1,  0,  0,  0,  0,  0,  0,
            0,  0,  1,  0,  0,  0,  1,  0,  0,  0,  1,  0,  1,  1,  0,  0,  1,
            1,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  1,  0,  1,  0,  0,
            0,  0,  1,  1,  0,  0,  0,  1,  0,  0,  0,  1,  0,  0,  1,  0,  0,
            0,  0,  0,  0,  0,  1,  1,  0,  0,  1,  0,  0,  0,  0,  1,  0,  1,
            0,  0,  0,  0,  1,  0,  1,  1,  1,  1,  0,  0,  1,  0,  0,  0,  0,
            0, -1,  0,  0,  0,  0,  0,  0,  0,  0, -1,  0,  0,  1,  0,  0,  1,
            0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
            0,  0,  1,  0,  0,  0,  1,  1,  1,  1,  1,  0,  0,  1,  1,  1,  0,
            0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,
            0,  0,  0,  0,  0,  0,  0,  1,  0,  1,  0,  0,  0,  0,  0,  0,  0,
            0,  0,  0,  0,  1,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  1,  0,
            0,  0,  0,  0,  0,  0,  1,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,
            1,  1,  0,  0,  0,  0,  0,  1,  1,  0,  0,  1,  0,  0,  1,  0,  0,
            0,  0,  1,  0,  1,  0, -1,  1,  0,  0,  0,  1,  0,  1,  0,  0,  0,
            0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  1,
            0, -1,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  1,  1,
            0,  0, -1,  0,  1,  0,  1,  1,  0,  0,  0,  0,  0,  0,  0,  0,  1,
            0,  0,  0,  0,  1,  0,  1,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,
            1,  1,  0, -1,  0,  1,  1,  0,  0,  0,  0,  0,  1,  0,  0,  1,  0,
            0,  0,  1,  0,  0,  0,  0,  1,  1,  0,  1,  1,  0,  1,  0,  1,  0,
            0,  0,  0,  0, -1,  1,  1,  1,  1,  0,  0,  0,  0,  0,  0,  0,  1,
            0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  1,  0,  1,  1,  0,  0,
            1,  1,  1,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,
            0,  0,  0,  1,  0,  0,  1,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,
            1,  0,  0,  1,  1,  0,  0,  1,  0,  0,  0,  0,  1,  0,  0,  1,  0,
            0,  1,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,
            0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  1,  0,  0,  0,  0,  0,  0,
            0,  0,  0,  0,  0,  0,  0,  1,  0,  0, -1, -1,  1,  0,  0,  0,  0,
           -1,  0,  0,  0,  1,  1,  0,  1,  0,  0,  0,  1,  0,  0,  1,  0,  1,
            0,  0,  0,  1,  1,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,
            0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  1,  0,  1,  1,  1,  0,  1,
            0,  0,  0,  1,  0,  0,  0,  0,  1, -1,  0,  0,  0,  0,  0,  0,  0,
            0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  1,  0,
            0,  0])




```python
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
core_samples_mask
```




    array([ True,  True,  True,  True,  True, False,  True,  True,  True,
            True,  True,  True,  True,  True, False,  True, False,  True,
           False,  True,  True,  True,  True,  True,  True,  True,  True,
            True,  True,  True,  True,  True, False,  True,  True,  True,
            True,  True,  True, False,  True,  True,  True,  True,  True,
            True,  True,  True,  True,  True,  True,  True, False,  True,
            True,  True,  True,  True,  True,  True,  True,  True,  True,
            True,  True,  True,  True,  True,  True,  True,  True,  True,
            True,  True,  True,  True, False, False,  True,  True,  True,
            True, False,  True,  True, False,  True,  True,  True,  True,
            True,  True,  True,  True,  True, False,  True,  True,  True,
            True,  True,  True,  True,  True,  True,  True,  True,  True,
           False,  True, False,  True,  True,  True,  True,  True,  True,
            True,  True,  True,  True,  True,  True,  True,  True,  True,
            True,  True,  True,  True,  True,  True,  True,  True,  True,
           False,  True,  True,  True,  True, False,  True,  True,  True,
            True,  True, False,  True,  True, False,  True,  True,  True,
            True,  True,  True, False,  True, False,  True,  True,  True,
            True,  True,  True, False,  True,  True,  True,  True,  True,
            True,  True,  True,  True,  True,  True,  True, False,  True,
            True,  True,  True,  True,  True,  True,  True,  True,  True,
            True,  True,  True,  True,  True,  True,  True, False,  True,
            True,  True,  True,  True, False,  True,  True,  True,  True,
            True,  True,  True,  True,  True,  True,  True,  True,  True,
            True,  True, False,  True,  True,  True,  True,  True,  True,
            True,  True,  True,  True,  True,  True,  True,  True,  True,
            True,  True,  True,  True,  True,  True,  True,  True,  True,
            True,  True, False,  True,  True,  True,  True,  True,  True,
            True,  True,  True,  True,  True,  True,  True,  True,  True,
            True,  True,  True,  True, False, False,  True,  True,  True,
            True,  True,  True, False,  True,  True,  True,  True,  True,
            True,  True,  True, False,  True,  True,  True, False,  True,
            True, False,  True,  True,  True, False,  True,  True,  True,
            True,  True,  True,  True,  True,  True,  True,  True,  True,
            True,  True,  True,  True,  True,  True,  True,  True,  True,
            True,  True,  True,  True,  True,  True, False,  True,  True,
            True,  True,  True,  True,  True,  True,  True,  True,  True,
            True,  True,  True,  True, False,  True,  True,  True,  True,
           False, False,  True,  True, False,  True,  True,  True,  True,
            True,  True,  True,  True,  True,  True,  True,  True,  True,
            True,  True, False,  True,  True,  True,  True,  True,  True,
            True,  True,  True,  True,  True,  True,  True,  True,  True,
            True,  True,  True,  True,  True,  True,  True,  True,  True,
           False,  True, False,  True,  True,  True,  True,  True,  True,
            True,  True,  True,  True,  True, False,  True,  True,  True,
            True,  True,  True,  True,  True,  True,  True,  True, False,
           False,  True,  True,  True,  True,  True,  True,  True,  True,
            True,  True,  True,  True,  True,  True,  True,  True,  True,
            True,  True,  True,  True,  True,  True, False,  True,  True,
            True,  True, False,  True,  True,  True,  True,  True,  True,
            True,  True,  True,  True, False,  True,  True,  True,  True,
            True,  True, False,  True,  True,  True,  True,  True,  True,
           False,  True,  True,  True,  True,  True,  True,  True,  True,
            True,  True,  True,  True,  True,  True, False,  True,  True,
            True,  True,  True,  True,  True,  True,  True,  True,  True,
            True, False,  True,  True,  True,  True,  True,  True,  True,
            True,  True,  True,  True,  True,  True,  True,  True,  True,
            True,  True, False,  True,  True,  True,  True,  True,  True,
            True,  True,  True,  True,  True,  True,  True,  True,  True,
           False,  True,  True,  True,  True,  True,  True,  True,  True,
            True,  True,  True,  True,  True,  True,  True, False,  True,
            True,  True,  True,  True,  True,  True,  True,  True,  True,
            True,  True,  True,  True, False,  True,  True,  True,  True,
            True,  True,  True, False,  True,  True,  True,  True,  True,
            True,  True,  True,  True,  True,  True,  True,  True,  True,
            True,  True,  True,  True,  True,  True,  True,  True,  True,
            True,  True,  True,  True,  True,  True,  True,  True,  True,
           False,  True,  True,  True,  True,  True,  True,  True,  True,
           False,  True,  True,  True,  True,  True,  True,  True,  True,
            True, False,  True,  True,  True,  True,  True,  True,  True,
           False,  True,  True,  True, False, False,  True, False,  True,
            True,  True,  True,  True,  True,  True,  True,  True,  True,
            True,  True,  True,  True,  True,  True,  True,  True, False,
           False,  True, False,  True,  True,  True, False,  True,  True,
            True,  True,  True,  True,  True, False,  True,  True,  True,
            True,  True,  True,  True, False, False,  True,  True,  True,
           False,  True,  True,  True,  True,  True,  True,  True,  True,
            True,  True,  True,  True,  True,  True,  True,  True,  True,
            True,  True,  True,  True,  True,  True,  True,  True,  True,
            True,  True,  True,  True,  True,  True,  True,  True,  True,
            True, False,  True, False,  True, False,  True,  True,  True,
            True,  True,  True,  True,  True,  True,  True,  True,  True,
            True,  True,  True, False,  True,  True,  True,  True,  True,
           False,  True,  True], dtype=bool)




```python
labels = db.labels_

# # Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_clusters_
```




    2




```python
labels_true
```




    array([0, 1, 0, 3, 0, 2, 2, 3, 0, 0, 1, 2, 1, 3, 1, 0, 1, 2, 3, 3, 3, 3, 3,
           3, 2, 2, 2, 0, 0, 2, 0, 1, 1, 0, 1, 0, 3, 0, 0, 3, 2, 2, 1, 2, 1, 2,
           1, 3, 0, 2, 3, 2, 2, 1, 3, 3, 2, 0, 2, 2, 2, 2, 2, 3, 3, 0, 3, 2, 0,
           0, 1, 3, 0, 1, 3, 1, 0, 1, 0, 3, 1, 1, 1, 0, 0, 0, 1, 2, 1, 3, 2, 0,
           2, 0, 2, 0, 2, 2, 0, 1, 3, 2, 3, 0, 3, 2, 3, 3, 0, 1, 1, 2, 1, 1, 1,
           0, 1, 1, 0, 2, 3, 2, 0, 1, 2, 3, 2, 0, 0, 3, 0, 3, 3, 3, 1, 1, 2, 3,
           0, 2, 1, 2, 1, 0, 3, 3, 3, 2, 2, 0, 3, 2, 2, 3, 3, 3, 0, 2, 0, 2, 0,
           1, 0, 3, 3, 1, 1, 2, 2, 1, 0, 1, 3, 2, 2, 1, 1, 2, 3, 0, 2, 3, 0, 0,
           3, 1, 1, 1, 0, 2, 0, 1, 1, 2, 3, 0, 1, 2, 1, 2, 2, 3, 3, 2, 2, 0, 3,
           0, 3, 3, 0, 3, 3, 2, 1, 1, 2, 1, 2, 3, 3, 3, 3, 2, 2, 3, 0, 1, 2, 0,
           1, 0, 2, 0, 1, 2, 1, 3, 1, 1, 0, 1, 3, 2, 2, 3, 3, 2, 0, 1, 1, 1, 2,
           0, 2, 0, 3, 1, 3, 0, 3, 2, 2, 2, 2, 0, 0, 2, 1, 0, 0, 3, 1, 2, 3, 1,
           1, 3, 1, 3, 0, 3, 3, 0, 1, 2, 3, 0, 2, 3, 0, 1, 3, 1, 3, 0, 3, 1, 1,
           1, 1, 1, 3, 1, 3, 3, 0, 3, 2, 0, 0, 3, 2, 2, 2, 2, 2, 1, 1, 2, 2, 2,
           0, 0, 1, 1, 2, 0, 3, 0, 1, 3, 3, 0, 0, 2, 0, 3, 1, 0, 3, 0, 3, 0, 3,
           3, 0, 2, 0, 2, 0, 3, 3, 1, 1, 1, 3, 1, 3, 0, 3, 2, 3, 3, 0, 2, 1, 1,
           0, 0, 1, 0, 2, 0, 3, 1, 1, 1, 1, 3, 2, 2, 1, 1, 1, 1, 0, 3, 1, 1, 3,
           2, 2, 1, 3, 3, 3, 1, 2, 2, 3, 0, 2, 1, 3, 2, 0, 1, 0, 1, 2, 1, 2, 3,
           1, 2, 0, 0, 3, 2, 3, 2, 3, 3, 1, 0, 0, 1, 0, 2, 0, 3, 1, 0, 1, 3, 0,
           1, 2, 0, 1, 2, 0, 3, 0, 3, 3, 3, 1, 1, 2, 0, 1, 0, 1, 1, 0, 2, 2, 3,
           3, 1, 1, 2, 3, 2, 2, 1, 1, 1, 0, 0, 0, 3, 3, 2, 3, 1, 0, 1, 2, 3, 2,
           0, 0, 3, 0, 1, 0, 2, 1, 0, 3, 2, 2, 1, 3, 0, 2, 2, 1, 0, 3, 0, 0, 2,
           1, 1, 2, 1, 1, 1, 2, 0, 0, 3, 1, 2, 2, 3, 2, 2, 1, 2, 0, 2, 1, 1, 1,
           1, 0, 2, 2, 2, 2, 2, 3, 3, 1, 3, 1, 1, 1, 2, 3, 0, 0, 0, 1, 3, 3, 0,
           3, 0, 2, 2, 0, 2, 2, 0, 0, 2, 2, 2, 2, 3, 1, 3, 1, 1, 2, 3, 0, 0, 3,
           0, 3, 2, 0, 3, 0, 2, 1, 1, 2, 0, 0, 1, 3, 2, 1, 1, 1, 3, 3, 2, 0, 3,
           2, 2, 1, 0, 2, 0, 3, 1, 3, 2, 0, 3, 2, 0, 1, 2, 1, 1, 2, 1, 0, 0, 3,
           0, 3, 1, 0, 1, 2, 0, 0, 1, 3, 3, 1, 0, 0, 0, 2, 1, 1, 2, 3, 1, 0, 0,
           3, 3, 0, 1, 3, 0, 1, 3, 3, 2, 1, 0, 1, 1, 2, 0, 0, 0, 3, 2, 3, 0, 1,
           2, 2, 1, 2, 0, 0, 3, 2, 3, 0, 2, 1, 2, 0, 3, 1, 2, 2, 3, 2, 3, 0, 3,
           3, 1, 0, 0, 0, 1, 2, 3, 1, 0, 3, 2, 1, 3, 3, 3, 0, 2, 1, 2, 2, 2, 1,
           2, 0, 3, 0, 2, 3, 0, 0, 3, 2, 3, 1, 3, 0, 0, 1, 1, 0, 3, 1, 3, 0, 2,
           0, 0, 3, 0, 3, 1, 1, 3, 2, 0, 2, 3, 1, 3])




```python
metrics.homogeneity_score(labels_true, labels)
```




    0.38495987157813472




```python
metrics.completeness_score(labels_true, labels)
```




    0.78345864156276968




```python
metrics.v_measure_score(labels_true, labels)
```




    0.51625361058689845




```python
metrics.adjusted_rand_score(labels_true, labels)
```

    /usr/local/lib/python3.5/dist-packages/sklearn/metrics/cluster/supervised.py:25: DeprecationWarning: `comb` is deprecated!
    Importing `comb` from scipy.misc is deprecated in scipy 1.0.0. Use `scipy.special.comb` instead.
      return comb(n, 2, exact=1)
    /usr/local/lib/python3.5/dist-packages/sklearn/metrics/cluster/supervised.py:25: DeprecationWarning: `comb` is deprecated!
    Importing `comb` from scipy.misc is deprecated in scipy 1.0.0. Use `scipy.special.comb` instead.
      return comb(n, 2, exact=1)
    /usr/local/lib/python3.5/dist-packages/sklearn/metrics/cluster/supervised.py:25: DeprecationWarning: `comb` is deprecated!
    Importing `comb` from scipy.misc is deprecated in scipy 1.0.0. Use `scipy.special.comb` instead.
      return comb(n, 2, exact=1)
    /usr/local/lib/python3.5/dist-packages/sklearn/metrics/cluster/supervised.py:25: DeprecationWarning: `comb` is deprecated!
    Importing `comb` from scipy.misc is deprecated in scipy 1.0.0. Use `scipy.special.comb` instead.
      return comb(n, 2, exact=1)
    /usr/local/lib/python3.5/dist-packages/sklearn/metrics/cluster/supervised.py:25: DeprecationWarning: `comb` is deprecated!
    Importing `comb` from scipy.misc is deprecated in scipy 1.0.0. Use `scipy.special.comb` instead.
      return comb(n, 2, exact=1)
    /usr/local/lib/python3.5/dist-packages/sklearn/metrics/cluster/supervised.py:25: DeprecationWarning: `comb` is deprecated!
    Importing `comb` from scipy.misc is deprecated in scipy 1.0.0. Use `scipy.special.comb` instead.
      return comb(n, 2, exact=1)
    /usr/local/lib/python3.5/dist-packages/sklearn/metrics/cluster/supervised.py:25: DeprecationWarning: `comb` is deprecated!
    Importing `comb` from scipy.misc is deprecated in scipy 1.0.0. Use `scipy.special.comb` instead.
      return comb(n, 2, exact=1)
    /usr/local/lib/python3.5/dist-packages/sklearn/metrics/cluster/supervised.py:25: DeprecationWarning: `comb` is deprecated!
    Importing `comb` from scipy.misc is deprecated in scipy 1.0.0. Use `scipy.special.comb` instead.
      return comb(n, 2, exact=1)
    /usr/local/lib/python3.5/dist-packages/sklearn/metrics/cluster/supervised.py:25: DeprecationWarning: `comb` is deprecated!
    Importing `comb` from scipy.misc is deprecated in scipy 1.0.0. Use `scipy.special.comb` instead.
      return comb(n, 2, exact=1)
    /usr/local/lib/python3.5/dist-packages/sklearn/metrics/cluster/supervised.py:25: DeprecationWarning: `comb` is deprecated!
    Importing `comb` from scipy.misc is deprecated in scipy 1.0.0. Use `scipy.special.comb` instead.
      return comb(n, 2, exact=1)
    /usr/local/lib/python3.5/dist-packages/sklearn/metrics/cluster/supervised.py:25: DeprecationWarning: `comb` is deprecated!
    Importing `comb` from scipy.misc is deprecated in scipy 1.0.0. Use `scipy.special.comb` instead.
      return comb(n, 2, exact=1)
    /usr/local/lib/python3.5/dist-packages/sklearn/metrics/cluster/supervised.py:25: DeprecationWarning: `comb` is deprecated!
    Importing `comb` from scipy.misc is deprecated in scipy 1.0.0. Use `scipy.special.comb` instead.
      return comb(n, 2, exact=1)
    /usr/local/lib/python3.5/dist-packages/sklearn/metrics/cluster/supervised.py:25: DeprecationWarning: `comb` is deprecated!
    Importing `comb` from scipy.misc is deprecated in scipy 1.0.0. Use `scipy.special.comb` instead.
      return comb(n, 2, exact=1)
    /usr/local/lib/python3.5/dist-packages/sklearn/metrics/cluster/supervised.py:25: DeprecationWarning: `comb` is deprecated!
    Importing `comb` from scipy.misc is deprecated in scipy 1.0.0. Use `scipy.special.comb` instead.
      return comb(n, 2, exact=1)
    /usr/local/lib/python3.5/dist-packages/sklearn/metrics/cluster/supervised.py:25: DeprecationWarning: `comb` is deprecated!
    Importing `comb` from scipy.misc is deprecated in scipy 1.0.0. Use `scipy.special.comb` instead.
      return comb(n, 2, exact=1)
    /usr/local/lib/python3.5/dist-packages/sklearn/metrics/cluster/supervised.py:25: DeprecationWarning: `comb` is deprecated!
    Importing `comb` from scipy.misc is deprecated in scipy 1.0.0. Use `scipy.special.comb` instead.
      return comb(n, 2, exact=1)
    /usr/local/lib/python3.5/dist-packages/sklearn/metrics/cluster/supervised.py:25: DeprecationWarning: `comb` is deprecated!
    Importing `comb` from scipy.misc is deprecated in scipy 1.0.0. Use `scipy.special.comb` instead.
      return comb(n, 2, exact=1)
    /usr/local/lib/python3.5/dist-packages/sklearn/metrics/cluster/supervised.py:25: DeprecationWarning: `comb` is deprecated!
    Importing `comb` from scipy.misc is deprecated in scipy 1.0.0. Use `scipy.special.comb` instead.
      return comb(n, 2, exact=1)
    /usr/local/lib/python3.5/dist-packages/sklearn/metrics/cluster/supervised.py:25: DeprecationWarning: `comb` is deprecated!
    Importing `comb` from scipy.misc is deprecated in scipy 1.0.0. Use `scipy.special.comb` instead.
      return comb(n, 2, exact=1)
    /usr/local/lib/python3.5/dist-packages/sklearn/metrics/cluster/supervised.py:190: DeprecationWarning: `comb` is deprecated!
    Importing `comb` from scipy.misc is deprecated in scipy 1.0.0. Use `scipy.special.comb` instead.
      prod_comb = (sum_comb_c * sum_comb_k) / float(comb(n_samples, 2))





    0.3216917121046558




```python
metrics.adjusted_mutual_info_score(labels_true, labels)
```




    0.3831341743911893




```python
metrics.silhouette_score(X, labels)
```




    0.2973846313006141




```python
# Black removed and is used for noise instead.
unique_labels = set(labels)
colors = [plt.cm.Spectral(each)
          for each in np.linspace(0, 1, len(unique_labels))]
colors
```




    [(0.61960784313725492, 0.0039215686274509803, 0.25882352941176473, 1.0),
     (0.99807766243752405, 0.99923106497500958, 0.74602076124567474, 1.0),
     (0.36862745098039218, 0.30980392156862746, 0.63529411764705879, 1.0)]




```python
for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = [0, 0, 0, 1]

    class_member_mask = (labels == k) # labels == unique_labels

    xy = X[class_member_mask & core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=6)

plt.title('Estimated number of clusters: %d' % n_clusters_)
plt.show()
```


![png](output_17_0.png)



```python
for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = [0, 0, 0, 1]

    class_member_mask = (labels == k) # labels == unique_labels

    xy = X[class_member_mask & ~core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=6)

plt.title('Estimated number of clusters: %d' % n_clusters_)
plt.show()
```


![png](output_18_0.png)



```python
import seaborn as sns
import sklearn.datasets as data
```


```python
moons, _ = data.make_moons(n_samples=50, noise=0.05)
blobs, _ = data.make_blobs(n_samples=50, centers=[(-0.75,2.25), (1.0, 2.0)], cluster_std=0.25)

plt.rcParams["figure.figsize"] = (8,7)
plt.scatter(moons[:,0], moons[:,1])
```




    <matplotlib.collections.PathCollection at 0x7f33a6d37e48>




![png](output_20_1.png)



```python
plt.scatter(blobs[:,0], blobs[:,1])
```




    <matplotlib.collections.PathCollection at 0x7f33a6d63438>




![png](output_21_1.png)



```python
test_data = np.vstack([moons, blobs])
print(moons.shape, blobs.shape, test_data.shape)
plt.scatter(test_data.T[0], test_data.T[1], color='b', **plot_kwds)
```

    (50, 2) (50, 2) (100, 2)





    <matplotlib.collections.PathCollection at 0x7f33a6d3a4e0>




![png](output_22_2.png)



```python
import hdbscan
```


```python
clusterer = hdbscan.HDBSCAN(min_cluster_size=5, gen_min_span_tree=True)
clusterer.fit(test_data)
```




    HDBSCAN(algorithm='best', allow_single_cluster=False, alpha=1.0,
        approx_min_span_tree=True, cluster_selection_method='eom',
        core_dist_n_jobs=4, gen_min_span_tree=True, leaf_size=40,
        match_reference_implementation=False, memory=Memory(cachedir=None),
        metric='euclidean', min_cluster_size=5, min_samples=None, p=None,
        prediction_data=False)



So now that we have clustered the data -- what actually happened? We can break it out into a series of steps

1. `Transform the space` according to the density/sparsity.
2. Build the `minimum spanning tree of the distance weighted graph`.
3. `Construct a cluster hierarchy` of connected components.
4. `Condense the cluster hierarchy` based on minimum cluster size.
5. `Extract the stable clusters` from the condensed tree.

## Transform the space

To find clusters we want to `find the islands of higher density amid a sea of sparser noise` 

The core of the clustering algorithm is single linkage clustering, and it can be quite `sensitive to noise`: 

`a single noise data point in the wrong place can act as a bridge between islands`, gluing them together. Obviously we want our algorithm to be robust against noise so we need to find a way to help 'lower the sea level' before running a single linkage algorithm.


> How does it work in practice? 

- We need a very inexpensive estimate of density
    - and the simplest is the distance to `the kth nearest neighbor` (how many neighbour points would be included in the cluster) 
        - If we have the distance matrix for our data, we can simply read that off
        - alternatively if our metric is supported (and dimension is low) this is the sort of query that [kd-trees](http://scikit-learn.org/stable/modules/neighbors.html#k-d-tree) are good for. 
        
        
## Formalize:

**core distance** defined for parameter *k* for a point *x* and denote as $\mathrm{core}_k(x)$. 

Now we need a way to spread apart points with low density (correspondingly high core distance). 

Define a new distance metric between pointsature) : **mutual reachability distance**.  

$$d_{\mathrm{mreach-}k}(a,b) = \max \{\mathrm{core}_k(a), \mathrm{core}_k(b), d(a,b) \}$$

- where d(a,b) is the original metric distance between a and b. 

- Under this metric 
    - dense points (with low core distance) remain the same distance from each other
    - sparser points are pushed away to be at least their core distance away from any other point. 
- Obviously this is dependent upon __the choice of k__

<img src="../images/distance1.svg" alt="Diagram demonstrating mutual reachability distance" width=640 height=480>

Pick another point and we can do the same thing, this time with a different set of neighbors (one of them even being the first point we picked out).

<img src="../images/distance2.svg" alt="Diagram demonstrating mutual reachability distance" width=640 height=480>

And we can do that a third time for good measure, with another set of six nearest neighbors and another circle with slightly different radius again.

<img src="../images/distance3.svg" alt="Diagram demonstrating mutual reachability distance" width=640 height=480>

Now if we want to know the mutual reachabiility distance between the blue and green points we can start by drawing in and arrow giving the distance between green and blue:

<img src="../images/distance4.svg" alt="Diagram demonstrating mutual reachability distance" width=640 height=480>

This passes through the blue circle, but not the green circle -- the core distance for green is larger than the distance between blue and green. Thus we need to mark the mutual reachability distance between blue and green as larger -- equal to the radius of the green circle (easiest to picture if we base one end at the green point).

$$d_{\mathrm{mreach-}k}(a,b) = \max \{\mathrm{core}_k(a), \mathrm{core}_k(b), d(a,b) \}$$

<img src="../images/distance4a.svg" alt="Diagram demonstrating mutual reachability distance" width=640 height=480>

On the other hand the mutual reachablity distance from red to green is simply distance from red to green since that distance is greater than either core distance (i.e. the distance arrow passes through both circles).


$$d_{\mathrm{mreach-}k}(a,b) = \max \{\mathrm{core}_k(a), \mathrm{core}_k(b), d(a,b) \}$$

<img src="../images/distance5.svg" alt="Diagram demonstrating mutual reachability distance" width=640 height=480>


## Build the minimum spanning tree

Now that we have a new mutual reachability metric on the data 

#### we want start finding the clusters on dense data.

Of course dense areas are relative, and different islands may have different densities. 

Conceptually what we will do is the following: 

- consider the data as a weighted graph 
    - with the data points as vertices
    - an edge between any two points with weight (`equal to the mutual reachability distance of those points`)

- Now consider a threshold value, starting high, and steadily being lowered. 
    - Drop any edges with weight above that threshold. 

- As we drop edges we will start to disconnect the graph into connected components. 

- Eventually we will have a hierarchy of connected components (from completely connected to completely disconnected) at varying threshold levels.


In practice this is very expensive: 

### there are n^2 edges and we don't want to have to run a connected components algorithm that many times. 

The right thing to do is to find a minimal set of edges such that dropping any edge from the set causes a disconnection of components. But we need more, we need this set to be such that there is no lower weight edge that could connect the components. Fortunately graph theory furnishes us with just such a thing: the minimum spanning tree of the graph.

We can build the minimum spanning tree very efficiently via [Prim's algorithm](https://en.wikipedia.org/wiki/Prim%27s_algorithm) 


### we build the tree one edge at a time, always adding the lowest weight edge that connects the current tree to a vertex not yet in the tree. 



```python
# clusterer = hdbscan.HDBSCAN(min_cluster_size=5, gen_min_span_tree=True)

plt.rcParams["figure.figsize"] = (15,10)
clusterer.minimum_spanning_tree_.plot(edge_cmap='viridis', 
                                      edge_alpha=0.6, 
                                      node_size=50, 
                                      edge_linewidth=2)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f33a5aa1ef0>




![png](output_30_1.png)


## Build the cluster hierarchy

Given the minimal spanning tree, the next step is to `convert that into the hierarchy of connected components`. This is most easily done in the reverse order: 

- sort the edges of the tree by distance (in increasing order) 

- then iterate through, creating a new merged cluster for each edge. 

- The only difficult part here is to identify the two clusters each edge will join together, but this is easy enough via a [union-find](https://en.wikipedia.org/wiki/Disjoint-set_data_structure) data structure.



```python
plt.rcParams["figure.figsize"] = (15, 10)
clusterer.single_linkage_tree_.plot(cmap='viridis', colorbar=True)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f33a5ccd7f0>




![png](output_32_1.png)


This brings us to the point `where robust single linkage stops`

We want more though; a cluster hierarchy is good, but we really want a set of flat clusters. We could do that by `drawing a a horizontal line through the above diagram and selecting the clusters that it cuts through`. 

This is in practice what [DBSCAN](http://scikit-learn.org/stable/modules/clustering.html#dbscan) effectively does (declaring any singleton clusters at the cut level as noise). 


The question is, `how do we know where to draw that line`?



DBSCAN simply leaves that as a (very unintuitive) parameter. 


Worse, we really want to deal with variable density clusters and `any choice of cut line is a choice of mutual reachability distance to cut at`, and hence a single fixed density level. 

Ideally we want to be able to cut the tree at different places to select our clusters. 

## thus: condense the cluster tree

The first step in cluster extraction is `condensing down the large and complicated cluster hierarchy into a smaller tree` with a little more data attached to each node. 


As you can see in the hierarchy above it is often the case that a cluster split is one or two points splitting off from a cluster; 

and that is the key point -- `rather than seeing it as a cluster splitting into two new clusters we want to view it as a single persistent cluster that is 'losing points'`. 

### minimum cluster size (critical parameter to HDBSCAN).

Once we have a value for minimum cluster size we can now walk through the hierarchy and `at each split ask if one of the new clusters created by the split has fewer points than the minimum cluster size`. 


If it is the case that we have fewer points than the minimum cluster size we declare it to be 'points falling out of a cluster' and `have the larger cluster retain the cluster identity of the parent`, marking down which points 'fell out of the cluster' and at what distance value that happened. 





```python
clusterer.condensed_tree_.plot()
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f33a5a75b38>




![png](output_35_1.png)


## Extract the clusters

We want to choose those clusters that have the `greatest area of ink in the plot`. 

To make a flat clustering we will need to add a further requirement that, `if you select a cluster, then you cannot select any cluster that is a descendant of it`. 


- First we need a different measure than distance to consider the persistence of clusters

- instead we will use $$\lambda = \frac{1}{\mathrm{distance}}$$. 

For a given cluster we can then define values $$\lambda_{\mathrm{birth}}$$ and $$\lambda_{\mathrm{death}}$$ to be the lambda value when the cluster split off and became its own cluster, and the lambda value when the cluster split into smaller clusters respectively. 



In turn, for a given cluster, for each point *p* in that cluster we can define the value $$\lambda_p$$ as the lambda value at which that point 'fell out of the cluster' 


which is a value somewhere between $\lambda_{\mathrm{birth}}$ and $\lambda_{\mathrm{death}}$ since the point either falls out of the cluster at some point in the cluster's lifetime, or leaves the cluster when the cluster splits into two smaller clusters. 

Now, for each cluster compute the **stability** to as

$$\sum_{p \in \mathrm{cluster}} (\lambda_p - \lambda_{\mathrm{birth}})$$.

`Declare all leaf nodes to be selected clusters`.

Now work up through the tree (the reverse topological sort order). 

- If the sum of the stabilities of the child clusters is greater than the stability of the cluster then we set the cluster stability to be the sum of the child stabilities. 
- If, on the other hand, the cluster's stability is greater than the sum of its children then we declare the cluster to be a selected cluster, and unselect all its descendants. 
- Once we reach the root node we call the current set of selected clusters our flat clustering and return that.



```python
clusterer.condensed_tree_.plot(select_clusters=True, selection_palette=sns.color_palette())
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f33a5aa19e8>




![png](output_37_1.png)


### Notes: The hdbscan library returns this as a `probabilities_` attribute of the clusterer object: strength of cluster membership for each point in the cluster


```python
palette = sns.color_palette()
cluster_colors = [sns.desaturate(palette[i], sat) 
                  if i >= 0 else (0.5, 0.5, 0.5) for i, sat in 
                  zip(clusterer.labels_, clusterer.probabilities_)] # for i>=0, exclude all noise points
plt.scatter(test_data.T[0], test_data.T[1], c=cluster_colors, **plot_kwds)
```




    <matplotlib.collections.PathCollection at 0x7f33a40c6198>




![png](output_39_1.png)
