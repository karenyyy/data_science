
### First, brief intro:

$$ \text{Definition for the functional margin of data point  } x^{(i)}$$

$$\hat \gamma^{(i)} = y^{(i)}(w^T x^{(i)} + b)$$

$$\text{Given    } y^{(i)} \text{  is   } \pm 1 \text{   for the positive and negative examples   },
 \hat \gamma^{(i)} \text{    is actually equivalent to }$$
  
$$| w^T x^{(i)} + b |$$


$$ \text{  which is a scaled distance of x to the line (verified by geometry)  }$$
 
$$w^T x^{(i)} + b = 0 \text{   and the scaler is     } \left\|w\right\|$$



$$ \text{  If we let   } x_0  \text{ be the vector that is perpendicular to the 
line, and also exactly on the line.}  $$

$$ \hat \gamma \text{ can also be written in terms of   } x_0$$


$$\hat \gamma^{(i)} = y^{(i)}(w^T x^{(i)} + b)= |w^T x^{(i)} + b| = y^{(i)}w^T(x^{(i)} - x_0)$$



$$ \text{ In the context of a training set with multiple data points  } x^{(1)}, x^{(2)}, 
\cdots, x^{(m)} \text{ , we define the functional margin of the training set (   } \hat \Gamma  
\text{  ) to be  }$$ 

$$\hat \Gamma = \mathrm{min}_{i=i,\cdots,m} \hat \gamma^{(i)}$$



## SVM

 After some reasoning: the SVM becomes an optimization problem for the following 
 optimal margin classifier:

     
$$\mathrm{min}_{\gamma, w, b} \frac{1}{2}\left\|w\right\|^2 $$
     
$$\text{ Subject to:   } y^{(i)}(w^Tx^{(i)} + b) \ge 1, i = 1, 2, \cdots, m$$



Applying Lagrange duality, the optimization problem becomes

![](https://raw.githubusercontent.com/karenyyy/data_science/master/py_datasci/images/svm1.png)

![](https://raw.githubusercontent.com/karenyyy/data_science/master/py_datasci/images/svm2.png)

![](https://raw.githubusercontent.com/karenyyy/data_science/master/py_datasci/images/svm3.png)

![](https://raw.githubusercontent.com/karenyyy/data_science/master/py_datasci/images/svm4.png)

![](https://raw.githubusercontent.com/karenyyy/data_science/master/py_datasci/images/svm5.png)

![](https://raw.githubusercontent.com/karenyyy/data_science/master/py_datasci/images/svm6.png)

> Then how do SVM complete a classification task?

After converting w, adjust the format of f(x):

![](https://raw.githubusercontent.com/karenyyy/data_science/master/py_datasci/images/svm7.png)

__Thus when attempting to classify a new data point, just calculate the inner product of this point 
and training data points whose cluster belonged to is already known, then check the sign of f(x)__


$$\max_{\alpha} W(\alpha) =$$

$$\sum_{i=1}^{m} \alpha_i - \frac{1}{2} \sum_{i,j=1}^{m}y^{(i)}y^{(j)} \alpha_i \alpha_j x_i^T x_j$$

Subject to:

$$0 \le \alpha_i \le C, i = 1, 2, \cdots, m$$

and

$$\sum_{i=1}^{m} \alpha_i y^{(i)} = 0$$



```python
import numpy as np
import matplotlib.pyplot as plt

%matplotlib inline
```


```python
# toy dataset
Xs = np.array([
        [0.5, 0.5],
        [1, 1],
        [1.5, 2],
        [2, 1]
    ])
ys = [-1, -1, 1, 1]
```


```python
plt.scatter(*Xs[:2].T, marker='x', s=50)
plt.scatter(*Xs[2:].T, marker='o', s=50)
```




    <matplotlib.collections.PathCollection at 0x7f8f94877da0>




![png](https://raw.githubusercontent.com/karenyyy/data_science/master/py_datasci/images2/output_8_1.png)



```python
def calc_f(xk, alphas, Xs, ys, b):
    return sum(alpha * yi * xi.dot(xk) for (alpha, xi, yi) in zip(alphas, Xs, ys)) + b
```


```python
def calc_E(xk, yk, alphas, Xs, ys, b):
    return calc_f(xk, alphas, Xs, ys, b) - yk
```


```python
def calc_Lower(yi, yj, ai, aj, C):
    if yi != yj:
        return max(0, aj - ai)
    elif yi == yj:
        return max(0, ai + aj - C)
    else:
        raise
        
def calc_Upper(ai, aj, yi, yj, C):
    if yi != yj:
        return min(C, aj - ai + C)
    elif yi == yj:
        return min(C, ai + aj)
    else:
        raise
```


```python
# determine a_j first
def update_a_j(aj, yj, Ei, Ej, eta, Upper, Lower):
    aj = aj - yj * (Ei - Ej) / eta
    if aj > Upper:
        return Upper
    elif aj < Lower:
        return Lower
    else:
        return aj

# then determine a_i based on a_j's pos
def update_a_i(ai, yi, yj, aj_old, aj):
    return ai + yi * yj * (aj_old - aj)
```


```python
def calc_b(b1, b2, ai, aj, C):
    if 0 < ai < C:
        return b1
    elif 0 < aj < C:
        return b2
    else:
        return (b1 + b2) / 2
```


```python
# the toy example is linearly separable, so C could be arbitrarily large
C = 1e8
tol = 0.001
max_iter = 10000

n = Xs.shape[0]
alphas = np.zeros(n)
b = 0
```


```python
iter = 0
while iter < max_iter:
    num_changed_alphas = 0
    for i in range(n):
        # initialize alpha,x,y,err
        ai = alphas[i]
        xi = Xs[i]
        yi = ys[i]
        Ei = calc_E(xi, yi, alphas, Xs, ys, b)
        if (yi * Ei < -tol and ai < C) or (yi * Ei > tol and ai > 0):
            j = np.random.choice([_ for _ in range(m) if _ != i])
            aj = alphas[j]
            xj = Xs[j]
            yj = ys[j]
            Ej = calc_E(xj, yj, alphas, Xs, ys, b)
            
            # preserve the previous position of a_i and a_j
            ai_old = ai
            aj_old = aj
            
            Lower = calc_Lower(ai, aj, yi, yj, C)
            Upper = calc_Upper(ai, aj, yi, yj, C)
            if Lower == Upper:
                continue
                
            eta = 2 * xi.dot(xj) - xi.dot(xi) - xj.dot(xj)
            if eta >= 0:
                continue
                
            aj = update_a_j(aj, yj, Ei, Ej, eta, Upper, Lower)
            alphas[j] = aj
            if np.abs(aj - aj_old) < 1e-5:
                continue
            
            ai = update_a_i(ai, yi, yj, aj_old, aj)
            
            alphas[i] = ai
            
            
            b1 = b - Ei - yi * (ai - ai_old) * xi.dot(xi) - yj * (aj - aj_old) * xi.dot(xj)
           
            b2 = b - Ej - yi * (ai - ai_old) * xi.dot(xj) - yj * (aj - aj_old) * xj.dot(xj)
            
            b = calc_b(b1, b2, ai, aj, C)
            num_changed_alphas += 1
            
    if num_changed_alphas == 0:
        iter += 1
    else:
        iter = 0
```


```python
alphas
```




    array([ 0.        ,  2.50031104,  1.00062208,  1.49968896])




```python
b
```




    -4.0006220800000012




```python
# most alphas should be 0
ws = sum([alphas[i] * ys[i] * Xs[i] for i in range(m)])
```


```python
norm_ws = ws / np.sqrt(ws.dot(ws))
(norm_ws**2).sum() # should be converging to 1 as much as possible
```




    0.99999999999999989




```python
margin = np.min(np.abs(Xs.dot(ws) + b) / np.sqrt(ws.dot(ws))) # the min distance of point xs to y=wx+b
```


```python

_x1s = np.array([0, 3])
_x2s = (-b - ws[0] * _x1s) / ws[1]


_xs = np.array([_x1s, _x2s]).T
assert (_xs[1] - _xs[0]).dot(ws) < 1e-12

# decision boundary
plt.plot(*_xs.T, lw=1, color='black')
# margin
plt.plot(*(_xs - norm_ws * margin).T, color='black', lw=1, linestyle='--')
plt.plot(*(_xs + norm_ws * margin).T, color='black', lw=1, linestyle='--')

plt.scatter(*Xs[:2].T, marker='x', s=50)
plt.scatter(*Xs[2:].T, marker='o', s=50)

plt.xlim(0.4, 2.1)
plt.ylim(0, 2.2)
```




    (0, 2.2)




![png](https://raw.githubusercontent.com/karenyyy/data_science/master/py_datasci/images2/output_21_1.png)



```python
import math
real=math.exp(1)
```


```python
a_l=[]
r_l[]
for i in range(1,20):
    n=10**i
    estimated=(1+1/n)**n
    a_e=estimated-real
    r_e=(estimated-real)/real
    a_l.append(a_e)
    r_l.append(r_e)
import matplotlib.pyplot as plt

plt.sub
```



# Implementation: Support Vector Machines


```python
%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
plt.rcParams["figure.figsize"] = (16,8)
# use seaborn plotting defaults
import seaborn as sns; sns.set()
```

## Support Vector Machines: Maximizing the *Margin*

Support vector machines offer one way to improve on this.
The intuition is this: rather than simply drawing a zero-width line between the classes, we can draw around each line a *margin* of some width, up to the nearest point.
Here is an example of how this might look:

In support vector machines, the line that maximizes this margin is the one we will choose as the optimal model.


```python
from sklearn.svm import SVC 
model = SVC(kernel='linear', C=1E8)
model.fit(X, y)
```




    SVC(C=100000000.0, cache_size=200, class_weight=None, coef0=0.0,
      decision_function_shape=None, degree=3, gamma='auto', kernel='linear',
      max_iter=-1, probability=False, random_state=None, shrinking=True,
      tol=0.001, verbose=False)




```python
def plot_svc_decision_function(model, ax=None, plot_support_vectors=True):
    """Plot the decision function for a 2D SVC"""
    if ax is None:
        ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    
    # create grid to evaluate model
    x = np.linspace(xlim[0], xlim[1], 30)
    y = np.linspace(ylim[0], ylim[1], 30)
    Y, X = np.meshgrid(y, x)
    xy = np.vstack([X.ravel(), Y.ravel()]).T
    P = model.decision_function(xy).reshape(X.shape)
    
    # plot decision boundary and margins
    ax.contour(X, Y, P, colors='k',
               levels=[-1, 0, 1], alpha=0.5,
               linestyles=['--', '-', '--'])
    
    # plot support vectors
    if plot_support_vectors:
        ax.scatter(model.support_vectors_[:, 0],
                   model.support_vectors_[:, 1],
                   s=300, linewidth=1, facecolors='none');
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
```


```python
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn')
plot_svc_decision_function(model);
```


![png](https://raw.githubusercontent.com/karenyyy/data_science/master/py_datasci/images2/output_6_0.png)



```python
model.support_vectors_
```




    array([[ 0.26604148,  2.71915015],
           [ 2.79207378,  3.14029479],
           [ 1.1167688 ,  2.45256061]])



A key to this classifier's success is that for the fit, __only the position of the support vectors matter__; any points further from the margin which are on the correct side do not modify the fit, these points do not contribute to the loss function used to fit the model, so their position and number do not matter so long as they do not cross the margin.



```python
def plot_svm(N, ax=None):
    X, y = make_blobs(n_samples=200, centers=2,
                      random_state=0, cluster_std=0.60)
    X = X[:N]
    y = y[:N]
    model = SVC(kernel='linear', C=1E8)
    model.fit(X, y)
    
    ax = ax or plt.gca()
    ax.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn')
    ax.set_xlim(-1, 4)
    ax.set_ylim(-1, 6)
    plot_svc_decision_function(model, ax)

fig, ax = plt.subplots(1, 2, figsize=(16, 6))
fig.subplots_adjust(left=0.0625, right=0.95, wspace=0.1)
for axi, N in zip(ax, [100, 200]):
    plot_svm(N, axi)
    axi.set_title('N = {0}'.format(N))
```


![png](https://raw.githubusercontent.com/karenyyy/data_science/master/py_datasci/images2/output_9_0.png)


### Beyond linear boundaries: Kernel SVM


```python
from sklearn.datasets.samples_generator import make_circles
X, y = make_circles(100, factor=.1, noise=.1)

model = SVC(kernel='linear').fit(X, y)
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn')
plot_svc_decision_function(model, plot_support_vectors=True);
```


![png](https://raw.githubusercontent.com/karenyyy/data_science/master/py_datasci/images2/output_11_0.png)


It is clear that no linear discrimination will be able to separate this data.


- kernel

one simple projection we could use would be to compute a *radial basis function* centered on the middle clump:


```python
kernel = np.exp(-(X ** 2).sum(1))
```


```python
from mpl_toolkits import mplot3d

def plot_3D(elev=30, azim=30, X=X, y=y):
    ax = plt.subplot(projection='3d')
    ax.scatter3D(X[:, 0], X[:, 1], kernel, c=y, s=50, cmap='autumn')
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('r')
    
    for tick in ax.w_xaxis.get_ticklines():
        tick.set_visible(False)
    for tick in ax.w_yaxis.get_ticklines():
        tick.set_visible(False)
    for tick in ax.w_zaxis.get_ticklines():
        tick.set_visible(False)

plot_3D()
```


![png](https://raw.githubusercontent.com/karenyyy/data_science/master/py_datasci/images2/output_14_0.png)



```python
clf = SVC(kernel='rbf', C=1E8)
clf.fit(X, y)
```




    SVC(C=100000000.0, cache_size=200, class_weight=None, coef0=0.0,
      decision_function_shape=None, degree=3, gamma='auto', kernel='rbf',
      max_iter=-1, probability=False, random_state=None, shrinking=True,
      tol=0.001, verbose=False)




```python
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn')
plot_svc_decision_function(clf)
plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1],
            s=300, lw=1, facecolors='none');
```


![png](https://raw.githubusercontent.com/karenyyy/data_science/master/py_datasci/images2/output_16_0.png)


### Tuning the SVM: Softening Margins

The conclusion above thus far has centered around very clean datasets, in which __a perfect decision boundary exists__. But what if your data __has some amount of overlap__?

For example, what if we have data like this:


```python
X, y = make_blobs(n_samples=100, centers=2,
                  random_state=0, cluster_std=1.2)
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn');
```


![png](https://raw.githubusercontent.com/karenyyy/data_science/master/py_datasci/images2/output_18_0.png)


To handle this case, the SVM implementation has a bit of a fudge-factor which __"softens" the margin__: that is, it allows some of the points to creep into the margin if that allows a better fit.


__The hardness of the margin is controlled by a tuning parameter, most often known as C.__

For very large C, the margin is hard, and points cannot lie in it.

For smaller C, the margin is softer, and can grow to encompass some points.

> how a changing C parameter affects the final fit via the softening of the margin?


```python
X, y = make_blobs(n_samples=100, centers=2,
                  random_state=0, cluster_std=0.8)

fig, ax = plt.subplots(2,2)
fig.subplots_adjust(left=0.0625, right=0.95, wspace=0.1)

for axi, C in zip(ax.flat, [1000, 10.0, 0.1, 0.01]):
    model = SVC(kernel='linear', C=C).fit(X, y)
    axi.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn')
    plot_svc_decision_function(model, axi)
    axi.scatter(model.support_vectors_[:, 0],
                model.support_vectors_[:, 1],
                s=300, lw=1, facecolors='none');
    axi.set_title('C = {0:.2f}'.format(C), size=14)
```


![png](https://raw.githubusercontent.com/karenyyy/data_science/master/py_datasci/images2/output_20_0.png)


The optimal value of the $C$ parameter will depend on your dataset, and should be tuned using cross-validation or a similar procedure 

## Example: Face Recognition



```python
from sklearn.datasets import fetch_lfw_people
faces = fetch_lfw_people(min_faces_per_person=60)
faces.target_names
```



    array(['Ariel Sharon', 'Colin Powell', 'Donald Rumsfeld', 'George W Bush',
           'Gerhard Schroeder', 'Hugo Chavez', 'Junichiro Koizumi',
           'Tony Blair'],
          dtype='<U17')




```python
faces.images.shape
```




    (1348, 62, 47)



Let's plot a few of these faces to see what we're working with:


```python
fig, ax = plt.subplots(3, 5)
for i, axi in enumerate(ax.flat):
    axi.imshow(faces.images[i], cmap="bone")
    axi.set(xticks=[], yticks=[],
            xlabel=faces.target_names[faces.target[i]])
```


![png](https://raw.githubusercontent.com/karenyyy/data_science/master/py_datasci/images2/output_26_0.png)


Each image contains [62×47] or nearly 3,000 pixels.


here:

__use a principal component analysis to extract 150 fundamental components to feed into our support vector machine classifier__


```python
from sklearn.svm import SVC
from sklearn.decomposition import RandomizedPCA
from sklearn.pipeline import make_pipeline

pca = RandomizedPCA(n_components=150, whiten=True, random_state=1)
svc = SVC(kernel='rbf', class_weight='balanced')
model = make_pipeline(pca, svc)
```


```python
from sklearn.cross_validation import train_test_split
Xtrain, Xtest, ytrain, ytest = train_test_split(faces.data, faces.target,
                                                random_state=1)
```

Here we will adjust ``C`` (__which controls the margin hardness__) and ``gamma`` (__which controls the size of the radial basis function kernel__), and determine the best model:


```python
from sklearn.grid_search import GridSearchCV
param_grid = {'svc__C': [1, 5, 10, 50],
              'svc__gamma': [0.0001, 0.0005, 0.001, 0.005]}
grid = GridSearchCV(model, param_grid)
```


```python
grid.score
```




    <bound method BaseSearchCV.score of GridSearchCV(cv=None, error_score='raise',
           estimator=Pipeline(steps=[('randomizedpca', RandomizedPCA(copy=True, iterated_power=3, n_components=150, random_state=1,
           whiten=True)), ('svc', SVC(C=1.0, cache_size=200, class_weight='balanced', coef0=0.0,
      decision_function_shape=None, degree=3, gamma='auto', kernel='rbf',
      max_iter=-1, probability=False, random_state=None, shrinking=True,
      tol=0.001, verbose=False))]),
           fit_params={}, iid=True, n_jobs=1,
           param_grid={'svc__gamma': [0.0001, 0.0005, 0.001, 0.005], 'svc__C': [1, 5, 10, 50]},
           pre_dispatch='2*n_jobs', refit=True, scoring=None, verbose=0)>




```python
model = grid.best_estimator_
yfit = model.predict(Xtest)
```


```python
fig, ax = plt.subplots(4, 6)
for i, axi in enumerate(ax.flat):
    axi.imshow(Xtest[i].reshape(62, 47), cmap='bone')
    axi.set(xticks=[], yticks=[])
    axi.set_ylabel(faces.target_names[yfit[i]].split()[-1],
                   color='black' if yfit[i] == ytest[i] else 'red')
fig.suptitle('Predicted Names; Incorrect Labels in Red', size=14);
```


```python
from sklearn.metrics import classification_report
print(classification_report(ytest, yfit,
                            target_names=faces.target_names))
```


```python
from sklearn.metrics import confusion_matrix
mat = confusion_matrix(ytest, yfit)
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False,
            xticklabels=faces.target_names,
            yticklabels=faces.target_names)
plt.xlabel('true label')
plt.ylabel('predicted label');
```

This helps us get a sense of which labels are likely to be confused by the estimator.

For a real-world facial recognition task, in which the photos do not come pre-cropped into nice grids, the only difference in the facial classification scheme is the feature selection: would need to use a more sophisticated algorithm to find the faces, and extract features that are independent of the pixellation.


For this kind of application, one good option is to make use of OpenCV, which, among other things, includes pre-trained implementations of some feature extraction tools for images in general and faces in particular.

## Support Vector Pros and Cons

- Pros 
    - Their dependence on relatively few support vectors indicates its compactness, and take up very little memory.
    - Because they are affected only by points near the margin, they work well with high-dimensional data—even data with more dimensions than samples.
    - Their integration with kernel methods makes them very versatile, able to adapt to many types of data.

- Cons:
    - The results are strongly dependent on a suitable choice for the softening parameter C. This must be carefully chosen via cross-validation, which can be expensive as datasets grow in size.
    - The results do not have a direct probabilistic interpretation. This can be estimated via an internal cross-validation (the ``probability`` parameter of ``SVC``), but this extra estimation is costly.



<style>
h1,
    h2,
    h3,
    h4,
    h5,
    h6,
    p,
    blockquote {
        margin: 0;
        padding: 0;
    }
    body {
        font-family: "Helvetica Neue", Helvetica, "Hiragino Sans GB", Arial, sans-serif;
        font-size: 13px;
        line-height: 18px;
        color: #737373;
        background-color: white;
        margin: 10px 13px 10px 13px;
    }
    table {
        margin: 10px 0 15px 0;
        border-collapse: collapse;
    }
    td,th {
        border: 1px solid #ddd;
        padding: 3px 10px;
    }
    th {
        padding: 5px 10px;
    }

    a {
        color: #0069d6;
    }
    a:hover {
        color: #0050a3;
        text-decoration: none;
    }
    a img {
        border: none;
    }
    p {
        margin-bottom: 9px;
    }
    h1,
    h2,
    h3,
    h4,
    h5,
    h6 {
        color: #404040;
        line-height: 36px;
    }
    h1 {
        margin-bottom: 18px;
        font-size: 30px;
    }
    h2 {
        font-size: 24px;
    }
    h3 {
        font-size: 18px;
    }
    h4 {
        font-size: 16px;
    }
    h5 {
        font-size: 14px;
    }
    h6 {
        font-size: 13px;
    }
    hr {
        margin: 0 0 19px;
        border: 0;
        border-bottom: 1px solid #ccc;
    }
    blockquote {
        padding: 13px 13px 21px 15px;
        margin-bottom: 18px;
        font-family:georgia,serif;
        font-style: italic;
    }
    blockquote:before {
        content:"\201C";
        font-size:40px;
        margin-left:-10px;
        font-family:georgia,serif;
        color:#eee;
    }
    blockquote p {
        font-size: 14px;
        font-weight: 300;
        line-height: 18px;
        margin-bottom: 0;
        font-style: italic;
    }
    code, pre {
        font-family: Monaco, Andale Mono, Courier New, monospace;
    }
    code {
        background-color: #fee9cc;
        color: rgba(0, 0, 0, 0.75);
        padding: 1px 3px;
        font-size: 12px;
        -webkit-border-radius: 3px;
        -moz-border-radius: 3px;
        border-radius: 3px;
    }
    pre {
        display: block;
        padding: 14px;
        margin: 0 0 18px;
        line-height: 16px;
        font-size: 11px;
        border: 1px solid #d9d9d9;
        white-space: pre-wrap;
        word-wrap: break-word;
    }
    pre code {
        background-color: #fff;
        color:#737373;
        font-size: 11px;
        padding: 0;
    }
    sup {
        font-size: 0.83em;
        vertical-align: super;
        line-height: 0;
    }
    * {
        -webkit-print-color-adjust: exact;
    }
    @media screen and (min-width: 914px) {
        body {
            width: 854px;
            margin:10px auto;
        }
    }
    @media print {
        body,code,pre code,h1,h2,h3,h4,h5,h6 {
            color: black;
        }
        table, pre {
            page-break-inside: avoid;
        }
    }

</style>