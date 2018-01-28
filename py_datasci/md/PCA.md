
## Brief intro

PCA is used as linear dimensionality reduction method. 


Sometimes it is used alone and sometimes `as a starting solution for other dimensionality reduction algorithms (eg: as initialization for T-SNE)`. 


PCA is` a projection based methods` which transforms the data by `projecting it onto another set of orthogonal axes`


>  Now, which variable would you choose for clustering? 

- If we choose a variable which varies significantly from one cluster to another cluster, we would be able segregate them easily.

- On the other hand if choose a variable which remains almost same in most of the clusters, then clusters will appear crowded and clustered in a single lump. 

> What if we don't have a variable which segragates clusters well?

- Create an artificial variable from original variables by combining them linearly like `2 * variable_1 - 3 * variable_2 + 5 * variable_3`  which has the `highest variance` among clusters. 

### This is what essentialy PCA does, it finds best available `linear combinations of original variables to maximize the variance`


```python
%matplotlib notebook
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from JSAnimation.IPython_display import display_animation
```


```python
np.random.seed(100)
X = np.random.rand(100,2)
t = np.array([[1, 0.7 ],[0.7, 0.7]])
X = X.dot(np.linalg.cholesky(t))
X = X - X.mean(axis=0)
```


```python
def init():
    fig = plt.figure(figsize= (7,5))
    ax = plt.axes(xlim=(-1, 1), ylim=(-0.5, 0.9))
    ax.scatter(X[:,0],X[:,1])
    line1, = ax.plot([], [], lw=2)
    line2, = ax.plot([], [], lw=2)

    proj_points1, = ax.plot([],[],"o")
    proj_points2, = ax.plot([],[],"o", markersize=10)

    text1 = ax.text(0.25, 0.8,'')
    text2 = ax.text(0.25, 0.7,'')

    lines = list(map(lambda i:ax.plot([],[],lw="1", c="m")[0], range(100)))
    
    line1.set_data([], [])
    line2.set_data([], [])
    proj_points1.set_data([],[])
    proj_points2.set_data([],[])
    text1.set_text('')
    text2.set_text('')
    for i in range(100):
        
        lines[i].set_data([],[])
        
    return line1,line2,proj_points2,proj_points1, lines, text1, text2

```


```python
def animate(i):
    j=10*i
    W = np.array([[np.cos(np.deg2rad(j))], [np.sin(np.deg2rad(j))]])
    Z = X.dot(W.dot(W.T))
    
    proj_points1.set_data(Z[:,0], Z[:,1])
    indices = np.argsort(Z[:,0])
    p = Z[indices[[0,-1]]]
    proj_points2.set_data(p[:,0], p[:,1])
    
    proj_dist = np.sqrt(np.sum((p[0]-p[1])**2))
    
    line1.set_data([-W[1]*0.5*(-1), -W[1]*0.5*1], [W[0]*0.5*(-1), W[0]*0.5*1])
    line2.set_data([W[0]*(-1), W[0]*1], [W[1]*(-1), W[1]*1])

    sum_of_projs =0.0
    
    for i in range(100):
        
        lines[i].set_data([X[i,0], Z[i,0]],[X[i,1], Z[i,1]])
        sum_of_projs += np.sqrt(np.sum((X[i]-Z[i])**2))
    
    text1.set_text("sum of errors = "+"{0:.2f}".format((sum_of_projs)))
    text2.set_text("projection distance = "+"{0:.2f}".format((proj_dist)))
    
    return line1,line2, proj_points1, lines, text1, text2
```


```python
anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=18, interval=260, blit=True)

from JSAnimation import HTMLWriter
anim.save('pca.html', writer=HTMLWriter(embed_frames=True))
display_animation(anim)
```


![]()

- As the yellow line rotates, distance between the red points varies according to the angle line creates with the x-axis. 

- The red lines represents the error when each point is approximated by its projection on the line. The length of red lines also varies with the angle of the yellow line. 

- PCA tries to `maximize the distance between green points` and `minimize the sum of length of red lines` (similar to SVM)

- If we observe the widget closely, the angle at which the `distance between green points is maximum is the same angle at which sum of errors is minimum`. 


### The direction along which `variance is highest` is called first `principal axis`.

After finding the first principal axis, we need to `subtract the variance along this axis from the dataset`. 

Whatever variance remains in the dataset, is used to find the direction of next principal axis by same procedure. 

Apart from being the direction of next highest projection distance, `next principal axis must be perpendicular to the previous principal axes`. 


#### All the principal axes are arranged accroding to `their eigenvalue from highest to lowest`. 

### Then, first K significant components are selected and original dataset is projected onto these chosen axes. The columns of projected dataset are called `principal compoenents`.

### Algorithm

- Standardizing all the columns of the dataset

- Obtain principal components of the dataset. 
    - if A is invertible:
        - create a covariance matrix from A and apply eigenvalue decomposition to the covariance matrix. 
            - eigenvectors: the principal axes 
            - their dot product with the dataset: principal components
    - If A is singular:
        - apply the singular value decomposition to A  
            - columns of $$V^T$$ gives principal axes
            - columns $$U\sigma$$ gives the principal components of the datasets.  

- After obtaining principal components, take `first K components corresponding to the highest K eigenvalues`. 

- Thus, eventually reducing the number of dimensions of the dataset.  

> __A is invertible__

$$C = A^T A$$  
$$eigvals, eigvecs = Eig(C)$$
$$principal \: components = dot(A, eigvecs)$$

- sort principal_components according to the eigenvalues and select first K components    

> __A is singular__  


$$U, \sigma, V = SVD(A)$$  
$$principal \: components = U\sigma$$

- select first K components from the prinicipal components   


```python
# necessary imports
from sklearn import datasets
import numpy as np

#read the datasets
iris = datasets.load_iris()
iris_data = iris.data
iris_target = iris.target
```


```python
# standardize the columns of the data
iris_data = (iris_data - iris_data.mean(axis=0))/iris_data.std(axis=0)
# create the covariance matrix
cov_mat = np.cov(iris_data.T)
# decomposition of conv matrix
eigvals, eigvecs = np.linalg.eig(cov_mat)

# sort the eigenvalues to obtain indices the highest K eigenvalues 
# flipud to reverse (i.e. argsort in desc order)
indices = np.flipud(np.argsort(eigvals))
eigvals = np.flipud(np.sort(eigvals))

# arange the eigenvectors according to the indices 
eigvecs = eigvecs[:,indices]
```

### Use `total variance explained` 


```python
cumsum = eigvals.cumsum() 
total_variance_explained = cumsum/eigvals.sum()
```


```python
K = np.argmax(total_variance_explained>0.95)+1 # so that it starts from 1, pick the top K eigenvalues which explains more than 95% of the variance

princ_axes = eigvecs[:,0:K] # from the first column to the Kth

# principal components = dot(A, eigvecs)
princ_comps = np.dot(iris_data, princ_axes)

print(princ_axes)
```

    [[ 0.52237162 -0.37231836]
     [-0.26335492 -0.92555649]
     [ 0.58125401 -0.02109478]
     [ 0.56561105 -0.06541577]]


### Notes:

> What does negative and positive values in the principal axes tell?

- between the corresponding compoenent and variable
    - Positive values signify positve relatioship 
    - negative value signify inverse relationship
    - Higher magnitude represents higher influlence. 
        - for example:
            - in 1st principal axes `[ 0.52237162 -0.26335492  0.58125401  0.56561105]`
            - positive correlation with first,third and fourth variables
            - negative correlation with second variable
            - influence of third variable is largest.


```python
princ_comps.shape
```

   (150, 2)



```python
iris_target
```




    array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
           1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
           1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
           2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
           2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2])




```python
%matplotlib inline
import matplotlib.pyplot as plt

setosa = princ_comps[iris_target==0]
versicolor = princ_comps[iris_target==1]
verginica = princ_comps[iris_target==2]
plt.scatter(setosa[:,0], setosa[:,1], c="b",label="setosa")
plt.scatter(versicolor[:,0], versicolor[:,1], c="g",label="versicolor")
plt.scatter(verginica[:,0], verginica[:,1], c="r",label="verginica")
plt.legend()
```




   <matplotlib.legend.Legend at 0x7fe3c590bfd0>




![png](https://raw.githubusercontent.com/karenyyy/data_science/master/py_datasci/images/output_16_1.png)


### cons

- Unable to model non-linear dataset
- Sensitive to outliers. 

> Solutions?

- Other versions of PCA 
    - kernel PCA(non -linear PCA)
    - Incremental PCA(online learning)
    - Robust PCA(robust to outliers).
