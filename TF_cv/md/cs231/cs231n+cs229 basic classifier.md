

```python
from IPython.core.display import Image
from scipy.ndimage import uniform_filter
import matplotlib
```

### Get Cifar10


```python
import pickle
import numpy as np
import os

def load_CIFAR_batch(filename):
  with open(filename, 'rb') as f:
    datadict = pickle.load(f,encoding='bytes')
    # print(datadict)
    X = datadict[b'data']
    Y = datadict[b'labels']
    X = X.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("float") # (10000, 32, 32, 3)
    Y = np.array(Y)
    return X, Y

def load_CIFAR10(ROOT):
  xs = []
  ys = []
  for b in range(1,6):
    f = os.path.join(ROOT, 'data_batch_%d' % (b, ))
    X, Y = load_CIFAR_batch(f)
    xs.append(X)
    ys.append(Y)    
  Xtr = np.concatenate(xs)
  Ytr = np.concatenate(ys)
  Xte, Yte = load_CIFAR_batch(os.path.join(ROOT, 'test_batch'))
  return Xtr, Ytr, Xte, Yte

Xtr, Ytr, Xte, Yte = load_CIFAR10('/home/karen/Downloads/data/cifar-10-batches-py/')
Xtr[0], Ytr[0], Xte[0], Yte[0]
```




    (array([[[ 59.,  62.,  63.],
             [ 43.,  46.,  45.],
             [ 50.,  48.,  43.],
             ...,
             [158., 132., 108.],
             [152., 125., 102.],
             [148., 124., 103.]],
     
            [[ 16.,  20.,  20.],
             [  0.,   0.,   0.],
             [ 18.,   8.,   0.],
             ...,
             [123.,  88.,  55.],
             [119.,  83.,  50.],
             [122.,  87.,  57.]],
     
            [[ 25.,  24.,  21.],
             [ 16.,   7.,   0.],
             [ 49.,  27.,   8.],
             ...,
             [118.,  84.,  50.],
             [120.,  84.,  50.],
             [109.,  73.,  42.]],
     
            ...,
     
            [[208., 170.,  96.],
             [201., 153.,  34.],
             [198., 161.,  26.],
             ...,
             [160., 133.,  70.],
             [ 56.,  31.,   7.],
             [ 53.,  34.,  20.]],
     
            [[180., 139.,  96.],
             [173., 123.,  42.],
             [186., 144.,  30.],
             ...,
             [184., 148.,  94.],
             [ 97.,  62.,  34.],
             [ 83.,  53.,  34.]],
     
            [[177., 144., 116.],
             [168., 129.,  94.],
             [179., 142.,  87.],
             ...,
             [216., 184., 140.],
             [151., 118.,  84.],
             [123.,  92.,  72.]]]), 6, array([[[158., 112.,  49.],
             [159., 111.,  47.],
             [165., 116.,  51.],
             ...,
             [137.,  95.,  36.],
             [126.,  91.,  36.],
             [116.,  85.,  33.]],
     
            [[152., 112.,  51.],
             [151., 110.,  40.],
             [159., 114.,  45.],
             ...,
             [136.,  95.,  31.],
             [125.,  91.,  32.],
             [119.,  88.,  34.]],
     
            [[151., 110.,  47.],
             [151., 109.,  33.],
             [158., 111.,  36.],
             ...,
             [139.,  98.,  34.],
             [130.,  95.,  34.],
             [120.,  89.,  33.]],
     
            ...,
     
            [[ 68., 124., 177.],
             [ 42., 100., 148.],
             [ 31.,  88., 137.],
             ...,
             [ 38.,  97., 146.],
             [ 13.,  64., 108.],
             [ 40.,  85., 127.]],
     
            [[ 61., 116., 168.],
             [ 49., 102., 148.],
             [ 35.,  85., 132.],
             ...,
             [ 26.,  82., 130.],
             [ 29.,  82., 126.],
             [ 20.,  64., 107.]],
     
            [[ 54., 107., 160.],
             [ 56., 105., 149.],
             [ 45.,  89., 132.],
             ...,
             [ 24.,  77., 124.],
             [ 34.,  84., 129.],
             [ 21.,  67., 110.]]]), 3)




```python
Xtr.shape, Ytr.shape, Xte.shape, Yte.shape
```




    ((50000, 32, 32, 3), (50000,), (10000, 32, 32, 3), (10000,))



### Get Gradient 

#### np.nditer


```python
index = np.nditer(np.array([[19,32,43],[74,59,61]]), flags=['multi_index'], op_flags=['readwrite'])
#index.iternext()
index.multi_index
```




    True




```python
from random import randrange

def eval_numerical_gradient(f, x):
  fx = f(x)
  grad = np.zeros(x.shape)
  h = 0.00001

  it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
  while not it.finished:

    # evaluate function at x+h
    ix = it.multi_index
    print(ix)
    x[ix] += h # increment by h
    fxh = f(x) # evalute f(x + h)
    x[ix] -= h # restore to previous value 

    # compute the partial derivative
    grad[ix] = (fxh - fx) / h # the slope
    print(ix, grad[ix])
    it.iternext() # step to next dimension
    print(it)
    return grad
```


```python
def grad_check_sparse(f, x, analytic_grad, num_checks):
  h = 1e-5
  x.shape
  for i in xrange(num_checks):
    ix = tuple([randrange(m) for m in x.shape])
    print ix
    
    # [f(x+h)-f(x-h)]/[(x+h)-(x-h)]
    
    x[ix] += h # increment by h
    fxph = f(x) # evaluate f(x + h)
    x[ix] -= 2 * h # increment by h
    fxmh = f(x) # evaluate f(x - h)
    x[ix] += h # reset

    grad_numerical = (fxph - fxmh) / (2 * h)
    grad_analytic = analytic_grad[ix]
    rel_error = abs(grad_numerical - grad_analytic) / (abs(grad_numerical) + abs(grad_analytic))
    print 'numerical: %f analytic: %f, relative error: %e' % (grad_numerical, grad_analytic, rel_error)
```

### Get Features


```python
def extract_features(imgs, feature_fns, verbose=False):
  """
  - imgs: N x H x W x C array of pixel data for N images.
  - feature_fns: List of k feature functions, fft, rgb, hog, img_histogram, etc..
  """
  num_images = imgs.shape[0]
  if num_images == 0:
    return np.array([])

  # Use the first image to determine feature dimensions
  feature_dims = []
  first_image_features = []
  for feature_fn in feature_fns:
    # print(feature_fn)
    feats = feature_fn(imgs[0].squeeze())
    # print(feats)
    assert len(feats.shape) == 1, 'Feature functions must be one-dimensional'
    feature_dims.append(feats.size)
    first_image_features.append(feats)

  # one column = features from all feature_functions of one image combined
  total_feature_dim = sum(feature_dims)
  imgs_features = np.zeros((total_feature_dim, num_images))
  imgs_features[:total_feature_dim, 0] = np.hstack(first_image_features)

  # Extract features for the rest of the images.
  for i in range(1, num_images):
    idx = 0
    for feature_fn, feature_dim in zip(feature_fns, feature_dims):
      next_idx = idx + feature_dim
      imgs_features[idx:next_idx, i] = feature_fn(imgs[i].squeeze())
      idx = next_idx
#     if verbose and i % 1000 == 0:
#       print('Extracted features for %d / %d images' % (i, num_images))

  return imgs_features
```


```python
def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.144])
```


```python
def fft_feature(im):
  """Return the 2D fft of the image"""
  if im.ndim == 3:
    image = rgb2gray(im)
  else:
    image = np.atleast_2d(im)

  feats = np.abs(np.fft.rfft2(image))**2
  return feats.ravel()
```


```python
def hog_feature(im):
  """Compute Histogram of Gradient (HOG) feature for an image"""
  if im.ndim == 3:
    image = rgb2gray(im)
  else:
    image = np.atleast_2d(im)

  sx, sy = image.shape # image size
  orientations = 9 # number of gradient bins
  cx, cy = (8, 8) # pixels per cell

  gx = np.zeros(image.shape)
  gy = np.zeros(image.shape)
  gx[:, :-1] = np.diff(image, n=1, axis=1) # compute gradient on x-direction
  gy[:-1, :] = np.diff(image, n=1, axis=0) # compute gradient on y-direction
  grad_mag = np.sqrt(gx ** 2 + gy ** 2) # gradient magnitude
  grad_ori = np.arctan2(gy, (gx + 1e-15)) * (180 / np.pi) + 90 # gradient orientation

  n_cellsx = int(np.floor(sx / cx))  # number of cells in x
  n_cellsy = int(np.floor(sy / cy))  # number of cells in y

  # compute orientations integral images
  orientation_histogram = np.zeros((n_cellsx, n_cellsy, orientations))

  for i in range(orientations):
    # create new integral image for this orientation
    # isolate orientations in this range
    temp_ori = np.where(grad_ori < 180 / orientations * (i + 1),
                        grad_ori, 0)
    temp_ori = np.where(grad_ori >= 180 / orientations * i,
                        temp_ori, 0)
    # select magnitudes for those orientations
    cond2 = temp_ori > 0
    temp_mag = np.where(cond2, grad_mag, 0)
    #print(uniform_filter(temp_mag, size=(cx, cy))[cx//2::cx, cy//2::cy].T)
    orientation_histogram[:,:,i] = uniform_filter(temp_mag, size=(cx, cy))[cx//2::cx, cy//2::cy]
    
  return orientation_histogram.ravel()
```



![]('https://raw.githubusercontent.com/karenyyy/data_science/master/TF_cv/images/cs231/hog1.png')





![png](https://raw.githubusercontent.com/karenyyy/data_science/master/TF_cv/images/cs231/output_14_0.png)





![]('https://raw.githubusercontent.com/karenyyy/data_science/master/TF_cv/images/cs231/hog2.png')





![png](https://raw.githubusercontent.com/karenyyy/data_science/master/TF_cv/images/cs231/output_15_0.png)




![]('https://raw.githubusercontent.com/karenyyy/data_science/master/TF_cv/images/cs231/hpg3.png')





![png](https://raw.githubusercontent.com/karenyyy/data_science/master/TF_cv/images/cs231/output_16_0.png)




![]('https://raw.githubusercontent.com/karenyyy/data_science/master/TF_cv/images/cs231/hog4.png')





![png](https://raw.githubusercontent.com/karenyyy/data_science/master/TF_cv/images/cs231/output_17_0.png)



#### Notes:

- The magnitude of gradient fires where ever there is a sharp change in intensity. None of them fire when the region is smooth. 
- The gradient image removed a lot of non-essential information ( e.g. constant colored background ), but highlighted outlines. (In other words, you can look at the gradient image and still easily say there is a person in the picture.)



![]('https://raw.githubusercontent.com/karenyyy/data_science/master/TF_cv/images/cs231/hog5.png')





![png](https://raw.githubusercontent.com/karenyyy/data_science/master/TF_cv/images/cs231/output_19_0.png)





![]('https://raw.githubusercontent.com/karenyyy/data_science/master/TF_cv/images/cs231/hog6.png')





![png](https://raw.githubusercontent.com/karenyyy/data_science/master/TF_cv/images/cs231/output_20_0.png)





![]('https://raw.githubusercontent.com/karenyyy/data_science/master/TF_cv/images/cs231/hog7.png')





![png](https://raw.githubusercontent.com/karenyyy/data_science/master/TF_cv/images/cs231/output_21_0.png)





![]('https://raw.githubusercontent.com/karenyyy/data_science/master/TF_cv/images/cs231/hog8.png')





![png](https://raw.githubusercontent.com/karenyyy/data_science/master/TF_cv/images/cs231/output_22_0.png)




![]('https://raw.githubusercontent.com/karenyyy/data_science/master/TF_cv/images/cs231/hog9.png')





![png](https://raw.githubusercontent.com/karenyyy/data_science/master/TF_cv/images/cs231/output_23_0.png)



#### Notes:

- The gradient of this patch contains 2 values ( magnitude and direction ) per pixel which adds up to 8x8x2 = 128 numbers.
- But why 8×8 patch ? Why not 32×32 ? It is a design choice informed by the scale of features we are looking for. HOG was used for pedestrian detection initially. 8×8 cells in a photo of a pedestrian scaled to 64×128 are big enough to capture interesting features ( e.g. the face, the top of the head etc. ).
- __The histogram is essentially a vector ( or an array ) of 9 bins ( numbers ) corresponding to angles 0, 20, 40, 60 … 160.__
- 


```python
def color_histogram_hsv(im, nbin=10, xmin=0, xmax=255, normalized=True):
  ndim = im.ndim
  bins = np.linspace(xmin, xmax, nbin+1)
  hsv = matplotlib.colors.rgb_to_hsv(im/xmax) * xmax
  imhist, bin_edges = np.histogram(hsv[:,:,0], bins=bins, density=normalized)
  imhist = imhist * np.diff(bin_edges)
  return imhist
```

### numpy axis review


```python
import numpy as np
import sys

x = np.array([[[ 0,  1,  2],
    [ 3,  4,  5],
    [ 6,  7,  8]],
   [[ 9, 10, 11],
    [12, 13, 14],
    [15, 16, 17]],
   [[18, 19, 20],
    [21, 22, 23],
    [24, 25, 26]]])

x.shape #(3, 3, 3)

#axis = 0 
print("axis = 0")
for j in range(0, x.shape[1]):
      for k in range(0, x.shape[2]):
        print( "element = ", (j,k), " ", [ x[i,j,k] for i in range(0, x.shape[0]) ])

        
x.sum(axis=0)            
np.array([[27, 30, 33],
       [36, 39, 42],
       [45, 48, 51]])

#axis = 1   
print("axis = 1")
for i in range(0, x.shape[0]):
    for k in range(0, x.shape[2]):
        print( "element = ", (i,k), " ", [ x[i,j,k] for j in range(0, x.shape[1]) ])

#axis = 2  
print("axis = 2")
for i in range(0, x.shape[0]):
    for j in range(0, x.shape[1]):
        print( "element = ", (i,j), " ", [ x[i,j,k] for k in range(0, x.shape[1]) ])

x.sum(0), x.sum(1), x.sum(2)

```

    axis = 0
    element =  (0, 0)   [0, 9, 18]
    element =  (0, 1)   [1, 10, 19]
    element =  (0, 2)   [2, 11, 20]
    element =  (1, 0)   [3, 12, 21]
    element =  (1, 1)   [4, 13, 22]
    element =  (1, 2)   [5, 14, 23]
    element =  (2, 0)   [6, 15, 24]
    element =  (2, 1)   [7, 16, 25]
    element =  (2, 2)   [8, 17, 26]
    axis = 1
    element =  (0, 0)   [0, 3, 6]
    element =  (0, 1)   [1, 4, 7]
    element =  (0, 2)   [2, 5, 8]
    element =  (1, 0)   [9, 12, 15]
    element =  (1, 1)   [10, 13, 16]
    element =  (1, 2)   [11, 14, 17]
    element =  (2, 0)   [18, 21, 24]
    element =  (2, 1)   [19, 22, 25]
    element =  (2, 2)   [20, 23, 26]
    axis = 2
    element =  (0, 0)   [0, 1, 2]
    element =  (0, 1)   [3, 4, 5]
    element =  (0, 2)   [6, 7, 8]
    element =  (1, 0)   [9, 10, 11]
    element =  (1, 1)   [12, 13, 14]
    element =  (1, 2)   [15, 16, 17]
    element =  (2, 0)   [18, 19, 20]
    element =  (2, 1)   [21, 22, 23]
    element =  (2, 2)   [24, 25, 26]





    (array([[27, 30, 33],
            [36, 39, 42],
            [45, 48, 51]]), array([[ 9, 12, 15],
            [36, 39, 42],
            [63, 66, 69]]), array([[ 3, 12, 21],
            [30, 39, 48],
            [57, 66, 75]]))



## KNN


```python
class KNearestNeighbor:
    def train(self, X, y):
        self.X_train = X
        self.y_train = y
        
    def compute_distances_one_loop(self, X):
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in range(num_test):
            print("training x:", self.X_train)
            print("testing x", X)
            print((self.X_train-X[i,:])**2)
            print("distance matrix each testing point: ", np.sum((self.X_train-X[i,:])**2,axis = 1))
            dists[i,:] = np.sum((self.X_train-X[i,:])**2,axis = 1)
        return dists
    
    def compute_distances_two_loops(self, X):
        number_test = X.shape[0]
        number_train = self.X_train.shape[0]
        dists = np.zeros((number_test, number_train))
        for i in range(number_test):
          for j in range(number_train):
            dists[i,j] = np.sum((X[i,:]-self.X_train[j,:])**2)
        return dists

    def compute_distances_no_loops(self, X):
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))

        T = np.sum(X**2,axis = 1)
        F = np.sum(self.X_train**2,axis = 1).T
        F = np.tile(F,(500,5000))
        FT = X.dot(self.X_train.T)
        print(T.shape,F.shape,FT.shape,X.shape,self.X_train.shape)
        dists = T+F-2*FT
        return dists

    def predict_labels(self, dists,k):
        num_test = dists.shape[0]
        y_pred = np.zeros(num_test)
        for i in range(num_test):
          closest_y = self.y_train[np.argsort(dists[i,:])[:k]]
          u, indices = np.unique(closest_y, return_inverse=True)
          y_pred[i] = u[np.argmax(np.bincount(indices))]
        return y_pred
```


```python
knn = KNearestNeighbor()
knn.train(np.array([[1,2,3],[4,5,6],[7,8,9]]), [1,2,3])
knn.compute_distances_one_loop(np.array([[7,8,9],[4,5,6],[1,2,3]]))
```

    training x: [[1 2 3]
     [4 5 6]
     [7 8 9]]
    testing x [[7 8 9]
     [4 5 6]
     [1 2 3]]
    [[36 36 36]
     [ 9  9  9]
     [ 0  0  0]]
    distance matrix each testing point:  [108  27   0]
    training x: [[1 2 3]
     [4 5 6]
     [7 8 9]]
    testing x [[7 8 9]
     [4 5 6]
     [1 2 3]]
    [[9 9 9]
     [0 0 0]
     [9 9 9]]
    distance matrix each testing point:  [27  0 27]
    training x: [[1 2 3]
     [4 5 6]
     [7 8 9]]
    testing x [[7 8 9]
     [4 5 6]
     [1 2 3]]
    [[ 0  0  0]
     [ 9  9  9]
     [36 36 36]]
    distance matrix each testing point:  [  0  27 108]





    array([[108.,  27.,   0.],
           [ 27.,   0.,  27.],
           [  0.,  27., 108.]])



## Linear SVM


```python
def svm_loss_naive(W, X, y, reg):
  """
  Inputs:
  - W: C x D array of weights
  - X: D x N array of data. Data are D-dimensional columns
  - y: 1-dimensional array of length N with labels 0...K-1, for K classes
  - reg: (float) regularization strength
  """
  dW = np.zeros(W.shape)

  num_classes = W.shape[0]
  num_train = X.shape[1]
  loss = 0.0
    
  for i in range(num_train):
    scores = W.dot(X[:, i])
    correct_class_score = scores[y[i]]
    indicator =  (scores-correct_class_score+1)>0
    for j in range(num_classes):
      if j == y[i]:
        dW[j,:] += -np.sum(np.delete(indicator,j))*X[:,i].T
        continue
      dW[j,:] += indicator[j]*X[:,i].T
      margin = scores[j] - correct_class_score + 1 
      if margin > 0:
        loss += margin

  loss /= num_train
  dW /= num_train
  
  # Add regularization to the loss and the gradient
  loss += 0.5 * reg * np.sum(W * W)
  dW += reg*W
  return loss,dW
```


```python
def svm_loss_vectorized(W, X, y, reg):
  loss = 0.0
  num_train = X.shape[1]
  dW = np.zeros(W.shape) 
  
  Loss = W.dot(X) - (W.dot(X))[y,np.arange(num_train)]+1
  Bool = Loss > 0
    
  Loss = np.sum(Loss * Bool , axis = 0) - 1.0
  Regularization = 0.5 * reg * np.sum(W*W)
  loss = np.sum(Loss) / float(num_train) +Regularization
   
  Bool = Bool*np.ones(Loss.shape)
  Bool[[y,np.arange(num_train)]] = -(np.sum(Bool,axis=0)-1)
  dW = Bool.dot(X.T) / float(num_train) # normalization
  dW += reg * W
  return loss, dW
```

## Softmax

### cs229:



![]('https://raw.githubusercontent.com/karenyyy/data_science/master/TF_cv/images/cs231/softmax1.png')





![png](https://raw.githubusercontent.com/karenyyy/data_science/master/TF_cv/images/cs231/output_36_0.png)





![]('https://raw.githubusercontent.com/karenyyy/data_science/master/TF_cv/images/cs231/softmax2.png')





![png](https://raw.githubusercontent.com/karenyyy/data_science/master/TF_cv/images/cs231/output_37_0.png)




![]('https://raw.githubusercontent.com/karenyyy/data_science/master/TF_cv/images/cs231/softmax3.png')





![png](https://raw.githubusercontent.com/karenyyy/data_science/master/TF_cv/images/cs231/output_38_0.png)





![]('https://raw.githubusercontent.com/karenyyy/data_science/master/TF_cv/images/cs231/softmax4.png')





![png](https://raw.githubusercontent.com/karenyyy/data_science/master/TF_cv/images/cs231/output_39_0.png)





![]('https://raw.githubusercontent.com/karenyyy/data_science/master/TF_cv/images/cs231/softmax5.png')





![png](https://raw.githubusercontent.com/karenyyy/data_science/master/TF_cv/images/cs231/output_40_0.png)



### cs231:



![]('https://raw.githubusercontent.com/karenyyy/data_science/master/TF_cv/images/cs231/softmax6.png')





![png](https://raw.githubusercontent.com/karenyyy/data_science/master/TF_cv/images/cs231/output_42_0.png)




![]('https://raw.githubusercontent.com/karenyyy/data_science/master/TF_cv/images/cs231/softmax7.png')  





![png](https://raw.githubusercontent.com/karenyyy/data_science/master/TF_cv/images/cs231/output_43_0.png)



#### Ans: number of class - 1 (because we need to loop through all the incorrect classes)



![]('https://raw.githubusercontent.com/karenyyy/data_science/master/TF_cv/images/cs231/softmax8.png')





![png](https://raw.githubusercontent.com/karenyyy/data_science/master/TF_cv/images/cs231/output_45_0.png)



####  will be a different loss function, linear -> non-linear
> why use hinge loss over square loss?

avoid magnifying loss from misclassified cases


```python
def L_i_vectorized(x, y, W):
    scores = W.dot(x)
    # correct class scores[y]
    margins = np.maximum(0, scores-scores[y]+1)
    margins[y] = 0
    loss_i= np.sum(margins)
    return loss_i
```



![]('https://raw.githubusercontent.com/karenyyy/data_science/master/TF_cv/images/cs231/regular1.png')





![png](https://raw.githubusercontent.com/karenyyy/data_science/master/TF_cv/images/cs231/output_48_0.png)





![]('https://raw.githubusercontent.com/karenyyy/data_science/master/TF_cv/images/cs231/softmax9.png')





![png](https://raw.githubusercontent.com/karenyyy/data_science/master/TF_cv/images/cs231/output_49_0.png)





![]('https://raw.githubusercontent.com/karenyyy/data_science/master/TF_cv/images/cs231/softmax10.png')





![png](https://raw.githubusercontent.com/karenyyy/data_science/master/TF_cv/images/cs231/output_50_0.png)




![]('https://raw.githubusercontent.com/karenyyy/data_science/master/TF_cv/images/cs231/loss.png')





![png](https://raw.githubusercontent.com/karenyyy/data_science/master/TF_cv/images/cs231/output_51_0.png)



![]('https://raw.githubusercontent.com/karenyyy/data_science/master/TF_cv/images/cs231/loss2.png')





![png](https://raw.githubusercontent.com/karenyyy/data_science/master/TF_cv/images/cs231/output_52_0.png)




```python
def softmax_loss_naive(W, X, y, reg):
  """
  Inputs:
  - W: C x D array of weights
  - X: D x N array of data. Data are D-dimensional columns
  - y: 1-dimensional array of length N with labels 0...K-1, for K classes
  - reg: (float) regularization strength
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_train = X.shape[1]
  num_class = dW.shape[0]

  loss = 0.0
  for i in range(num_train):
    X_i =  X[:,i]
    score_i = W.dot(X_i)
    stability = -score_i.max()
    exp_score_i = np.exp(score_i+stability)
    exp_score_total_i = np.sum(exp_score_i , axis = 0)
    for j in xrange(num_class):
      if j == y[i]:
        dW[j,:] += -X_i.T + (exp_score_i[j] / exp_score_total_i) * X_i.T
      else:
        dW[j,:] += (exp_score_i[j] / exp_score_total_i) * X_i.T
    numerator = np.exp(score_i[y[i]]+stability)
    denom = np.sum(np.exp(score_i+stability),axis = 0)
    loss += -np.log(numerator / float(denom))
  loss = loss / float(num_train) + 0.5 * reg * np.sum(W*W)
  dW = dW / float(num_train) + reg * W
  
  return loss, dW
```


```python
def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.
  Inputs and outputs are the same as softmax_loss_naive.
  """

  loss = 0.0
  dW = np.zeros_like(W)
  num_train = X.shape[1]
  num_class = W.shape[0]
  dW = np.zeros(W.shape) 
  
  score = W.dot(X)
  score += - np.max(score , axis=0)
  exp_score = np.exp(score)
  sum_exp_score_col = np.sum(exp_score , axis = 0)

  loss = np.log(sum_exp_score_col)
  loss = loss - score[y,np.arange(num_train)]
  loss = np.sum(loss) / float(num_train) + 0.5 * reg * np.sum(W*W)
  
  Grad = exp_score / sum_exp_score_col
  Grad[y,np.arange(num_train)] += -1.0
  dW = Grad.dot(X.T) / float(num_train) + reg*W
 
  return loss, dW
```

## Linear Classifier


```python
class LinearClassifier:

  def __init__(self):
    self.W = None
    
  def loss(self, X_batch, y_batch, reg):
    """interface"""
    pass

  def train(self, X, 
                  y, 
                  learning_rate=1e-3, 
                  reg=1e-5, 
                  num_iters=100,
                  batch_size=200, 
                  verbose=False):
    
    dim, num_train = X.shape
    num_classes = np.max(y) + 1 
    if self.W is None:
      # lazily initialize W
      self.W = np.random.randn(num_classes, dim) * 0.001

    # Run stochastic gradient descent to optimize W
    loss_history = []
    for it in range(num_iters):
      indices = np.random.choice(num_train,batch_size,replace = True) #  choose num_train from batch_size
      X_batch = X[:,indices]
      y_batch = y[indices]
      pass

      # evaluate loss and gradient
      loss, grad = self.loss(X_batch, y_batch, reg)
      loss_history.append(loss)
    
      self.W = self.W - learning_rate*grad

      if verbose and it % 100 == 0:
        print('iteration %d / %d: loss %f' % (it, num_iters, loss))

    return loss_history

  def predict(self, X):
    y_pred = np.zeros(X.shape[1])
    y_pred = np.argmax(self.W.dot(X),axis = 0) 
    return y_pred
    
class LinearSVM(LinearClassifier):
  """ A subclass that uses the Multiclass SVM loss function """
  def loss(self, X_batch, y_batch, reg):
    return svm_loss_vectorized(self.W, X_batch, y_batch, reg)


class Softmax(LinearClassifier):
  """ A subclass that uses the Softmax + Cross-entropy loss function """
  def loss(self, X_batch, y_batch, reg):
    return softmax_loss_vectorized(self.W, X_batch, y_batch, reg)
```

## Implementation

### KNN


```python
import random
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (20.0, 10.0) # set default size of plots
```


```python
cifar10_dir = '/home/karen/Downloads/data/cifar-10-batches-py'
X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)
```


```python
# show a few examples of training images from each class.
classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
num_classes = len(classes)
samples_per_class = 4
for y, cls in enumerate(classes):
  idxs = np.flatnonzero(y_train == y)
  idxs = np.random.choice(idxs, samples_per_class, replace=False)
  for i, idx in enumerate(idxs):
    plt_idx = i * num_classes + y + 1
    plt.subplot(samples_per_class, num_classes, plt_idx)
    plt.imshow(X_train[idx].astype('uint8'))
    plt.axis('off')
    if i == 0:
      plt.title(cls)
plt.show()
```


![png](https://raw.githubusercontent.com/karenyyy/data_science/master/TF_cv/images/cs231/output_61_0.png)



```python
num_training = 5000
mask = range(num_training)
X_train = X_train[mask]
y_train = y_train[mask]

num_test = 500
mask = range(num_test)
X_test = X_test[mask]
y_test = y_test[mask]

X_train.shape, X_test.shape

# (5000,32,32,3)->(5000,3072)
X_train = np.reshape(X_train, (X_train.shape[0], -1))
X_test = np.reshape(X_test, (X_test.shape[0], -1))
X_train.shape, X_test.shape

```




    ((5000, 3072), (500, 3072))




```python
knn= KNearestNeighbor()
knn.train(X_train, y_train)
dists = knn.compute_distances_two_loops(X_test)
dists
```




    array([[14469834., 17729119., 30294615., ..., 16061239., 17667570.,
            18959080.],
           [40155461., 27775852., 16326740., ..., 23320722., 22034553.,
            60347005.],
           [27298944., 18067965., 14242667., ..., 14188899., 19936218.,
            40367876.],
           ...,
           [28803995., 25632726., 40473234., ..., 26281702., 20587147.,
            35057549.],
           [13483064., 14888853., 23492261., ..., 12397759., 10127462.,
            19790514.],
           [48454469., 37011572., 40171948., ..., 37009642., 17042427.,
            64658521.]])




```python
plt.imshow(dists,interpolation = None)
```




    <matplotlib.image.Axes![] at 0x7f1b4a8f8320>




![png](https://raw.githubusercontent.com/karenyyy/data_science/master/TF_cv/images/cs231/output_64_1.png)



```python
y_test_pred = knn.predict_labels(dists,6)

num_correct = np.sum(y_test_pred == y_test)
accuracy = float(num_correct) / num_test
num_correct, num_test, accuracy
```




    (141, 500, 0.282)



### Linear SVM


```python
def split_CIFAR10_data_with_validation(num_training=49000, num_validation=1000, num_test=1000):
  cifar10_dir = '/home/karen/Downloads/data/cifar-10-batches-py'
  X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)
  
  # Subsample the data
  mask = range(num_training, num_training + num_validation) # subsample 1000 to be validation set
  X_val = X_train[mask]
  y_val = y_train[mask]
  mask = range(num_training)
  X_train = X_train[mask]
  y_train = y_train[mask]
  mask = range(num_test)
  X_test = X_test[mask]
  y_test = y_test[mask]

  return X_train, y_train, X_val, y_val, X_test, y_test

X_train, y_train, X_val, y_val, X_test, y_test = split_CIFAR10_data_with_validation()
```


```python
X_train.shape
```




    (49000, 32, 32, 3)




```python
def feature_preparation(feature_fns):
    X_train_feats = extract_features(X_train,feature_fns)
    X_val_feats = extract_features(X_val,feature_fns)
    X_test_feats = extract_features(X_test,feature_fns)

    # Preprocessing: Subtract the mean feature
    mean_feat = np.mean(X_train_feats, axis=1)
    mean_feat = np.expand_dims(mean_feat, axis=1)
    X_train_feats -= mean_feat
    X_val_feats -= mean_feat
    X_test_feats -= mean_feat

    # Preprocessing: Divide by standard deviation. This ensures that each feature
    # has roughly the same scale.
    std_feat = np.std(X_train_feats, axis=1)
    std_feat = np.expand_dims(std_feat, axis=1)
    X_train_feats /= std_feat
    X_val_feats /= std_feat
    X_test_feats /= std_feat

    # Preprocessing: Add a bias dimension
    X_train_feats = np.vstack([X_train_feats, np.ones((1, X_train_feats.shape[1]))])
    X_val_feats = np.vstack([X_val_feats, np.ones((1, X_val_feats.shape[1]))])
    X_test_feats = np.vstack([X_test_feats, np.ones((1, X_test_feats.shape[1]))])
    return X_train_feats, X_val_feats, X_test_feats

```


```python
num_color_bins = 10 # Number of bins in the color histogram
feature_fns = [lambda img: fft_feature(img)]
X_train_feats, X_val_feats, X_test_feats = feature_preparation(feature_fns)
X_train_feats.shape, X_val_feats.shape, X_test_feats.shape
```




    ((545, 49000), (545, 1000), (545, 1000))




```python
learning_rates = [10**(-f) for f in np.arange(2,4,0.2)]
regularization_strengths = [10**(f) for f in np.arange(.5,1.5,.2)]
```


```python
def get_best_model(learning_rates, regularization_strengths):
    results = {}
    best_val = -1
    best_svm = None
    for lr in learning_rates:
      for rs in regularization_strengths:
        svm = LinearSVM()
        svm.train(X_train_feats,y_train, learning_rate = lr, reg =  rs,
                      num_iters = 100, batch_size = 200,verbose = False)
        acc_training = np.mean(svm.predict(X_train_feats) == y_train)
        acc_val = np.mean(svm.predict(X_val_feats) == y_val)
        #print(X_val_feats, y_val)
        results.update({(lr,rs):(acc_training,acc_val)})
        if acc_val> best_val:
          best_val = acc_val
          best_svm = svm
          best_params = (lr,rs)
    # Print out results.
    for lr, reg in sorted(results):
        train_accuracy, val_accuracy = results[(lr, reg)]
        print('lr %e reg %e train accuracy: %f val accuracy: %f' % (
                    lr, reg, train_accuracy, val_accuracy))
    print('best validation accuracy achieved during cross-validation: %f' % best_val)
    return best_val, best_svm, best_params
```


```python
_, best_svm, _ = get_best_model(learning_rates, regularization_strengths)
```

    lr 1.584893e-04 reg 3.162278e+00 train accuracy: 0.307694 val accuracy: 0.328000
    lr 1.584893e-04 reg 5.011872e+00 train accuracy: 0.305592 val accuracy: 0.322000
    lr 1.584893e-04 reg 7.943282e+00 train accuracy: 0.316694 val accuracy: 0.328000
    lr 1.584893e-04 reg 1.258925e+01 train accuracy: 0.312490 val accuracy: 0.331000
    lr 1.584893e-04 reg 1.995262e+01 train accuracy: 0.311816 val accuracy: 0.320000
    lr 2.511886e-04 reg 3.162278e+00 train accuracy: 0.323204 val accuracy: 0.331000
    lr 2.511886e-04 reg 5.011872e+00 train accuracy: 0.317143 val accuracy: 0.333000
    lr 2.511886e-04 reg 7.943282e+00 train accuracy: 0.327796 val accuracy: 0.330000
    lr 2.511886e-04 reg 1.258925e+01 train accuracy: 0.325531 val accuracy: 0.334000
    lr 2.511886e-04 reg 1.995262e+01 train accuracy: 0.324551 val accuracy: 0.335000
    lr 3.981072e-04 reg 3.162278e+00 train accuracy: 0.333286 val accuracy: 0.335000
    lr 3.981072e-04 reg 5.011872e+00 train accuracy: 0.332857 val accuracy: 0.336000
    lr 3.981072e-04 reg 7.943282e+00 train accuracy: 0.328857 val accuracy: 0.336000
    lr 3.981072e-04 reg 1.258925e+01 train accuracy: 0.321878 val accuracy: 0.338000
    lr 3.981072e-04 reg 1.995262e+01 train accuracy: 0.329816 val accuracy: 0.341000
    lr 6.309573e-04 reg 3.162278e+00 train accuracy: 0.340286 val accuracy: 0.346000
    lr 6.309573e-04 reg 5.011872e+00 train accuracy: 0.344878 val accuracy: 0.342000
    lr 6.309573e-04 reg 7.943282e+00 train accuracy: 0.333163 val accuracy: 0.334000
    lr 6.309573e-04 reg 1.258925e+01 train accuracy: 0.331306 val accuracy: 0.337000
    lr 6.309573e-04 reg 1.995262e+01 train accuracy: 0.327755 val accuracy: 0.333000
    lr 1.000000e-03 reg 3.162278e+00 train accuracy: 0.343633 val accuracy: 0.344000
    lr 1.000000e-03 reg 5.011872e+00 train accuracy: 0.354041 val accuracy: 0.349000
    lr 1.000000e-03 reg 7.943282e+00 train accuracy: 0.350469 val accuracy: 0.356000
    lr 1.000000e-03 reg 1.258925e+01 train accuracy: 0.342163 val accuracy: 0.342000
    lr 1.000000e-03 reg 1.995262e+01 train accuracy: 0.328245 val accuracy: 0.333000
    lr 1.584893e-03 reg 3.162278e+00 train accuracy: 0.360306 val accuracy: 0.354000
    lr 1.584893e-03 reg 5.011872e+00 train accuracy: 0.354041 val accuracy: 0.352000
    lr 1.584893e-03 reg 7.943282e+00 train accuracy: 0.350959 val accuracy: 0.349000
    lr 1.584893e-03 reg 1.258925e+01 train accuracy: 0.354796 val accuracy: 0.354000
    lr 1.584893e-03 reg 1.995262e+01 train accuracy: 0.323918 val accuracy: 0.333000
    lr 2.511886e-03 reg 3.162278e+00 train accuracy: 0.362082 val accuracy: 0.350000
    lr 2.511886e-03 reg 5.011872e+00 train accuracy: 0.355327 val accuracy: 0.351000
    lr 2.511886e-03 reg 7.943282e+00 train accuracy: 0.362286 val accuracy: 0.361000
    lr 2.511886e-03 reg 1.258925e+01 train accuracy: 0.336653 val accuracy: 0.341000
    lr 2.511886e-03 reg 1.995262e+01 train accuracy: 0.329122 val accuracy: 0.326000
    lr 3.981072e-03 reg 3.162278e+00 train accuracy: 0.359510 val accuracy: 0.352000
    lr 3.981072e-03 reg 5.011872e+00 train accuracy: 0.356286 val accuracy: 0.337000
    lr 3.981072e-03 reg 7.943282e+00 train accuracy: 0.352531 val accuracy: 0.357000
    lr 3.981072e-03 reg 1.258925e+01 train accuracy: 0.338551 val accuracy: 0.343000
    lr 3.981072e-03 reg 1.995262e+01 train accuracy: 0.327449 val accuracy: 0.350000
    lr 6.309573e-03 reg 3.162278e+00 train accuracy: 0.352653 val accuracy: 0.356000
    lr 6.309573e-03 reg 5.011872e+00 train accuracy: 0.359408 val accuracy: 0.352000
    lr 6.309573e-03 reg 7.943282e+00 train accuracy: 0.350673 val accuracy: 0.356000
    lr 6.309573e-03 reg 1.258925e+01 train accuracy: 0.341082 val accuracy: 0.354000
    lr 6.309573e-03 reg 1.995262e+01 train accuracy: 0.331510 val accuracy: 0.331000
    lr 1.000000e-02 reg 3.162278e+00 train accuracy: 0.355224 val accuracy: 0.350000
    lr 1.000000e-02 reg 5.011872e+00 train accuracy: 0.339980 val accuracy: 0.340000
    lr 1.000000e-02 reg 7.943282e+00 train accuracy: 0.330408 val accuracy: 0.314000
    lr 1.000000e-02 reg 1.258925e+01 train accuracy: 0.330224 val accuracy: 0.321000
    lr 1.000000e-02 reg 1.995262e+01 train accuracy: 0.326367 val accuracy: 0.342000
    best validation accuracy achieved during cross-validation: 0.361000



```python
y_test_pred = best_svm.predict(X_test_feats)
test_accuracy = np.mean(y_test == y_test_pred)
test_accuracy
```




    0.374




```python
# display the misclassified ones
examples_per_class = 8
classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
for cls, cls_name in enumerate(classes):
    # print(y_test.shape, y_test_pred.shape)
    idxs = np.where((y_test != cls) & (y_test_pred == cls))[0]
    idxs = np.random.choice(idxs, examples_per_class, replace=False)
    for i, idx in enumerate(idxs):
        plt.subplot(examples_per_class, len(classes), i * len(classes) + cls + 1)
        plt.imshow(X_test[idx].astype('uint8'))
        plt.axis('off')
        if i == 0:
            plt.title(cls_name)
plt.show()
```


![png](https://raw.githubusercontent.com/karenyyy/data_science/master/TF_cv/images/cs231/output_75_0.png)

