
## numpy


```python
import numpy as np
```

- np.expand_dims


```python
x = np.array([1,2])
x.shape
```




  (2,)




```python
y=np.expand_dims(x, axis=0) # expand by row
y.shape
```




  (1, 2)




```python
y
```




  array([[1, 2]])




```python
# equivalent to x[np.newaxis]
x[np.newaxis]
```




  array([[1, 2]])




```python
y=np.expand_dims(x, axis=1) # expand by column
y.shape
```




  (2, 1)




```python
y
```




  array([[1],
           [2]])




```python
x[np.newaxis]
```




  array([[1, 2]])



## pandas
