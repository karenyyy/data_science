

```python
import numpy as np
import pandas as pd
```


```python
vals1 = np.array([1, None, 3, 4])
vals1
```




    array([1, None, 3, 4], dtype=object)



### ``NaN``: Missing numerical data

The other missing data representation, ``NaN`` (acronym for *Not a Number*), is different; it is a special floating-point value recognized by all systems that use the standard IEEE floating-point representation:


```python
vals2 = np.array([1, np.nan, 3, 4]) 
vals2.dtype
```




    dtype('float64')



__(Attention!!)__   

``NaN`` is a bit like a data virus–it infects any other object it touches.Regardless of the operation, the result of arithmetic with ``NaN`` will be another ``NaN``:


```python
1 + np.nan
```




    nan




```python
0 *  np.nan
```




    nan




```python
vals2.sum(), vals2.min(), vals2.max()
```




    (nan, nan, nan)



> Solutions?

NumPy does provide some special aggregations that will ignore these missing values:


```python
np.nansum(vals2), np.nanmin(vals2), np.nanmax(vals2)
```




    (8.0, 1.0, 4.0)



Keep in mind that ``NaN`` is specifically a floating-point value; there is no equivalent NaN value for integers, strings, or other types.

### NaN and None in Pandas

``NaN`` and ``None`` both have their place, and Pandas is built to handle the two of them nearly interchangeably, converting between them where appropriate:


```python
pd.Series([1, np.nan, 2, None])
```




    0    1.0
    1    NaN
    2    2.0
    3    NaN
    dtype: float64




```python
x = pd.Series(range(2), dtype=int)
x
```




    0    0
    1    1
    dtype: int64




```python
x[0] = None
x
```




    0    NaN
    1    1.0
    dtype: float64



__Notice that in addition to casting the integer array to floating point, Pandas automatically converts the ``None`` to a ``NaN`` value.__



The following table lists the upcasting conventions in Pandas when NA values are introduced:

|Typeclass     | Conversion When Storing NAs | NA Sentinel Value      |
|--------------|-----------------------------|------------------------|
| ``floating`` | No change                   | ``np.nan``             |
| ``object``   | No change                   | ``None`` or ``np.nan`` |
| ``integer``  | Cast to ``float64``         | ``np.nan``             |
| ``boolean``  | Cast to ``object``          | ``None`` or ``np.nan`` |

__Keep in mind that in Pandas, string data is always stored with an ``object`` dtype.__

## Operating on Null Values


- ``isnull()``: Generate a boolean mask indicating missing values
- ``notnull()``: Opposite of ``isnull()``
- ``dropna()``: Return a filtered version of the data
- ``fillna()``: Return a copy of the data with missing values filled or imputed



```python
data = pd.Series([1, np.nan, 'hello', None])
```


```python
data.isnull()
```




    0    False
    1     True
    2    False
    3     True
    dtype: bool




```python
data[data.notnull()]
```




    0        1
    2    hello
    dtype: object




```python
data.dropna()
```




    0        1
    2    hello
    dtype: object




```python
df = pd.DataFrame([[1,      np.nan, 2],
                   [2,      3,      5],
                   [np.nan, 4,      6]])
df
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>NaN</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>3</td>
      <td>5</td>
    </tr>
    <tr>
      <th>2</th>
      <td>NaN</td>
      <td>4</td>
      <td>6</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.dropna()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>2.0</td>
      <td>3.0</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.dropna(axis='columns')
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5</td>
    </tr>
    <tr>
      <th>2</th>
      <td>6</td>
    </tr>
  </tbody>
</table>
</div>



#### dropping rows or columns with *all* NA values, or a majority of NA values.

This can be specified through the ``how`` or ``thresh`` parameters, which allow fine control of the number of nulls to allow through.

The default is ``how='any'``, such that __any row or column (depending on the ``axis`` keyword) containing a null value will be dropped__.

You can also specify ``how='all'``, which will __only drop rows/columns that are *all* null values__


```python
df[4] = np.nan
df=df.drop([4], axis=1)
df
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>NaN</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>3</td>
      <td>5</td>
    </tr>
    <tr>
      <th>2</th>
      <td>NaN</td>
      <td>4</td>
      <td>6</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.dropna(axis='columns', how='all')
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.0</td>
      <td>NaN</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2.0</td>
      <td>3.0</td>
      <td>5</td>
    </tr>
    <tr>
      <th>2</th>
      <td>NaN</td>
      <td>4.0</td>
      <td>6</td>
    </tr>
  </tbody>
</table>
</div>



For finer-grained control, the ``thresh`` parameter lets you specify a minimum number of non-null values for the row/column to be kept:


```python
df.dropna(axis='rows', thresh=3)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>2.0</td>
      <td>3.0</td>
      <td>5</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
data = pd.Series([1, np.nan, 2, None, 3], index=list('abcde'))
data
```




    a     1
    b   NaN
    c     2
    d   NaN
    e     3
    dtype: float64



We can fill NA entries with a single value, such as zero:


```python
data.fillna(0)
```




    a    1.0
    b    0.0
    c    2.0
    d    0.0
    e    3.0
    dtype: float64



We can specify a __forward-fill to propagate the previous value forward__:


```python
# forward-fill
data.fillna(method='ffill')
```




    a    1.0
    b    1.0
    c    2.0
    d    2.0
    e    3.0
    dtype: float64



Or we can specify a __back-fill to propagate the next values backward__:


```python
# back-fill
data.fillna(method='bfill')
```




    a    1.0
    b    2.0
    c    2.0
    d    3.0
    e    3.0
    dtype: float64




```python
df
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>NaN</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>3</td>
      <td>5</td>
    </tr>
    <tr>
      <th>2</th>
      <td>NaN</td>
      <td>4</td>
      <td>6</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.fillna(method='ffill', axis=1)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>3</td>
      <td>5</td>
    </tr>
    <tr>
      <th>2</th>
      <td>NaN</td>
      <td>4</td>
      <td>6</td>
    </tr>
  </tbody>
</table>
</div>



__Notice that if a previous value is not available during a forward fill, the NA value remains__.
