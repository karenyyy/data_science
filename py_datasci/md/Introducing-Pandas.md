

```python
import numpy as np
import pandas as pd
```

## The Pandas Series Object

A Pandas ``Series`` is a one-dimensional array of indexed data.
It can be created from a list or array as follows:


```python
data = pd.Series([0.25, 0.5, 0.75, 1.0])
data
```




    0    0.25
    1    0.50
    2    0.75
    3    1.00
    dtype: float64




```python
data.values
```




   array([ 0.25,  0.5 ,  0.75,  1.  ])




```python
data.index
```




   RangeIndex(start=0, stop=4, step=1)




```python
data[1]
```




   0.5




```python
data[1:3]
```




    1    0.50
    2    0.75
    dtype: float64



### ``Series`` as generalized NumPy array


```python
data = pd.Series([0.25, 0.5, 0.75, 1.0],
                 index=['a', 'b', 'c', 'd'])
data
```




    a    0.25
    b    0.50
    c    0.75
    d    1.00
    dtype: float64




```python
data['b']
```




   0.5




```python
data = pd.Series([0.25, 0.5, 0.75, 1.0],
                 index=[2, 5, 3, 7])
data
```




    2    0.25
    5    0.50
    3    0.75
    7    1.00
    dtype: float64




```python
data[5]
```




   0.5




```python
population_dict = {'California': 38332521,
                   'Texas': 26448193,
                   'New York': 19651127,
                   'Florida': 19552860,
                   'Illinois': 12882135}
population = pd.Series(population_dict)
population
```




    California    38332521
    Florida       19552860
    Illinois      12882135
    New York      19651127
    Texas         26448193
    dtype: int64




```python
population['California']
```




   38332521




```python
population['California':'Illinois']
```




    California    38332521
    Florida       19552860
    Illinois      12882135
    dtype: int64




```python
pd.Series([2, 4, 6])
```




    0    2
    1    4
    2    6
    dtype: int64




```python
pd.Series(5, index=[100, 200, 300])
```




    100    5
    200    5
    300    5
    dtype: int64




```python
pd.Series({2:'a', 1:'b', 3:'c'})
```




    1    b
    2    a
    3    c
    dtype: object




```python
pd.Series({2:'a', 1:'b', 3:'c'}, index=[3, 2])
```




    3    c
    2    a
    dtype: object



## The Pandas DataFrame Object

The ``DataFrame`` can be thought of either as a generalization of a NumPy array, or as a specialization of a Python dictionary.

### DataFrame as a generalized NumPy array
If a ``Series`` is an analog of a one-dimensional array with flexible indices, a ``DataFrame`` is an analog of a two-dimensional array with both flexible row indices and flexible column names.
Just as you might think of a two-dimensional array as an ordered sequence of aligned one-dimensional columns, you can think of a ``DataFrame`` as a sequence of aligned ``Series`` objects.


```python
area_dict = {'California': 423967, 'Texas': 695662, 'New York': 141297,
             'Florida': 170312, 'Illinois': 149995}
area = pd.Series(area_dict)
area
```




    California    423967
    Florida       170312
    Illinois      149995
    New York      141297
    Texas         695662
    dtype: int64




```python
states = pd.DataFrame({'population': population,
                       'area': area})
states
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>area</th>
      <th>population</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>California</th>
      <td>423967</td>
      <td>38332521</td>
    </tr>
    <tr>
      <th>Florida</th>
      <td>170312</td>
      <td>19552860</td>
    </tr>
    <tr>
      <th>Illinois</th>
      <td>149995</td>
      <td>12882135</td>
    </tr>
    <tr>
      <th>New York</th>
      <td>141297</td>
      <td>19651127</td>
    </tr>
    <tr>
      <th>Texas</th>
      <td>695662</td>
      <td>26448193</td>
    </tr>
  </tbody>
</table>
</div>




```python
states.index
```




   Index(['California', 'Florida', 'Illinois', 'New York', 'Texas'], dtype='object')




```python
states.columns
```




   Index(['area', 'population'], dtype='object')



Thus the ``DataFrame`` can be thought of as a generalization of a two-dimensional NumPy array, where both the rows and columns have a generalized index for accessing the data.


```python
states['area']
```




    California    423967
    Florida       170312
    Illinois      149995
    New York      141297
    Texas         695662
    Name: area, dtype: int64



### Constructing DataFrame objects

#### From a single Series object

A ``DataFrame`` is a collection of ``Series`` objects, and a single-column ``DataFrame`` can be constructed from a single ``Series``:


```python
pd.DataFrame(population, columns=['population'])
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>population</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>California</th>
      <td>38332521</td>
    </tr>
    <tr>
      <th>Florida</th>
      <td>19552860</td>
    </tr>
    <tr>
      <th>Illinois</th>
      <td>12882135</td>
    </tr>
    <tr>
      <th>New York</th>
      <td>19651127</td>
    </tr>
    <tr>
      <th>Texas</th>
      <td>26448193</td>
    </tr>
  </tbody>
</table>
</div>



#### From a list of dicts

Any list of dictionaries can be made into a ``DataFrame``.
We'll use a simple list comprehension to create some data:


```python
data = [{'a': i, 'b': 2 * i}
        for i in range(3)]
pd.DataFrame(data)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>a</th>
      <th>b</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>



Even if some keys in the dictionary are missing, Pandas will fill them in with ``NaN`` (i.e., "not a number") values:


```python
pd.DataFrame([{'a': 1, 'b': 2}, {'b': 3, 'c': 4}])
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>a</th>
      <th>b</th>
      <th>c</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.0</td>
      <td>2</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>NaN</td>
      <td>3</td>
      <td>4.0</td>
    </tr>
  </tbody>
</table>
</div>



#### From a dictionary of Series objects

As we saw before, a ``DataFrame`` can be constructed from a dictionary of ``Series`` objects as well:


```python
pd.DataFrame({'population': population,
              'area': area})
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>area</th>
      <th>population</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>California</th>
      <td>423967</td>
      <td>38332521</td>
    </tr>
    <tr>
      <th>Florida</th>
      <td>170312</td>
      <td>19552860</td>
    </tr>
    <tr>
      <th>Illinois</th>
      <td>149995</td>
      <td>12882135</td>
    </tr>
    <tr>
      <th>New York</th>
      <td>141297</td>
      <td>19651127</td>
    </tr>
    <tr>
      <th>Texas</th>
      <td>695662</td>
      <td>26448193</td>
    </tr>
  </tbody>
</table>
</div>



#### From a two-dimensional NumPy array

Given a two-dimensional array of data, we can create a ``DataFrame`` with any specified column and index names.
If omitted, an integer index will be used for each:


```python
pd.DataFrame(np.random.randn(3, 2),
             columns=['foo', 'bar'],
             index=['a', 'b', 'c'])
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>foo</th>
      <th>bar</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>a</th>
      <td>-0.119918</td>
      <td>0.169147</td>
    </tr>
    <tr>
      <th>b</th>
      <td>0.381620</td>
      <td>0.285877</td>
    </tr>
    <tr>
      <th>c</th>
      <td>-0.059211</td>
      <td>-0.776034</td>
    </tr>
  </tbody>
</table>
</div>




```python
A = np.zeros(3, dtype=[('A', 'i8'), ('B', 'f8')])
A
```




   array([(0, 0.0), (0, 0.0), (0, 0.0)], 
          dtype=[('A', '<i8'), ('B', '<f8')])




```python
pd.DataFrame(A)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>A</th>
      <th>B</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>



### Indexers: loc, iloc, and ix

These slicing and indexing conventions can be a source of confusion.
For example, if your ``Series`` has an explicit integer index, an indexing operation such as ``data[1]`` will use the explicit indices, while a slicing operation like ``data[1:3]`` will use the implicit Python-style index.


```python
data = pd.Series(['a', 'b', 'c'], index=[1, 3, 5])
data
```




    1    a
    3    b
    5    c
    dtype: object




```python
# explicit index when indexing
data[1]
```




    'a'




```python
# implicit index when slicing
data[1:3]
```




    3    b
    5    c
    dtype: object



Because of this potential confusion in the case of integer indexes, Pandas provides some special *indexer* attributes that explicitly expose certain indexing schemes.
These are not functional methods, but attributes that expose a particular slicing interface to the data in the ``Series``.

First, the ``loc`` attribute allows indexing and slicing that always references the __explicit index__:


```python
data.loc[1]
```




    'a'




```python
data.loc[1:3]
```




    1    a
    3    b
    dtype: object



The ``iloc`` attribute allows indexing and slicing that always references the __implicit Python-style index__:


```python
data.iloc[1]
```




    'b'




```python
data.iloc[1:3]
```




    3    b
    5    c
    dtype: object




```python
area = pd.Series({'California': 423967, 'Texas': 695662,
                  'New York': 141297, 'Florida': 170312,
                  'Illinois': 149995})
pop = pd.Series({'California': 38332521, 'Texas': 26448193,
                 'New York': 19651127, 'Florida': 19552860,
                 'Illinois': 12882135})
data = pd.DataFrame({'area':area, 'pop':pop})
data
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>area</th>
      <th>pop</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>California</th>
      <td>423967</td>
      <td>38332521</td>
    </tr>
    <tr>
      <th>Florida</th>
      <td>170312</td>
      <td>19552860</td>
    </tr>
    <tr>
      <th>Illinois</th>
      <td>149995</td>
      <td>12882135</td>
    </tr>
    <tr>
      <th>New York</th>
      <td>141297</td>
      <td>19651127</td>
    </tr>
    <tr>
      <th>Texas</th>
      <td>695662</td>
      <td>26448193</td>
    </tr>
  </tbody>
</table>
</div>




```python
data.area is data['area']
```




    True




```python
data.pop is data['pop']
```




    False




```python
data['density'] = data['pop'] / data['area']
data
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>area</th>
      <th>pop</th>
      <th>density</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>California</th>
      <td>423967</td>
      <td>38332521</td>
      <td>90.413926</td>
    </tr>
    <tr>
      <th>Florida</th>
      <td>170312</td>
      <td>19552860</td>
      <td>114.806121</td>
    </tr>
    <tr>
      <th>Illinois</th>
      <td>149995</td>
      <td>12882135</td>
      <td>85.883763</td>
    </tr>
    <tr>
      <th>New York</th>
      <td>141297</td>
      <td>19651127</td>
      <td>139.076746</td>
    </tr>
    <tr>
      <th>Texas</th>
      <td>695662</td>
      <td>26448193</td>
      <td>38.018740</td>
    </tr>
  </tbody>
</table>
</div>




```python
data.values
```




    array([[  4.23967000e+05,   3.83325210e+07,   9.04139261e+01],
           [  1.70312000e+05,   1.95528600e+07,   1.14806121e+02],
           [  1.49995000e+05,   1.28821350e+07,   8.58837628e+01],
           [  1.41297000e+05,   1.96511270e+07,   1.39076746e+02],
           [  6.95662000e+05,   2.64481930e+07,   3.80187404e+01]])




```python
data.T
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>California</th>
      <th>Florida</th>
      <th>Illinois</th>
      <th>New York</th>
      <th>Texas</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>area</th>
      <td>423967.000000</td>
      <td>170312.000000</td>
      <td>149995.000000</td>
      <td>141297.000000</td>
      <td>695662.00000</td>
    </tr>
    <tr>
      <th>pop</th>
      <td>38332521.000000</td>
      <td>19552860.000000</td>
      <td>12882135.000000</td>
      <td>19651127.000000</td>
      <td>26448193.00000</td>
    </tr>
    <tr>
      <th>density</th>
      <td>90.413926</td>
      <td>114.806121</td>
      <td>85.883763</td>
      <td>139.076746</td>
      <td>38.01874</td>
    </tr>
  </tbody>
</table>
</div>




```python
data.values[0]
```




    array([  4.23967000e+05,   3.83325210e+07,   9.04139261e+01])




```python
data['area']
```




    California    423967
    Florida       170312
    Illinois      149995
    New York      141297
    Texas         695662
    Name: area, dtype: int64




```python
data.iloc[:3, :2]
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>area</th>
      <th>pop</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>California</th>
      <td>423967</td>
      <td>38332521</td>
    </tr>
    <tr>
      <th>Florida</th>
      <td>170312</td>
      <td>19552860</td>
    </tr>
    <tr>
      <th>Illinois</th>
      <td>149995</td>
      <td>12882135</td>
    </tr>
  </tbody>
</table>
</div>



Similarly, using the ``loc`` indexer we can index the underlying data in an array-like style but using the explicit index and column names:


```python
data.loc[:'Illinois', :'pop']
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>area</th>
      <th>pop</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>California</th>
      <td>423967</td>
      <td>38332521</td>
    </tr>
    <tr>
      <th>Florida</th>
      <td>170312</td>
      <td>19552860</td>
    </tr>
    <tr>
      <th>Illinois</th>
      <td>149995</td>
      <td>12882135</td>
    </tr>
  </tbody>
</table>
</div>




```python
data.ix[:3, :'pop']
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>area</th>
      <th>pop</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>California</th>
      <td>423967</td>
      <td>38332521</td>
    </tr>
    <tr>
      <th>Florida</th>
      <td>170312</td>
      <td>19552860</td>
    </tr>
    <tr>
      <th>Illinois</th>
      <td>149995</td>
      <td>12882135</td>
    </tr>
  </tbody>
</table>
</div>




```python
data.loc[data.density > 100, ['pop', 'density']]
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>pop</th>
      <th>density</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Florida</th>
      <td>19552860</td>
      <td>114.806121</td>
    </tr>
    <tr>
      <th>New York</th>
      <td>19651127</td>
      <td>139.076746</td>
    </tr>
  </tbody>
</table>
</div>




```python
data.iloc[0, 2] = 90
data
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>area</th>
      <th>pop</th>
      <th>density</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>California</th>
      <td>423967</td>
      <td>38332521</td>
      <td>90.000000</td>
    </tr>
    <tr>
      <th>Florida</th>
      <td>170312</td>
      <td>19552860</td>
      <td>114.806121</td>
    </tr>
    <tr>
      <th>Illinois</th>
      <td>149995</td>
      <td>12882135</td>
      <td>85.883763</td>
    </tr>
    <tr>
      <th>New York</th>
      <td>141297</td>
      <td>19651127</td>
      <td>139.076746</td>
    </tr>
    <tr>
      <th>Texas</th>
      <td>695662</td>
      <td>26448193</td>
      <td>38.018740</td>
    </tr>
  </tbody>
</table>
</div>




```python
data['Florida':'Illinois']
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>area</th>
      <th>pop</th>
      <th>density</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Florida</th>
      <td>170312</td>
      <td>19552860</td>
      <td>114.806121</td>
    </tr>
    <tr>
      <th>Illinois</th>
      <td>149995</td>
      <td>12882135</td>
      <td>85.883763</td>
    </tr>
  </tbody>
</table>
</div>




```python
data.iloc[1:3]
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>area</th>
      <th>pop</th>
      <th>density</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Florida</th>
      <td>170312</td>
      <td>19552860</td>
      <td>114.806121</td>
    </tr>
    <tr>
      <th>Illinois</th>
      <td>149995</td>
      <td>12882135</td>
      <td>85.883763</td>
    </tr>
  </tbody>
</table>
</div>




```python
data[data.density > 100]
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>area</th>
      <th>pop</th>
      <th>density</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Florida</th>
      <td>170312</td>
      <td>19552860</td>
      <td>114.806121</td>
    </tr>
    <tr>
      <th>New York</th>
      <td>141297</td>
      <td>19651127</td>
      <td>139.076746</td>
    </tr>
  </tbody>
</table>
</div>




```python
rng = np.random.RandomState(42)
```


```python
A = pd.Series([2, 4, 6], index=[0, 1, 2])
B = pd.Series([1, 3, 5], index=[1, 2, 3])
A + B
```




    0   NaN
    1     5
    2     9
    3   NaN
    dtype: float64



If using NaN values is not the desired behavior, the fill value can be modified using appropriate object methods in place of the operators.
For example, calling ``A.add(B)`` is equivalent to calling ``A + B``, but allows optional explicit specification of the fill value for any elements in ``A`` or ``B`` that might be missing:


```python
A.add(B, fill_value=1)
```




    0    3
    1    5
    2    9
    3    6
    dtype: float64




```python
A.add(B, fill_value=2)
```




    0    4
    1    5
    2    9
    3    7
    dtype: float64




```python
A.add(B, fill_value=3)
```




    0    5
    1    5
    2    9
    3    8
    dtype: float64



### Index alignment in DataFrame

A similar type of alignment takes place for *both* columns and indices when performing operations on ``DataFrame``s:


```python
A = pd.DataFrame(rng.randint(0, 20, (2, 2)),
                 columns=list('AB'))
A
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>A</th>
      <th>B</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>6</td>
      <td>19</td>
    </tr>
    <tr>
      <th>1</th>
      <td>14</td>
      <td>10</td>
    </tr>
  </tbody>
</table>
</div>




```python
B = pd.DataFrame(rng.randint(0, 10, (3, 3)),
                 columns=list('BAC'))
B
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>B</th>
      <th>A</th>
      <th>C</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>7</td>
      <td>4</td>
      <td>6</td>
    </tr>
    <tr>
      <th>1</th>
      <td>9</td>
      <td>2</td>
      <td>6</td>
    </tr>
    <tr>
      <th>2</th>
      <td>7</td>
      <td>4</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>




```python
A + B
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>A</th>
      <th>B</th>
      <th>C</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>10</td>
      <td>26</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>16</td>
      <td>19</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
fill = A.stack().mean()
A.add(B, fill_value=fill) # fill value start from 'fill', here we have 5 NAN, fill=11, so fill with 12, 13, 14, 15, 16
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>A</th>
      <th>B</th>
      <th>C</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>10.00</td>
      <td>26.00</td>
      <td>18.25</td>
    </tr>
    <tr>
      <th>1</th>
      <td>16.00</td>
      <td>19.00</td>
      <td>18.25</td>
    </tr>
    <tr>
      <th>2</th>
      <td>16.25</td>
      <td>19.25</td>
      <td>15.25</td>
    </tr>
  </tbody>
</table>
</div>




```python
A = rng.randint(10, size=(3, 4))
A
```




    array([[5, 8, 0, 9],
           [2, 6, 3, 8],
           [2, 4, 2, 6]])




```python
A - A[0]
```




    array([[ 0,  0,  0,  0],
           [-3, -2,  3, -1],
           [-3, -4,  2, -3]])




```python
df = pd.DataFrame(A, columns=list('QRST'))
df
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Q</th>
      <th>R</th>
      <th>S</th>
      <th>T</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5</td>
      <td>8</td>
      <td>0</td>
      <td>9</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>6</td>
      <td>3</td>
      <td>8</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>4</td>
      <td>2</td>
      <td>6</td>
    </tr>
  </tbody>
</table>
</div>




```python
df=df - df.iloc[0]
df
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Q</th>
      <th>R</th>
      <th>S</th>
      <th>T</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-3</td>
      <td>-2</td>
      <td>3</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-3</td>
      <td>-4</td>
      <td>2</td>
      <td>-3</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.subtract(df['R'], axis=0)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Q</th>
      <th>R</th>
      <th>S</th>
      <th>T</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-1</td>
      <td>0</td>
      <td>5</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>0</td>
      <td>6</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
halfrow = df.iloc[0, ::2]
halfrow
```




    Q    0
    S    0
    Name: 0, dtype: int64




```python
df - halfrow
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Q</th>
      <th>R</th>
      <th>S</th>
      <th>T</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>NaN</td>
      <td>0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-3</td>
      <td>NaN</td>
      <td>3</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-3</td>
      <td>NaN</td>
      <td>2</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




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
