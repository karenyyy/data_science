

```python
import pandas as pd
```

# 1.1 Reading data from a csv file

[original page](http://donnees.ville.montreal.qc.ca/dataset/velos-comptage) (in French), but it's already included in this repository.

This dataset is a list of how many people were on 7 different bike paths in Montreal, each day.


```python
df = pd.read_csv('/home/karen/Downloads/data/bikes.csv')
```


```python
# Look at the first 3 rows
df[:4]
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Date</th>
      <th>Unnamed: 1</th>
      <th>Berri1</th>
      <th>Maisonneuve_1</th>
      <th>Maisonneuve_2</th>
      <th>Brébeuf</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>01/01/2009</td>
      <td>00:00</td>
      <td>29</td>
      <td>20</td>
      <td>35</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>02/01/2009</td>
      <td>00:00</td>
      <td>19</td>
      <td>3</td>
      <td>22</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>03/01/2009</td>
      <td>00:00</td>
      <td>24</td>
      <td>12</td>
      <td>22</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>04/01/2009</td>
      <td>00:00</td>
      <td>24</td>
      <td>8</td>
      <td>15</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



# 1.2 Selecting a column


```python
df['Berri1']
```




    0        29
    1        19
    2        24
    3        24
    4       120
    5       261
    6        60
    7        24
    8        35
    9        81
    10      318
    11      105
    12      168
    13      145
    14      131
    15       93
    16       25
    17       52
    18      136
    19      147
    20      109
    21      172
    22      148
    23       15
    24       35
    25       93
    26      209
    27       92
    28      110
    29      105
           ... 
    335    1377
    336     606
    337    1108
    338     594
    339     501
    340     669
    341     570
    342     185
    343     219
    344     194
    345     106
    346     130
    347     271
    348     308
    349     296
    350     239
    351     214
    352     133
    353     135
    354     239
    355     207
    356     158
    357      74
    358      34
    359      40
    360      66
    361      61
    362      89
    363      76
    364      53
    Name: Berri1, dtype: int64



# 1.3 Plotting a column


```python
df['Berri1'].plot(figsize=(15,10))
```


![png](https://raw.githubusercontent.com/karenyyy/data_science/master/py_datasci/images/output_8_3.png)



```python
df.plot(figsize=(15, 10))
```


![png](https://raw.githubusercontent.com/karenyyy/data_science/master/py_datasci/images/output_9_3.png)

