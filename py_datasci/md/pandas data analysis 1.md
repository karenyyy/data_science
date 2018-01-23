

```python
# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import matplotlib

%matplotlib inline
```


```python
# set seed
np.random.seed(111)
def CreateDataSet(Number=1):
    
    Output = []
    
    for i in range(Number):
        
        # Create a weekly (mondays) date range
        rng = pd.date_range(start='1/1/2009', end='12/31/2012', freq='W-MON')
        
        # Create random data
        data = np.random.randint(low=25,high=1000,size=len(rng))
        
        # Status pool
        status = [1,2,3]
        
        # Make a random list of statuses
        random_status = [status[np.random.randint(low=0,high=len(status))] for i in range(len(rng))]
        
        # State pool
        states = ['GA','FL','fl','NY','NJ','TX']
        
        # Make a random list of states 
        random_states = [states[np.random.randint(low=0,high=len(states))] for i in range(len(rng))]
    
        Output.extend(zip(random_states, random_status, data, rng)) # vstack
        
    return Output
```


```python
dataset = CreateDataSet(2)
df = pd.DataFrame(data=dataset, columns=['State','Status','CustomerCount','StatusDate'])
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 418 entries, 0 to 417
    Data columns (total 4 columns):
    State            418 non-null object
    Status           418 non-null int64
    CustomerCount    418 non-null int64
    StatusDate       418 non-null datetime64[ns]
    dtypes: datetime64[ns](1), int64(2), object(1)
    memory usage: 16.3+ KB



```python
df.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>State</th>
      <th>Status</th>
      <th>CustomerCount</th>
      <th>StatusDate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>GA</td>
      <td>1</td>
      <td>877</td>
      <td>2009-01-05</td>
    </tr>
    <tr>
      <th>1</th>
      <td>FL</td>
      <td>1</td>
      <td>901</td>
      <td>2009-01-12</td>
    </tr>
    <tr>
      <th>2</th>
      <td>fl</td>
      <td>3</td>
      <td>749</td>
      <td>2009-01-19</td>
    </tr>
    <tr>
      <th>3</th>
      <td>FL</td>
      <td>3</td>
      <td>111</td>
      <td>2009-01-26</td>
    </tr>
    <tr>
      <th>4</th>
      <td>GA</td>
      <td>1</td>
      <td>300</td>
      <td>2009-02-02</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Save results to excel
df.to_csv('tocsv.csv')
df.to_excel('toexcel.xlsx', index=False)
```


```python
df = pd.read_excel('toexcel.xlsx', 0, index_col='StatusDate')
df.dtypes
```




    State            object
    Status            int64
    CustomerCount     int64
    dtype: object




```python
df.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>State</th>
      <th>Status</th>
      <th>CustomerCount</th>
    </tr>
    <tr>
      <th>StatusDate</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2009-01-05</th>
      <td>GA</td>
      <td>1</td>
      <td>877</td>
    </tr>
    <tr>
      <th>2009-01-12</th>
      <td>FL</td>
      <td>1</td>
      <td>901</td>
    </tr>
    <tr>
      <th>2009-01-19</th>
      <td>fl</td>
      <td>3</td>
      <td>749</td>
    </tr>
    <tr>
      <th>2009-01-26</th>
      <td>FL</td>
      <td>3</td>
      <td>111</td>
    </tr>
    <tr>
      <th>2009-02-02</th>
      <td>GA</td>
      <td>1</td>
      <td>300</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.index
```




    DatetimeIndex(['2009-01-05', '2009-01-12', '2009-01-19', '2009-01-26',
                   '2009-02-02', '2009-02-09', '2009-02-16', '2009-02-23',
                   '2009-03-02', '2009-03-09',
                   ...
                   '2012-10-29', '2012-11-05', '2012-11-12', '2012-11-19',
                   '2012-11-26', '2012-12-03', '2012-12-10', '2012-12-17',
                   '2012-12-24', '2012-12-31'],
                  dtype='datetime64[ns]', name='StatusDate', length=418, freq=None)




```python
df2=pd.read_csv('tocsv.csv', index_col='StatusDate')[['State', 'Status', 'CustomerCount']]
df2.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>State</th>
      <th>Status</th>
      <th>CustomerCount</th>
    </tr>
    <tr>
      <th>StatusDate</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2009-01-05</th>
      <td>GA</td>
      <td>1</td>
      <td>877</td>
    </tr>
    <tr>
      <th>2009-01-12</th>
      <td>FL</td>
      <td>1</td>
      <td>901</td>
    </tr>
    <tr>
      <th>2009-01-19</th>
      <td>fl</td>
      <td>3</td>
      <td>749</td>
    </tr>
    <tr>
      <th>2009-01-26</th>
      <td>FL</td>
      <td>3</td>
      <td>111</td>
    </tr>
    <tr>
      <th>2009-02-02</th>
      <td>GA</td>
      <td>1</td>
      <td>300</td>
    </tr>
  </tbody>
</table>
</div>



# Prepare Data  

This section attempts to clean up the data for analysis.  
1. Make sure the state column is all in upper case  
2. Only select records where the account status is equal to "1"  
3. Merge (NJ and NY) to NY in the state column  
4. Remove any outliers (any odd results in the data set)


Lets take a quick look on how some of the *State* values are upper case and some are lower case


```python
df['State'].unique()
```




    array(['GA', 'FL', 'fl', 'TX', 'NY', 'NJ'], dtype=object)



To convert all the State values to upper case we will use the ***upper()*** function and the dataframe's ***apply*** attribute. The ***lambda*** function simply will apply the upper function to each value in the *State* column.


```python
# Clean State Column, convert to upper case
df['State'] = df.State.apply(lambda x: x.upper())
```


```python
df['State'].unique()
```




    array(['GA', 'FL', 'TX', 'NY', 'NJ'], dtype=object)




```python
# Only grab where Status == 1
mask = df['Status'] == 1
df = df[mask]
```

To turn the ***NJ*** states to ***NY*** we simply...  

***[df.State == 'NJ']*** - Find all records in the *State* column where they are equal to *NJ*.  
***df.State[df.State == 'NJ'] = 'NY'*** - For all records in the *State* column where they are equal to *NJ*, replace them with *NY*.


```python
# Convert NJ to NY
mask = df.State == 'NJ'
df['State'][mask] = 'NY'
```


```python
df['State'].unique()
```




    array(['GA', 'FL', 'NY', 'TX'], dtype=object)




```python
df['CustomerCount'].plot(figsize=(15,5));
```

    /usr/local/lib/python3.5/dist-packages/matplotlib/__init__.py:830: MatplotlibDeprecationWarning: axes.color_cycle is deprecated and replaced with axes.prop_cycle; please use the latter.
      mplDeprecation)



![png](output_19_1.png)



```python
sortdf = df[df['State']=='NY'].sort_values(by='CustomerCount', ascending=False)
sortdf
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>State</th>
      <th>Status</th>
      <th>CustomerCount</th>
    </tr>
    <tr>
      <th>StatusDate</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2009-03-09</th>
      <td>NY</td>
      <td>1</td>
      <td>992</td>
    </tr>
    <tr>
      <th>2009-07-27</th>
      <td>NY</td>
      <td>1</td>
      <td>990</td>
    </tr>
    <tr>
      <th>2009-05-11</th>
      <td>NY</td>
      <td>1</td>
      <td>987</td>
    </tr>
    <tr>
      <th>2011-04-18</th>
      <td>NY</td>
      <td>1</td>
      <td>973</td>
    </tr>
    <tr>
      <th>2012-04-02</th>
      <td>NY</td>
      <td>1</td>
      <td>971</td>
    </tr>
    <tr>
      <th>2011-08-29</th>
      <td>NY</td>
      <td>1</td>
      <td>956</td>
    </tr>
    <tr>
      <th>2011-12-05</th>
      <td>NY</td>
      <td>1</td>
      <td>941</td>
    </tr>
    <tr>
      <th>2011-03-07</th>
      <td>NY</td>
      <td>1</td>
      <td>926</td>
    </tr>
    <tr>
      <th>2011-04-18</th>
      <td>NY</td>
      <td>1</td>
      <td>894</td>
    </tr>
    <tr>
      <th>2011-08-29</th>
      <td>NY</td>
      <td>1</td>
      <td>890</td>
    </tr>
    <tr>
      <th>2010-10-04</th>
      <td>NY</td>
      <td>1</td>
      <td>843</td>
    </tr>
    <tr>
      <th>2009-04-20</th>
      <td>NY</td>
      <td>1</td>
      <td>820</td>
    </tr>
    <tr>
      <th>2010-10-18</th>
      <td>NY</td>
      <td>1</td>
      <td>816</td>
    </tr>
    <tr>
      <th>2009-09-14</th>
      <td>NY</td>
      <td>1</td>
      <td>772</td>
    </tr>
    <tr>
      <th>2009-04-27</th>
      <td>NY</td>
      <td>1</td>
      <td>753</td>
    </tr>
    <tr>
      <th>2012-11-26</th>
      <td>NY</td>
      <td>1</td>
      <td>748</td>
    </tr>
    <tr>
      <th>2012-09-10</th>
      <td>NY</td>
      <td>1</td>
      <td>747</td>
    </tr>
    <tr>
      <th>2010-09-06</th>
      <td>NY</td>
      <td>1</td>
      <td>708</td>
    </tr>
    <tr>
      <th>2009-10-12</th>
      <td>NY</td>
      <td>1</td>
      <td>694</td>
    </tr>
    <tr>
      <th>2011-06-27</th>
      <td>NY</td>
      <td>1</td>
      <td>688</td>
    </tr>
    <tr>
      <th>2010-01-25</th>
      <td>NY</td>
      <td>1</td>
      <td>640</td>
    </tr>
    <tr>
      <th>2010-10-11</th>
      <td>NY</td>
      <td>1</td>
      <td>629</td>
    </tr>
    <tr>
      <th>2009-08-31</th>
      <td>NY</td>
      <td>1</td>
      <td>602</td>
    </tr>
    <tr>
      <th>2011-02-07</th>
      <td>NY</td>
      <td>1</td>
      <td>588</td>
    </tr>
    <tr>
      <th>2010-05-24</th>
      <td>NY</td>
      <td>1</td>
      <td>538</td>
    </tr>
    <tr>
      <th>2009-01-19</th>
      <td>NY</td>
      <td>1</td>
      <td>522</td>
    </tr>
    <tr>
      <th>2012-12-10</th>
      <td>NY</td>
      <td>1</td>
      <td>500</td>
    </tr>
    <tr>
      <th>2012-07-02</th>
      <td>NY</td>
      <td>1</td>
      <td>491</td>
    </tr>
    <tr>
      <th>2012-07-09</th>
      <td>NY</td>
      <td>1</td>
      <td>461</td>
    </tr>
    <tr>
      <th>2009-04-27</th>
      <td>NY</td>
      <td>1</td>
      <td>447</td>
    </tr>
    <tr>
      <th>2009-05-25</th>
      <td>NY</td>
      <td>1</td>
      <td>378</td>
    </tr>
    <tr>
      <th>2010-04-12</th>
      <td>NY</td>
      <td>1</td>
      <td>375</td>
    </tr>
    <tr>
      <th>2009-03-16</th>
      <td>NY</td>
      <td>1</td>
      <td>355</td>
    </tr>
    <tr>
      <th>2009-09-28</th>
      <td>NY</td>
      <td>1</td>
      <td>349</td>
    </tr>
    <tr>
      <th>2012-09-03</th>
      <td>NY</td>
      <td>1</td>
      <td>344</td>
    </tr>
    <tr>
      <th>2011-02-28</th>
      <td>NY</td>
      <td>1</td>
      <td>336</td>
    </tr>
    <tr>
      <th>2010-07-26</th>
      <td>NY</td>
      <td>1</td>
      <td>314</td>
    </tr>
    <tr>
      <th>2010-09-27</th>
      <td>NY</td>
      <td>1</td>
      <td>307</td>
    </tr>
    <tr>
      <th>2010-08-02</th>
      <td>NY</td>
      <td>1</td>
      <td>261</td>
    </tr>
    <tr>
      <th>2011-08-01</th>
      <td>NY</td>
      <td>1</td>
      <td>244</td>
    </tr>
    <tr>
      <th>2012-10-22</th>
      <td>NY</td>
      <td>1</td>
      <td>203</td>
    </tr>
    <tr>
      <th>2011-03-28</th>
      <td>NY</td>
      <td>1</td>
      <td>198</td>
    </tr>
    <tr>
      <th>2012-07-30</th>
      <td>NY</td>
      <td>1</td>
      <td>169</td>
    </tr>
    <tr>
      <th>2011-08-15</th>
      <td>NY</td>
      <td>1</td>
      <td>148</td>
    </tr>
    <tr>
      <th>2011-09-26</th>
      <td>NY</td>
      <td>1</td>
      <td>140</td>
    </tr>
    <tr>
      <th>2012-03-19</th>
      <td>NY</td>
      <td>1</td>
      <td>91</td>
    </tr>
    <tr>
      <th>2010-08-30</th>
      <td>NY</td>
      <td>1</td>
      <td>79</td>
    </tr>
    <tr>
      <th>2011-01-03</th>
      <td>NY</td>
      <td>1</td>
      <td>51</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.reset_index().head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>StatusDate</th>
      <th>State</th>
      <th>Status</th>
      <th>CustomerCount</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2009-01-05</td>
      <td>GA</td>
      <td>1</td>
      <td>877</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2009-01-12</td>
      <td>FL</td>
      <td>1</td>
      <td>901</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2009-02-02</td>
      <td>GA</td>
      <td>1</td>
      <td>300</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2009-03-09</td>
      <td>NY</td>
      <td>1</td>
      <td>992</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2009-04-06</td>
      <td>FL</td>
      <td>1</td>
      <td>291</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Group by State and StatusDate
Daily = df.reset_index().groupby(['State','StatusDate']).aggregate(sum)
Daily.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>Status</th>
      <th>CustomerCount</th>
    </tr>
    <tr>
      <th>State</th>
      <th>StatusDate</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="5" valign="top">FL</th>
      <th>2009-01-12</th>
      <td>1</td>
      <td>901</td>
    </tr>
    <tr>
      <th>2009-04-06</th>
      <td>1</td>
      <td>291</td>
    </tr>
    <tr>
      <th>2009-07-06</th>
      <td>1</td>
      <td>723</td>
    </tr>
    <tr>
      <th>2009-07-20</th>
      <td>1</td>
      <td>710</td>
    </tr>
    <tr>
      <th>2009-08-24</th>
      <td>1</td>
      <td>991</td>
    </tr>
  </tbody>
</table>
</div>




```python
del Daily['Status']
Daily.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>CustomerCount</th>
    </tr>
    <tr>
      <th>State</th>
      <th>StatusDate</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="5" valign="top">FL</th>
      <th>2009-01-12</th>
      <td>901</td>
    </tr>
    <tr>
      <th>2009-04-06</th>
      <td>291</td>
    </tr>
    <tr>
      <th>2009-07-06</th>
      <td>723</td>
    </tr>
    <tr>
      <th>2009-07-20</th>
      <td>710</td>
    </tr>
    <tr>
      <th>2009-08-24</th>
      <td>991</td>
    </tr>
  </tbody>
</table>
</div>




```python
# What is the index of the dataframe
Daily.index[0]
```




    ('FL', Timestamp('2009-01-12 00:00:00'))




```python
# Select the State index
Daily.index.levels[0]
```




    Index(['FL', 'GA', 'NY', 'TX'], dtype='object', name='State')




```python
# Select the StatusDate index
Daily.index.levels[1]
```




    DatetimeIndex(['2009-01-05', '2009-01-12', '2009-01-19', '2009-02-02',
                   '2009-03-09', '2009-03-16', '2009-03-23', '2009-04-06',
                   '2009-04-20', '2009-04-27',
                   ...
                   '2012-09-03', '2012-09-10', '2012-09-17', '2012-09-24',
                   '2012-10-22', '2012-10-29', '2012-11-12', '2012-11-19',
                   '2012-11-26', '2012-12-10'],
                  dtype='datetime64[ns]', name='StatusDate', length=118, freq=None)




```python
Daily.loc['FL'].plot()
Daily.loc['GA'].plot()
Daily.loc['NY'].plot()
Daily.loc['TX'].plot();
```

    /usr/local/lib/python3.5/dist-packages/matplotlib/__init__.py:830: MatplotlibDeprecationWarning: axes.color_cycle is deprecated and replaced with axes.prop_cycle; please use the latter.
      mplDeprecation)



![png](output_27_1.png)



![png](output_27_2.png)



![png](output_27_3.png)



![png](output_27_4.png)



```python
Daily.loc['FL']['2012':].plot()
Daily.loc['GA']['2012':].plot()
Daily.loc['NY']['2012':].plot()
Daily.loc['TX']['2012':].plot();
```

    /usr/local/lib/python3.5/dist-packages/matplotlib/__init__.py:830: MatplotlibDeprecationWarning: axes.color_cycle is deprecated and replaced with axes.prop_cycle; please use the latter.
      mplDeprecation)



![png](output_28_1.png)



![png](output_28_2.png)



![png](output_28_3.png)



![png](output_28_4.png)



```python
Daily.index.get_level_values(0)
```




    Index(['FL', 'FL', 'FL', 'FL', 'FL', 'FL', 'FL', 'FL', 'FL', 'FL',
           ...
           'TX', 'TX', 'TX', 'TX', 'TX', 'TX', 'TX', 'TX', 'TX', 'TX'],
          dtype='object', name='State', length=138)




```python
Daily.index.get_level_values(1)
```




    DatetimeIndex(['2009-01-12', '2009-04-06', '2009-07-06', '2009-07-20',
                   '2009-08-24', '2009-09-07', '2009-09-21', '2009-09-28',
                   '2009-10-05', '2010-01-04',
                   ...
                   '2011-09-12', '2011-10-03', '2011-11-07', '2011-12-26',
                   '2012-01-02', '2012-01-09', '2012-02-27', '2012-03-12',
                   '2012-09-03', '2012-10-29'],
                  dtype='datetime64[ns]', name='StatusDate', length=138, freq=None)




```python
# Calculate Outliers
StateYearMonth = Daily.groupby([Daily.index.get_level_values(0), Daily.index.get_level_values(1).year, Daily.index.get_level_values(1).month])
Daily['Lower'] = StateYearMonth['CustomerCount'].transform( lambda x: x.quantile(q=.25) - (1.5*x.quantile(q=.75)-x.quantile(q=.25)) )
Daily['Upper'] = StateYearMonth['CustomerCount'].transform( lambda x: x.quantile(q=.75) + (1.5*x.quantile(q=.75)-x.quantile(q=.25)) )
Daily['Outlier'] = (Daily['CustomerCount'] < Daily['Lower']) | (Daily['CustomerCount'] > Daily['Upper']) 
```


```python
(Daily['Outlier'] == False).count()
```




    138




```python
Daily = Daily[Daily['Outlier'] == False]
```


```python
Daily.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>CustomerCount</th>
      <th>Lower</th>
      <th>Upper</th>
      <th>Outlier</th>
    </tr>
    <tr>
      <th>State</th>
      <th>StatusDate</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="5" valign="top">FL</th>
      <th>2009-01-12</th>
      <td>901</td>
      <td>450.500</td>
      <td>1351.500</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2009-04-06</th>
      <td>291</td>
      <td>145.500</td>
      <td>436.500</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2009-07-06</th>
      <td>723</td>
      <td>346.875</td>
      <td>1086.125</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2009-07-20</th>
      <td>710</td>
      <td>346.875</td>
      <td>1086.125</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2009-08-24</th>
      <td>991</td>
      <td>495.500</td>
      <td>1486.500</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>




```python
ALL = pd.DataFrame(Daily['CustomerCount'].groupby(Daily.index.get_level_values(1)).aggregate(sum))
ALL.columns = ['CustomerCount'] # rename column
ALL.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CustomerCount</th>
    </tr>
    <tr>
      <th>StatusDate</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2009-01-05</th>
      <td>877</td>
    </tr>
    <tr>
      <th>2009-01-12</th>
      <td>901</td>
    </tr>
    <tr>
      <th>2009-01-19</th>
      <td>522</td>
    </tr>
    <tr>
      <th>2009-02-02</th>
      <td>300</td>
    </tr>
    <tr>
      <th>2009-03-09</th>
      <td>992</td>
    </tr>
  </tbody>
</table>
</div>




```python
# # Group by Year and Month
YearMonth = ALL.groupby([lambda x: x.year, lambda x: x.month])

# # What is the max customer count per Year and Month
ALL['Max'] = YearMonth['CustomerCount'].transform(lambda x: x.max())
ALL.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CustomerCount</th>
      <th>Max</th>
    </tr>
    <tr>
      <th>StatusDate</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2009-01-05</th>
      <td>877</td>
      <td>901</td>
    </tr>
    <tr>
      <th>2009-01-12</th>
      <td>901</td>
      <td>901</td>
    </tr>
    <tr>
      <th>2009-01-19</th>
      <td>522</td>
      <td>901</td>
    </tr>
    <tr>
      <th>2009-02-02</th>
      <td>300</td>
      <td>300</td>
    </tr>
    <tr>
      <th>2009-03-09</th>
      <td>992</td>
      <td>992</td>
    </tr>
  </tbody>
</table>
</div>



----------------------------------  
There is also an interest to gauge if the current customer counts were reaching certain goals the company had established. The task here is to visually show if the current customer counts are meeting the goals listed below. We will call the goals ***BHAG*** (Big Hairy Annual Goal).  

* 12/31/2011 - 1,000 customers  
* 12/31/2012 - 2,000 customers  
* 12/31/2013 - 3,000 customers  

We will be using the **date_range** function to create our dates.  

***Definition:*** date_range(start=None, end=None, periods=None, freq='D', tz=None, normalize=False, name=None, closed=None)  

__frequency: A (or annual)__


```python
# Create the BHAG dataframe
data = [1000,2000,3000]
idx = pd.date_range(start='12/31/2011', end='12/31/2013', freq="A")
BHAG = pd.DataFrame(data, index=idx, columns=['BHAG'])
BHAG
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>BHAG</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2011-12-31</th>
      <td>1000</td>
    </tr>
    <tr>
      <th>2012-12-31</th>
      <td>2000</td>
    </tr>
    <tr>
      <th>2013-12-31</th>
      <td>3000</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Combine the BHAG and the ALL data set 
combined = pd.concat([ALL,BHAG], axis=0)
combined.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>BHAG</th>
      <th>CustomerCount</th>
      <th>Max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2009-01-05</th>
      <td>NaN</td>
      <td>877</td>
      <td>901</td>
    </tr>
    <tr>
      <th>2009-01-12</th>
      <td>NaN</td>
      <td>901</td>
      <td>901</td>
    </tr>
    <tr>
      <th>2009-01-19</th>
      <td>NaN</td>
      <td>522</td>
      <td>901</td>
    </tr>
    <tr>
      <th>2009-02-02</th>
      <td>NaN</td>
      <td>300</td>
      <td>300</td>
    </tr>
    <tr>
      <th>2009-03-09</th>
      <td>NaN</td>
      <td>992</td>
      <td>992</td>
    </tr>
  </tbody>
</table>
</div>




```python
combined = combined.sort_index(axis=0)
combined.tail()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>BHAG</th>
      <th>CustomerCount</th>
      <th>Max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2012-11-19</th>
      <td>NaN</td>
      <td>136</td>
      <td>963</td>
    </tr>
    <tr>
      <th>2012-11-26</th>
      <td>NaN</td>
      <td>748</td>
      <td>963</td>
    </tr>
    <tr>
      <th>2012-12-10</th>
      <td>NaN</td>
      <td>500</td>
      <td>500</td>
    </tr>
    <tr>
      <th>2012-12-31</th>
      <td>2000</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2013-12-31</th>
      <td>3000</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
fig, axes = plt.subplots(figsize=(15, 5))

combined['BHAG'].fillna(method='pad').plot(color='green', label='BHAG')
combined['Max'].plot(color='blue', label='All Markets')
plt.legend(loc='best');
```


![png](output_41_0.png)



```python
# Group by Year and then get the max value per year
Year = combined.groupby(lambda x: x.year).median() # or .min() .max()
Year
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>BHAG</th>
      <th>CustomerCount</th>
      <th>Max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2009</th>
      <td>NaN</td>
      <td>710.0</td>
      <td>990</td>
    </tr>
    <tr>
      <th>2010</th>
      <td>NaN</td>
      <td>629.0</td>
      <td>792</td>
    </tr>
    <tr>
      <th>2011</th>
      <td>1000</td>
      <td>479.5</td>
      <td>926</td>
    </tr>
    <tr>
      <th>2012</th>
      <td>2000</td>
      <td>476.0</td>
      <td>888</td>
    </tr>
    <tr>
      <th>2013</th>
      <td>3000</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Add a column representing the percent change per year
Year['YR_PCT_Change'] = Year['Max'].pct_change(periods=1)
Year
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>BHAG</th>
      <th>CustomerCount</th>
      <th>Max</th>
      <th>YR_PCT_Change</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2009</th>
      <td>NaN</td>
      <td>710.0</td>
      <td>990</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2010</th>
      <td>NaN</td>
      <td>629.0</td>
      <td>792</td>
      <td>-0.200000</td>
    </tr>
    <tr>
      <th>2011</th>
      <td>1000</td>
      <td>479.5</td>
      <td>926</td>
      <td>0.169192</td>
    </tr>
    <tr>
      <th>2012</th>
      <td>2000</td>
      <td>476.0</td>
      <td>888</td>
      <td>-0.041037</td>
    </tr>
    <tr>
      <th>2013</th>
      <td>3000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
interest=Year.ix[2012,'YR_PCT_Change']
interest
```




    -0.041036717062635009




```python
(1 + interest) * Year.loc[2012,'Max']
```




    851.55939524838016




```python
# First Graph
ALL['Max'].plot(figsize=(10, 5));plt.title('ALL Markets')

# Last four Graphs
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(20, 10))
fig.subplots_adjust(hspace=1.0) ## Create space between plots

Daily.loc['FL']['CustomerCount']['2012':].fillna(method='pad').plot(ax=axes[0,0])
Daily.loc['GA']['CustomerCount']['2012':].fillna(method='pad').plot(ax=axes[0,1]) 
Daily.loc['TX']['CustomerCount']['2012':].fillna(method='pad').plot(ax=axes[1,0]) 
Daily.loc['NY']['CustomerCount']['2012':].fillna(method='pad').plot(ax=axes[1,1]) 

# Add titles
axes[0,0].set_title('Florida')
axes[0,1].set_title('Georgia')
axes[1,0].set_title('Texas')
axes[1,1].set_title('North East');
```

    /usr/local/lib/python3.5/dist-packages/matplotlib/__init__.py:830: MatplotlibDeprecationWarning: axes.color_cycle is deprecated and replaced with axes.prop_cycle; please use the latter.
      mplDeprecation)



![png](output_46_1.png)



![png](output_46_2.png)

